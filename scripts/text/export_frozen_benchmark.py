#!/usr/bin/env python3
"""Copy frozen FineMotion lists and audit current captions into a NEW directory.

Historical memberships are copied byte-for-byte. Caption regex matches are an
audit column, never a replacement for the frozen lists. No motion is copied.
The taxonomy module must be a trusted copy of semantic_subset_config_v3.py.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys

STYLE_CLASSES = ('impact', 'strike', 'balance', 'tremor')
CONTENT_CLASSES = ('walk', 'run', 'turn', 'circle')
ATTRIBUTES = ('periodic', 'secondary_rich')
PARTITIONS = ('all', 'train', 'val', 'test')
EXPECTED_TAXONOMY_SHA256 = '837a60d651c6ec05f5d594a73923cdc36f404eb6326da86e25dd347a062d7d0f'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_ids(path: Path) -> list[str]:
    ids = [s.strip() for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    for mid in ids:
        # Original MotionX filenames contain commas and other punctuation.
        # Each ID must remain one filename component, without path traversal.
        if (mid in ('.', '..') or any(char in mid for char in ('/', '\\'))
                or any(ord(char) < 32 or ord(char) == 127 for char in mid)
                or re.match(r'^[A-Za-z]:', mid)):
            raise ValueError(f'Unsafe clip ID in {path}: {mid!r}')
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate clip IDs in {path}')
    return ids


def load_partitions(root: Path) -> dict[str, list[str]]:
    return {split: read_ids(root / f'{split}.txt') for split in PARTITIONS}


def split_map(parts: dict[str, list[str]]) -> dict[str, str]:
    result = {}
    for split in ('train', 'val', 'test'):
        for mid in parts[split]:
            if mid in result:
                raise ValueError(f'{mid} belongs to both {result[mid]} and {split}')
            result[mid] = split
    outside = set(result) - set(parts['all'])
    if outside:
        raise ValueError(f'Partition contains IDs missing from all.txt: {sorted(outside)[:10]}')
    return result


def trusted_taxonomy(path: Path):
    if sha256(path) != EXPECTED_TAXONOMY_SHA256:
        raise ValueError('Taxonomy hash differs from the audited historical v3 configuration')
    spec = importlib.util.spec_from_file_location('_finemotion_frozen_taxonomy', path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Cannot load taxonomy: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous_bytecode_policy = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode_policy
    if tuple(b.key for b in module.BUCKETS_V3) != STYLE_CLASSES:
        raise ValueError('Taxonomy class order differs from the historical v3 configuration')
    return module


def source_name(mid: str) -> str:
    for prefix, source in [('bandai_30hz__', 'bandai'), ('lafan1_30hz__', 'lafan1'),
                           ('motionx_clean_30hz__', 'motionx_clean'), ('motionx_30hz__', 'motionx'),
                           ('motorica_30hz__', 'motorica'), ('finedance_30hz__', 'finedance')]:
        if mid.startswith(prefix):
            return source
    if re.fullmatch(r'M?\d+(?:__s\d{6}_e\d{6})?', mid):
        return 'humanml3d'
    return 'unknown'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--semantic-root', required=True, type=Path,
                    help='Frozen splits_sem_v3-1 root; use the March SOTA root for paper statistics.')
    ap.add_argument('--content-root', type=Path,
                    help='Optional root containing walk/run/turn/circle folders, e.g. January hfcontent_v0_4class.')
    ap.add_argument('--base-splits', required=True, type=Path)
    ap.add_argument('--training-splits', type=Path, help='Optional pose69-valid all/train/val/test lists.')
    ap.add_argument('--texts-dir', required=True, type=Path)
    ap.add_argument('--taxonomy-config', required=True, type=Path)
    ap.add_argument('--keep-ids', type=Path)
    ap.add_argument('--paper-examples', type=Path,
                    help='Optional figure manifest; copied separately without changing captions or evaluation lists.')
    ap.add_argument('--out-dir', required=True, type=Path, help='Must not exist and must be outside all input roots.')
    ap.add_argument('--release-id', default='frozen_benchmark_v1')
    args = ap.parse_args()

    sources = [args.semantic_root, args.base_splits, args.texts_dir,
               args.taxonomy_config, args.content_root, args.training_splits,
               args.keep_ids, args.paper_examples]
    sources = [p.resolve(strict=True) for p in sources if p is not None]
    out = args.out_dir.resolve()
    if args.out_dir.exists() or args.out_dir.is_symlink() or out.exists():
        raise FileExistsError(f'Refusing to reuse output path: {args.out_dir}')
    for source in sources:
        if out == source or source in out.parents or out in source.parents:
            raise ValueError(f'Output overlaps an input path: {source}')

    style_root = args.semantic_root / 'single_strict'
    attr_root = args.semantic_root / 'attrs'
    if not style_root.is_dir():
        style_root = args.semantic_root / 'style/single_strict'
    if not attr_root.is_dir():
        attr_root = args.semantic_root / 'style/attrs'
    views = {'style': (style_root, STYLE_CLASSES), 'attributes': (attr_root, ATTRIBUTES)}
    if args.content_root:
        views['content'] = (args.content_root, CONTENT_CLASSES)
    frozen = {}
    for view, (root, classes) in views.items():
        frozen[view] = {label: load_partitions(root / label) for label in classes}
        for parts in frozen[view].values():
            split_map(parts)
    base = load_partitions(args.base_splits)
    base_map = split_map(base)
    training = load_partitions(args.training_splits) if args.training_splits else None
    training_map = split_map(training) if training else {}
    keep = set(read_ids(args.keep_ids)) if args.keep_ids else None
    taxonomy = trusted_taxonomy(args.taxonomy_config)
    patterns = {'style': {}, 'attributes': {}}
    for view, groups in [('style', taxonomy.BUCKETS_V3), ('attributes', taxonomy.ATTRIBUTES_V3)]:
        for group in groups:
            patterns[view][group.key] = [(f'{view}.{group.key}.{i:03d}', pattern,
                                         re.compile(pattern, flags=re.IGNORECASE))
                                        for i, pattern in enumerate(group.patterns)]

    all_ids = set(base['all'])
    for groups in frozen.values():
        for parts in groups.values():
            all_ids.update(parts['all'])
    if training:
        all_ids.update(training['all'])
    frozen_sets = {view: {label: {s: set(ids) for s, ids in parts.items()}
                          for label, parts in groups.items()} for view, groups in frozen.items()}

    # Exclusive mkdir is the only creation of the output root. No overwrite/delete path exists.
    out.mkdir(parents=True, exist_ok=False)
    copied = []

    def copy_file(source: Path, relative: str):
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        info = {'source': str(source.resolve()), 'output': relative, 'sha256': sha256(source),
                'bytes': source.stat().st_size}
        if source.suffix == '.txt':
            info['count'] = len(read_ids(source))
        if sha256(target) != info['sha256']:
            raise RuntimeError(f'Copy hash mismatch: {source}')
        copied.append(info)

    for view, (root, classes) in views.items():
        for label in classes:
            for split in (*PARTITIONS, 'train_val'):
                source = root / label / f'{split}.txt'
                if source.is_file():
                    copy_file(source, f'lists/{view}/{label}/{split}.txt')
    for split in PARTITIONS:
        copy_file(args.base_splits / f'{split}.txt', f'lists/base263/{split}.txt')
        if args.training_splits:
            copy_file(args.training_splits / f'{split}.txt', f'lists/training_pose69/{split}.txt')
    copy_file(args.taxonomy_config, 'provenance/semantic_subset_config_v3.py')
    if (args.semantic_root / 'report_all.json').is_file():
        copy_file(args.semantic_root / 'report_all.json', 'provenance/semantic_report_all.json')
    if args.keep_ids:
        copy_file(args.keep_ids, 'lists/keep_ids.txt')
    if args.paper_examples:
        copy_file(args.paper_examples, 'figure_examples/paper_examples.json')

    derived_views = []
    if 'content' in frozen:
        for label, parts in frozen['content'].items():
            assigned = set().union(*(set(parts[s]) for s in ('train', 'val', 'test')))
            unassigned = sorted(set(parts['all']) - assigned)
            relative = f'derived/content_assigned/{label}/all.txt'
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(''.join(mid + '\n' for mid in sorted(assigned)), encoding='utf-8')
            derived_views.append({'output': relative, 'sha256': sha256(target), 'count': len(assigned),
                                  'derivation': 'Union of saved content train/val/test IDs; original all.txt is preserved.',
                                  'unassigned_ids_in_original_all': unassigned})

    mismatches = missing = empty = 0
    snapshot_lines = []
    jsonl_path = out / 'caption_membership_audit.jsonl'
    with jsonl_path.open('x', encoding='utf-8', newline='\n') as stream:
        for mid in sorted(all_ids):
            text_path = args.texts_dir / f'{mid}.txt'
            text_bytes = text_path.read_bytes() if text_path.is_file() else None
            raw_lines = text_bytes.decode('utf-8', errors='replace').splitlines() if text_bytes is not None else []
            captions = [line.strip().split('#', 1)[0].strip() for line in raw_lines if line.strip()]
            captions = [caption for caption in captions if caption]
            document = re.sub(r'\s+', ' ', ' | '.join(captions).strip().lower())
            matched = {view: {label: [{'pattern_id': pid, 'regex': regex,
                                      'match_count': len(compiled.findall(document))}
                                     for pid, regex, compiled in rules if compiled.search(document)]
                             for label, rules in groups.items()} for view, groups in patterns.items()}
            hit_classes = [label for label in STYLE_CLASSES if matched['style'][label]]
            observed_strict = hit_classes if len(hit_classes) == 1 else []
            memberships = {view: {label: [s for s in PARTITIONS if mid in parts[s]]
                                  for label, parts in groups.items() if mid in parts['all']}
                           for view, groups in frozen_sets.items()}
            frozen_style = list(memberships['style'])
            mismatch = frozen_style != observed_strict
            mismatches += int(mismatch)
            if text_bytes is None:
                exclusion = 'missing_caption_file'
                missing += 1
            elif not captions:
                exclusion = 'empty_caption'
                empty += 1
            elif not hit_classes:
                exclusion = 'no_style_family_hit'
            elif len(hit_classes) > 1:
                exclusion = 'multiple_style_family_hits'
            else:
                exclusion = None
            text_hash = hashlib.sha256(text_bytes).hexdigest() if text_bytes is not None else None
            snapshot_lines.append(f'{mid}\t{text_hash or "MISSING"}\n')
            row = {'id': mid, 'source': source_name(mid), 'base_split': base_map.get(mid),
                   'training_pose69_split': training_map.get(mid),
                   'pose69_available': mid in keep if keep is not None else None,
                   'captions': captions, 'humanml_lines': raw_lines, 'text_sha256': text_hash,
                   'frozen_memberships': memberships, 'matched_patterns': matched,
                   'current_caption_hit_classes': hit_classes,
                   'current_caption_strict_class': observed_strict[0] if observed_strict else None,
                   'current_caption_strict_exclusion': exclusion,
                   'frozen_vs_current_caption_strict_mismatch': mismatch}
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')

    manifest = {
        'release_id': args.release_id, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'complete': True, 'copied_files': copied, 'derived_views': derived_views,
        'exporter_sha256': sha256(Path(__file__)),
        'class_order': list(STYLE_CLASSES), 'paper_display_aliases': {'tremor': 'Shake'},
        'caption_audit': {'file': jsonl_path.name, 'sha256': sha256(jsonl_path), 'rows': len(all_ids),
                          'missing_text_files': missing, 'empty_caption_files': empty,
                          'strict_membership_mismatches': mismatches,
                          'snapshot_index_sha256': hashlib.sha256(''.join(snapshot_lines).encode()).hexdigest(),
                          'role': 'Current text snapshot audit; historical caption provenance is not assumed.'},
        'policy': {'frozen_lists_preserved': True, 'regex_matches_do_not_change_frozen_memberships': True,
                   'figure_examples_do_not_change_evaluation': True,
                   'no_motion_files_copied': True, 'no_source_files_modified': True},
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(out), 'rows': len(all_ids), 'mismatches': mismatches}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
