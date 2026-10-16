#!/usr/bin/env python3
"""Create a separate, hash-bound text compatibility view using only local files.

Default: copy flat .txt inputs unchanged. Explicit flags can quarantine confirmed
non-action clips / invalid interval lines and apply the historical loader's
per-field NaN-to-zero interpretation. No captions, tokens or splits are invented
or regenerated. Output must be fresh and separate from the input directory.

Example (ledger paths are supplied separately with the release):
  python scripts/text/prepare_compatibility_text_view.py \
    --source-root paper_exact_v1/texts --output-root usable_compat_v1 \
    --quarantine-manifest junk_safe_exclusion_manifest.json \
    --format-ledger format_issue_ledger.json \
    --apply-confirmed-quarantine --normalize-nan-times

Use --allow-source-subset only when intentionally processing a source subset;
all missing applicable ledger IDs are then recorded. Original archives and
historical split/benchmark lists remain the reproducibility reference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

HEX = re.compile(r'^[0-9a-f]{64}$')
DEVICE_NAMES = {'CON', 'PRN', 'AUX', 'NUL', 'CONIN$', 'CONOUT$'} | {
    f'{prefix}{digit}' for prefix in ('COM', 'LPT') for digit in '123456789¹²³'
}
CLIP_SCOPE = 'exclude_clip_from_separate_usable_view'
LINE_SCOPE = 'exclude_selected_line_only_from_separate_usable_view'
NAN_ISSUE = 'legacy_nan_time_loader_compatibility'
EMPTY_ISSUE = 'invalid_nonzero_empty_interval'


class ViewError(ValueError):
    pass


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ViewError(message)


def safe_id(cid):
    require(isinstance(cid, str) and cid not in ('', '.', '..') and
            not any(c in '/\\' or ord(c) < 32 or ord(c) == 127 for c in cid),
            f'Unsafe clip ID: {cid!r}')
    if os.name == 'nt':
        require(not any(c in '<>:"|?*' for c in cid) and
                cid.split('.', 1)[0].rstrip(' ').upper() not in DEVICE_NAMES,
                f'Clip ID cannot be preserved on Windows; use Linux/macOS: {cid!r}')


def safe_path(path, *, existing):
    path = Path(path).expanduser()
    require('..' not in path.parts, f'Parent traversal is not accepted: {path}')
    path = path.absolute()
    for ancestor in (path, *path.parents):
        require(not ancestor.is_symlink(), f'Symbolic-link path refused: {ancestor}')
        require(not (hasattr(ancestor, 'is_junction') and ancestor.is_junction()),
                f'Junction path refused: {ancestor}')
    if existing:
        require(path.exists(), f'Missing input: {path}')
    return path.resolve(strict=existing)


def load_ledger(path):
    path = safe_path(path, existing=True)
    require(path.is_file(), f'Ledger is not a regular file: {path}')
    raw = path.read_bytes()
    value = json.loads(raw.decode('utf-8-sig'))
    require(isinstance(value, dict) and isinstance(value.get('records'), list),
            f'Ledger must contain a records list: {path}')
    return value, {'path': str(path), 'sha256': sha(raw), 'bytes': len(raw)}


def check_record(row):
    require(isinstance(row, dict), 'Ledger record is not an object.')
    cid = row.get('id')
    safe_id(cid)
    require(type(row.get('line')) is int and row['line'] >= 1,
            f'Invalid one-based line number for {cid}')
    require(row.get('relative_text_path') == f'texts/{cid}.txt',
            f'Unexpected relative text path for {cid}')
    for key in ('old_text_sha256', 'old_line_sha256_utf8_without_newline'):
        require(isinstance(row.get(key), str) and HEX.fullmatch(row[key]),
                f'Invalid {key} for {cid}')
    require(isinstance(row.get('old_caption'), str), f'Missing old caption: {cid}')
    require(row.get('replacement_caption') is None, f'Caption replacement refused: {cid}')
    require(isinstance(row.get('stratification', {}), dict), f'Invalid stratification: {cid}')
    require(not row.get('stratification', {}).get('paper_locked'),
            f'Paper-locked annotation cannot be changed by this tool: {cid}')


def split_raw_lines(raw):
    bom = b'\xef\xbb\xbf' if raw.startswith(b'\xef\xbb\xbf') else b''
    chunks = raw[len(bom):].splitlines(keepends=True)
    parsed = []
    for chunk in chunks:
        ending = b'\r\n' if chunk.endswith(b'\r\n') else (
            b'\n' if chunk.endswith(b'\n') else b'\r' if chunk.endswith(b'\r') else b'')
        body = chunk[:-len(ending)] if ending else chunk
        parsed.append((body, ending))
    return bom, parsed


def check_bound_line(row, raw, lines):
    cid = row['id']
    require(sha(raw) == row['old_text_sha256'], f'Old file SHA mismatch: {cid}')
    index = row['line'] - 1
    require(index < len(lines), f'Missing selected line: {cid}:{row["line"]}')
    body, ending = lines[index]
    require(sha(body) == row['old_line_sha256_utf8_without_newline'],
            f'Old line SHA mismatch: {cid}:{row["line"]}')
    text = body.decode('utf-8')
    fields = text.split('#')
    require(len(fields) == 4 and fields[0] == row['old_caption'],
            f'Old caption/field mismatch: {cid}:{row["line"]}')
    if 'old_complete_line' in row:
        require(text == row['old_complete_line'], f'Complete old line mismatch: {cid}:{row["line"]}')
    if 'old_tokens' in row:
        require(fields[1] == row['old_tokens'], f'Old token field mismatch: {cid}:{row["line"]}')
    if 'old_time_fields' in row:
        require(fields[2:] == row['old_time_fields'], f'Old time fields mismatch: {cid}:{row["line"]}')
    return fields, body, ending


def jsonl_bytes(rows):
    return ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode('utf-8')


def build_view(args):
    source = safe_path(args.source_root, existing=True)
    require(source.is_dir(), 'Source root is not a directory.')
    output = safe_path(args.output_root, existing=False)
    require(not output.exists(), f'Output already exists; choose a fresh directory: {output}')
    require(output.parent.is_dir(), f'Output parent must already exist: {output.parent}')
    require(source != output and source not in output.parents and output not in source.parents,
            'Input/output roots must be separate and must not contain one another.')
    require(not args.apply_confirmed_quarantine or (args.quarantine_manifest and args.format_ledger),
            '--apply-confirmed-quarantine requires both --quarantine-manifest and --format-ledger.')
    require(not args.normalize_nan_times or args.format_ledger,
            '--normalize-nan-times requires --format-ledger.')

    provenance = {}
    quarantine, formatting = {}, {}
    if args.quarantine_manifest:
        quarantine, provenance['quarantine_manifest'] = load_ledger(args.quarantine_manifest)
    if args.format_ledger:
        formatting, provenance['format_ledger'] = load_ledger(args.format_ledger)
    format_by_key = {}
    for row in formatting.get('records', []):
        check_record(row)
        key = (row['id'], row['line'])
        require(key not in format_by_key, f'Duplicate format record: {key}')
        require(row.get('issue') in (NAN_ISSUE, EMPTY_ISSUE), f'Unsupported format issue: {key}')
        format_by_key[key] = row
    if formatting:
        format_counts = Counter(row['issue'] for row in format_by_key.values())
        require(format_counts[NAN_ISSUE] == formatting.get('nonfinite_time_lines') and
                format_counts[EMPTY_ISSUE] == formatting.get('invalid_nonzero_empty_intervals'),
                'Format summary counts disagree with its records.')

    operations = defaultdict(dict)
    whole_clips = set()
    if args.apply_confirmed_quarantine:
        seen = set()
        scopes = Counter()
        for row in quarantine['records']:
            check_record(row)
            key = (row['id'], row['line'])
            require(key not in seen, f'Duplicate quarantine record: {key}')
            seen.add(key)
            scope = row.get('scope')
            if scope == CLIP_SCOPE:
                require(row.get('reason_code') == 'confirmed_non_action_chat_or_grammar_output' and
                        row.get('action') == 'quarantine_pending_visual_recaption',
                        f'Only confirmed non-action clip quarantine is supported: {key}')
                require(row['id'] not in whole_clips, f'Duplicate whole-clip quarantine: {row["id"]}')
                whole_clips.add(row['id'])
                operation = 'quarantine_clip'
                bound = row
            elif scope == LINE_SCOPE:
                bound = format_by_key.get(key)
                require(bound is not None and bound.get('issue') == EMPTY_ISSUE,
                        f'Empty-interval quarantine lacks matching format evidence: {key}')
                require(all(row.get(k) == bound.get(k) for k in
                        ('old_text_sha256', 'old_line_sha256_utf8_without_newline', 'old_caption')),
                        f'Quarantine/format ledger disagreement: {key}')
                require(row.get('reason_code') == EMPTY_ISSUE and row.get('action') ==
                        'quarantine_selected_line_pending_interval_recovery_keep_other_lines',
                        f'Unsupported annotation quarantine: {key}')
                operation = 'quarantine_empty_interval_line'
            else:
                raise ViewError(f'Unsupported quarantine scope: {key}')
            operations[row['id']][row['line']] = (operation, bound)
            scopes[scope] += 1
        require(scopes[CLIP_SCOPE] == quarantine.get('whole_clip_exclusions') and
                scopes[LINE_SCOPE] == quarantine.get('line_only_exclusions'),
                'Quarantine summary counts disagree with its records.')
    if args.normalize_nan_times:
        for key, row in format_by_key.items():
            if row['issue'] != NAN_ISSUE:
                continue
            require(row['line'] not in operations[row['id']], f'Conflicting operations: {key}')
            operations[row['id']][row['line']] = ('normalize_nan_times', row)

    originals, ignored = {}, []
    for path in sorted(source.iterdir()):
        require(not path.is_symlink() and not (hasattr(path, 'is_junction') and path.is_junction()),
                f'Input symlink/junction refused: {path}')
        require(path.is_file(), f'Input must be a flat directory of regular files: {path}')
        if path.suffix != '.txt':
            ignored.append({'name': path.name, 'sha256': sha(path.read_bytes())})
            continue
        safe_id(path.stem)
        originals[path.stem] = path.read_bytes()
    require(originals, 'No .txt inputs found.')
    missing = sorted(set(operations) - set(originals))
    require(not missing or args.allow_source_subset,
            f'{len(missing)} applicable ledger IDs are absent; use --allow-source-subset only for an intentional subset.')
    changes, file_rows, planned = [], [], {}
    counts = Counter({'input_files': len(originals), 'applicable_ledger_ids_absent': len(missing)})
    for cid, raw in originals.items():
        ops = operations.get(cid, {})
        if not ops:
            planned[cid] = raw
            counts['unchanged_files'] += 1
            file_rows.append({'id': cid, 'action': 'unchanged', 'old_sha256': sha(raw), 'new_sha256': sha(raw),
                              'old_bytes': len(raw), 'new_bytes': len(raw)})
            continue
        bom, lines = split_raw_lines(raw)
        replacements = {}
        if cid in whole_clips:
            require({i + 1 for i, (body, _) in enumerate(lines) if body.strip()} == set(ops),
                    f'Whole-clip quarantine would remove other nonempty annotations: {cid}')
        for number, (operation, row) in sorted(ops.items()):
            fields, body, ending = check_bound_line(row, raw, lines)
            new_body = None
            changed_fields = []
            if operation == 'normalize_nan_times':
                times = [float(x) for x in fields[2:]]
                require(any(math.isnan(x) for x in times) and
                        all(math.isfinite(x) or math.isnan(x) for x in times),
                        f'NaN policy cannot normalize this interval: {cid}:{number}')
                new_fields = fields[:]
                for index, value in enumerate(times, 2):
                    if math.isnan(value):
                        new_fields[index] = '0.0'
                        changed_fields.append(index + 1)
                        counts['nan_fields_normalized'] += 1
                new_body = '#'.join(new_fields).encode('utf-8')
                require(new_fields[:2] == fields[:2], 'Caption/token mutation refused.')
                replacements[number] = new_body + ending
                counts['nan_lines_normalized'] += 1
            elif operation == 'quarantine_empty_interval_line':
                a, b = map(float, fields[2:])
                require(math.isfinite(a) and math.isfinite(b) and a == b and a != 0.0,
                        f'Only nonzero empty intervals may be removed: {cid}:{number}')
                replacements[number] = b''
                counts['empty_interval_lines_quarantined'] += 1
            elif operation == 'quarantine_clip':
                replacements[number] = b''
                counts['confirmed_nonaction_lines_quarantined'] += 1
            else:
                raise ViewError(f'Unsupported operation: {operation}')
            changes.append({'id': cid, 'old_line_number': number, 'operation': operation,
                 'new_line_number': number - sum(1 for n, (op, _) in ops.items()
                                                 if n < number and op != 'normalize_nan_times')
                                    if new_body is not None else None,
                 'old_file_sha256': sha(raw), 'old_line_sha256_utf8_without_newline': sha(body),
                 'old_line': body.decode('utf-8'), 'new_line': new_body.decode('utf-8') if new_body is not None else None,
                 'changed_field_numbers': changed_fields, 'caption_rewritten': False, 'tokens_regenerated': False,
                 'new_line_sha256_utf8_without_newline': sha(new_body) if new_body is not None else None})
        if cid in whole_clips:
            result = None
            action = 'quarantined_clip'
            counts['quarantined_clips'] += 1
        else:
            result = bom + b''.join(replacements.get(i + 1, body + ending)
                                    for i, (body, ending) in enumerate(lines))
            require(any(body.strip() for body, _ in split_raw_lines(result)[1]),
                    f'Line filtering would leave an empty file: {cid}')
            action = 'modified_time_or_line_filter'
            counts['modified_retained_files'] += 1
        planned[cid] = result
        counts['quarantine_original_files'] += 1
        file_rows.append({'id': cid, 'action': action, 'old_sha256': sha(raw),
                          'new_sha256': sha(result) if result is not None else None,
                          'old_bytes': len(raw), 'new_bytes': len(result) if result is not None else 0,
                          'original_backup': f'quarantine_originals/{cid}.txt'})
    counts['output_text_files'] = sum(raw is not None for raw in planned.values())
    summary = {'schema_version': 1, 'status': 'dry_run' if args.dry_run else 'complete',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'source_root': str(source),
        'output_root': str(output), 'options': {'apply_confirmed_quarantine': args.apply_confirmed_quarantine,
           'normalize_nan_times': args.normalize_nan_times, 'allow_source_subset': args.allow_source_subset},
        'counts': dict(counts), 'missing_input_ids': missing,
        'missing_entries': [{'id': cid, 'line': number, 'operation': op}
                           for cid in missing for number, (op, _) in sorted(operations[cid].items())],
        'ledger_provenance': provenance, 'ignored_non_txt_input_files': ignored,
        'captions_rewritten': 0, 'tokens_regenerated': 0, 'source_edits': 0, 'model_calls': 0,
        'split_or_benchmark_changes': 0,
        'policy': 'Separate compatibility view; no visual correctness claim. Original archives and historical protocol remain authoritative.'}
    if args.dry_run:
        return summary
    # All requested edits and their hashes were checked before creating output.
    output.mkdir()
    (output / 'texts').mkdir()
    (output / 'quarantine_originals').mkdir()
    for cid, raw in planned.items():
        if raw is not None:
            (output / 'texts' / f'{cid}.txt').write_bytes(raw)
        if operations.get(cid):
            (output / 'quarantine_originals' / f'{cid}.txt').write_bytes(originals[cid])
    manifests = {'file_manifest.jsonl': jsonl_bytes(file_rows), 'line_changes.jsonl': jsonl_bytes(changes)}
    for name, raw in manifests.items():
        (output / name).write_bytes(raw)
    for cid, raw in originals.items():
        require((source / f'{cid}.txt').read_bytes() == raw, f'Input changed during the operation: {cid}')
        target = planned[cid]
        if target is not None:
            require((output / 'texts' / f'{cid}.txt').read_bytes() == target, f'Output verification failed: {cid}')
        if operations.get(cid):
            require((output / 'quarantine_originals' / f'{cid}.txt').read_bytes() == raw,
                    f'Original backup verification failed: {cid}')
    summary['manifest_sha256'] = {name: sha(raw) for name, raw in manifests.items()}
    # Written last: a directory without this complete receipt is not a finished view.
    (output / 'completion.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source-root', type=Path, required=True, help='Flat directory of paper-exact/original .txt files.')
    parser.add_argument('--output-root', type=Path, required=True, help='Fresh separate directory; its parent must exist.')
    parser.add_argument('--quarantine-manifest', type=Path)
    parser.add_argument('--format-ledger', type=Path)
    parser.add_argument('--apply-confirmed-quarantine', action='store_true')
    parser.add_argument('--normalize-nan-times', action='store_true')
    parser.add_argument('--allow-source-subset', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Validate and report; create no output files.')
    args = parser.parse_args()
    try:
        summary = build_view(args)
    except (ViewError, OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f'ERROR: {error}\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
