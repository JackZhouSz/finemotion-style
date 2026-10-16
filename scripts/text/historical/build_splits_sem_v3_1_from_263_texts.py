#!/usr/bin/env python3
"""
Build sem_v3-1 split lists directly from a built HumanML3D-263 dataset's texts/.

Why:
  - `datasets/subsets/sem_v3-1/` was produced from dataset-specific original texts.
  - For finemotion_263_v4 you now have updated, VLM-refined captions under:
      <dataset_root>/texts/*.txt
    and you want semantic splits to be derived from THESE texts (and include motorica/finedance).

Inputs:
  <dataset_root>/
    - texts/*.txt            (HumanML3D-style: caption#tokens#... per line; may have multiple lines)
    - splits/{train,val,test}.txt (base split assignment for each motion-id)
    - new_joint_vecs/*.npy   (only used for optional sanity checks by the caller)

Outputs:
  <dataset_root>/splits_sem_v3-1/<view>/<bucket>/{train,val,test,all,train_val}.txt
    view ∈ {multi, single_argmax, single_strict, attrs}
    bucket ∈ {impact,strike,balance,tremor,all4}  (attrs: periodic, secondary_rich)
  Also writes per-bucket `report.json` and root `report_all.json`.

Selection rules:
  - Same keyword taxonomy as `tools/semantic_subset_config_v3.py` (sem_v3-1).
  - Captions are extracted from the part BEFORE the first '#'.
  - `multi`: a motion is included in each bucket it hits (overlaps allowed).
  - `single_argmax`: assign any-hit to the highest scoring bucket (tie-break by fixed bucket order).
  - `single_strict`: keep only motions that hit exactly ONE bucket.
  - `attrs`: periodic / secondary_rich attributes by keyword match (orthogonal to buckets).

NOTE:
  This script ONLY decides membership using texts. It uses the existing base splits under
  `<dataset_root>/splits/` to write train/val/test partitions.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from semantic_subset_config_v3 import (  # noqa: E402
    ATTRIBUTES_V3,
    BUCKETS_V3,
    MICRO_MODIFIERS_V3,
    SCORE_WEIGHTS_V3,
)


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _now_human() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_ids(p: Path) -> List[str]:
    if not p.is_file():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]


def _write_ids(p: Path, ids: Iterable[str]) -> None:
    _ensure_dir(p.parent)
    data = sorted(dict.fromkeys([x.strip() for x in ids if x and x.strip()]))
    p.write_text("\n".join(data) + ("\n" if data else ""), encoding="utf-8")


def _extract_captions(text_file: Path) -> List[str]:
    caps: List[str] = []
    with text_file.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cap = line.split("#", 1)[0].strip()
            if cap:
                caps.append(cap)
    return caps


def _norm_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _compile(patterns: Sequence[str]) -> List[re.Pattern]:
    return [re.compile(p, flags=re.IGNORECASE) for p in patterns]


def _count_matches(text: str, regs: Sequence[re.Pattern]) -> int:
    c = 0
    for r in regs:
        c += len(r.findall(text))
    return int(c)


def _infer_source(mid: str) -> str:
    if mid.startswith("bandai_30hz__"):
        return "bandai"
    if mid.startswith("lafan1_30hz__"):
        return "lafan1"
    if mid.startswith("motionx_clean_30hz__"):
        return "motionx_clean"
    if mid.startswith("motionx_30hz__"):
        return "motionx"
    if mid.startswith("motorica_30hz__"):
        return "motorica"
    if mid.startswith("finedance_30hz__"):
        return "finedance"
    # HumanML3D ids are pure digits, with optional segmentation suffix.
    if re.fullmatch(r"\d+(?:__s\d{6}_e\d{6})?", mid) is not None:
        return "humanml3d"
    return "unknown"


def _load_base_splits(splits_dir: Path) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    split_map: Dict[str, str] = {}
    by: Dict[str, Set[str]] = {}
    for sp in ("train", "val", "test", "all", "train_val"):
        by[sp] = set(_read_ids(splits_dir / f"{sp}.txt"))

    # Fill missing derived splits.
    if not by["all"]:
        by["all"] = set().union(by["train"], by["val"], by["test"])
    if not by["train_val"]:
        by["train_val"] = set().union(by["train"], by["val"])

    for sp in ("train", "val", "test"):
        for mid in by[sp]:
            split_map[mid] = sp

    return split_map, by


def _pick_best_bucket(hits: Dict[str, int], scores: Dict[str, float]) -> Optional[str]:
    hit_keys = [b.key for b in BUCKETS_V3 if int(hits.get(b.key, 0)) > 0]
    if not hit_keys:
        return None
    priority = [b.key for b in BUCKETS_V3]  # impact first, then strike, balance, tremor
    best = hit_keys[0]
    best_s = float(scores.get(best, 0.0))
    for k in hit_keys[1:]:
        s = float(scores.get(k, 0.0))
        if s > best_s + 1e-9:
            best, best_s = k, s
        elif abs(s - best_s) <= 1e-9:
            if priority.index(k) < priority.index(best):
                best, best_s = k, s
    return best


def _counts_by_split(ids: Sequence[str], split_map: Dict[str, str]) -> Dict[str, int]:
    out = {"train": 0, "val": 0, "test": 0, "all": 0, "train_val": 0}
    for mid in ids:
        out["all"] += 1
        sp = split_map.get(mid)
        if sp in ("train", "val", "test"):
            out[sp] += 1
            if sp in ("train", "val"):
                out["train_val"] += 1
    return out


def _counts_by_source(ids: Sequence[str], split_map: Dict[str, str]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for mid in ids:
        src = _infer_source(mid)
        out.setdefault(src, {"train": 0, "val": 0, "test": 0, "all": 0, "train_val": 0})
        out[src]["all"] += 1
        sp = split_map.get(mid)
        if sp in ("train", "val", "test"):
            out[src][sp] += 1
            if sp in ("train", "val"):
                out[src]["train_val"] += 1
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Built 263 dataset root containing texts/ and splits/.",
    )
    ap.add_argument(
        "--out_split_root",
        type=str,
        default="",
        help="Output split root. Default: <dataset_root>/splits_sem_v3-1",
    )
    ap.add_argument(
        "--backup_existing",
        action="store_true",
        help="If set and out_split_root exists, move it aside with a timestamp suffix before writing.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    texts_dir = dataset_root / "texts"
    splits_dir = dataset_root / "splits"

    if not texts_dir.is_dir():
        raise SystemExit(f"Missing texts/: {texts_dir}")
    if not splits_dir.is_dir():
        raise SystemExit(f"Missing splits/: {splits_dir}")

    out_root = Path(args.out_split_root) if str(args.out_split_root).strip() else (dataset_root / "splits_sem_v3-1")

    if out_root.exists() and bool(args.backup_existing):
        bak = out_root.with_name(out_root.name + f"_backup_{_now_tag()}")
        print(f"[INFO] backing up existing split root: {out_root} -> {bak}")
        shutil.move(str(out_root), str(bak))

    split_map, base_splits = _load_base_splits(splits_dir)
    all_ids = sorted(base_splits["all"])

    bucket_res = {b.key: _compile(b.patterns) for b in BUCKETS_V3}
    attr_res = {a.key: _compile(a.patterns) for a in ATTRIBUTES_V3}
    micro_res = _compile(MICRO_MODIFIERS_V3)

    # Selected id sets
    sel_multi: Dict[str, Set[str]] = {b.key: set() for b in BUCKETS_V3}
    sel_single_argmax: Dict[str, Set[str]] = {b.key: set() for b in BUCKETS_V3}
    sel_single_strict: Dict[str, Set[str]] = {b.key: set() for b in BUCKETS_V3}
    sel_attrs: Dict[str, Set[str]] = {a.key: set() for a in ATTRIBUTES_V3}

    scanned = 0
    missing_text = 0
    skipped_no_caption = 0

    for mid in all_ids:
        scanned += 1
        tp = texts_dir / f"{mid}.txt"
        if not tp.is_file():
            missing_text += 1
            continue
        caps = _extract_captions(tp)
        if not caps:
            skipped_no_caption += 1
            continue
        doc = " | ".join(caps)
        t = _norm_text(doc)

        micro = _count_matches(t, micro_res)

        # attrs first (also contribute to bucket score, following build_semantic_subsets_v3.py)
        attr_hits: Dict[str, int] = {}
        for a in ATTRIBUTES_V3:
            n = _count_matches(t, attr_res[a.key])
            attr_hits[a.key] = n
            if n > 0:
                sel_attrs[a.key].add(mid)

        # buckets: hit counts + scoring
        bucket_hits: Dict[str, int] = {}
        bucket_scores: Dict[str, float] = {}
        for b in BUCKETS_V3:
            n = _count_matches(t, bucket_res[b.key])
            bucket_hits[b.key] = n
            bucket_scores[b.key] = float(n) * float(SCORE_WEIGHTS_V3["bucket_hit"]) + float(micro) * float(
                SCORE_WEIGHTS_V3["micro_modifier"]
            )

        # Add attr weights (same as build_semantic_subsets_v3.py; added to all buckets equally).
        sr = int(attr_hits.get("secondary_rich", 0))
        pe = int(attr_hits.get("periodic", 0))
        if sr:
            for b in BUCKETS_V3:
                bucket_scores[b.key] += float(sr) * float(SCORE_WEIGHTS_V3["secondary_rich"])
        if pe:
            for b in BUCKETS_V3:
                bucket_scores[b.key] += float(pe) * float(SCORE_WEIGHTS_V3["periodic"])

        # multi buckets
        hit_keys = [b.key for b in BUCKETS_V3 if int(bucket_hits.get(b.key, 0)) > 0]
        for k in hit_keys:
            sel_multi[k].add(mid)

        # single_strict
        if len(hit_keys) == 1:
            sel_single_strict[hit_keys[0]].add(mid)

        # single_argmax
        best = _pick_best_bucket(bucket_hits, bucket_scores)
        if best is not None:
            sel_single_argmax[best].add(mid)

    # union all4
    def _union_all4(m: Dict[str, Set[str]]) -> Set[str]:
        out: Set[str] = set()
        for b in BUCKETS_V3:
            out |= set(m.get(b.key, set()))
        return out

    all4_multi = _union_all4(sel_multi)
    all4_argmax = _union_all4(sel_single_argmax)
    all4_strict = _union_all4(sel_single_strict)

    # Write outputs
    generated: List[Dict[str, object]] = []

    def _write_view_bucket(view: str, bucket: str, ids: Sequence[str]) -> None:
        out_dir = out_root / view / bucket
        _ensure_dir(out_dir)

        ids = sorted(dict.fromkeys([x for x in ids if x and x.strip()]))
        ids_train = [x for x in ids if split_map.get(x) == "train"]
        ids_val = [x for x in ids if split_map.get(x) == "val"]
        ids_test = [x for x in ids if split_map.get(x) == "test"]
        ids_train_val = sorted(dict.fromkeys(ids_train + ids_val))

        _write_ids(out_dir / "train.txt", ids_train)
        _write_ids(out_dir / "val.txt", ids_val)
        _write_ids(out_dir / "test.txt", ids_test)
        _write_ids(out_dir / "all.txt", ids)
        _write_ids(out_dir / "train_val.txt", ids_train_val)

        rep = {
            "generated_at": _now_human(),
            "dataset_root": str(dataset_root),
            "texts_dir": str(texts_dir),
            "splits_dir": str(splits_dir),
            "view": view,
            "bucket": bucket,
            "counts_final": _counts_by_split(ids, split_map),
            "counts_by_source": _counts_by_source(ids, split_map),
        }
        (out_dir / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        generated.append({"view": view, "bucket": bucket, "counts_final": rep["counts_final"]})

    # Buckets (multi / single_argmax / single_strict)
    for b in BUCKETS_V3:
        _write_view_bucket("multi", b.key, sorted(sel_multi[b.key]))
    _write_view_bucket("multi", "all4", sorted(all4_multi))

    for b in BUCKETS_V3:
        _write_view_bucket("single_argmax", b.key, sorted(sel_single_argmax[b.key]))
    _write_view_bucket("single_argmax", "all4", sorted(all4_argmax))

    for b in BUCKETS_V3:
        _write_view_bucket("single_strict", b.key, sorted(sel_single_strict[b.key]))
    _write_view_bucket("single_strict", "all4", sorted(all4_strict))

    # Attributes
    for a in ATTRIBUTES_V3:
        _write_view_bucket("attrs", a.key, sorted(sel_attrs[a.key]))

    # Root report
    _ensure_dir(out_root)
    root_report = {
        "generated_at": _now_human(),
        "dataset_root": str(dataset_root),
        "texts_dir": str(texts_dir),
        "splits_dir": str(splits_dir),
        "scanned_ids_from_base_splits_all": int(len(all_ids)),
        "missing_text_files": int(missing_text),
        "skipped_no_caption": int(skipped_no_caption),
        "taxonomy": {
            "buckets": [{"key": b.key, "desc_zh": b.desc_zh, "patterns": list(b.patterns)} for b in BUCKETS_V3],
            "attributes": [{"key": a.key, "desc_zh": a.desc_zh, "patterns": list(a.patterns)} for a in ATTRIBUTES_V3],
            "micro_modifiers": list(MICRO_MODIFIERS_V3),
            "score_weights": dict(SCORE_WEIGHTS_V3),
        },
        "generated": generated,
    }
    (out_root / "report_all.json").write_text(
        json.dumps(root_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("[OK] wrote splits under:", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

