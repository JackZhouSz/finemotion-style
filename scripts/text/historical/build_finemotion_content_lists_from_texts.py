#!/usr/bin/env python3
"""
Build FineMotion content subset lists by keyword-matching captions (Stage-A text filtering),
following the same rule used for HumanML3D step0 content4:

- locomotion classes (mutually exclusive by priority):
    circle > turn > run > walk

- optional style exclusion:
    if a sample matches any strong style keyword (impact/strike/balance/tremor),
    it will NOT be added to content lists (keeps content pool "cleaner").

Inputs (default):
  - <dataset_root>/texts/*.txt  (HumanML3D-style: caption#tokens#f_tag#to_tag per line)
  - <dataset_root>/splits/{train,val,test}.txt (plus optional all/train_val)

Outputs (default):
  - datasets/subsets/sem_v3-1/content/hfcontent_v0_4class/<class>/{all,train,val,test,train_val}.txt
  - datasets/subsets/sem_v3-1/content/report_finemotion_content4_v0.{json,md}
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ----------------------------
# Content patterns (copied from cursor_humanml3d_subdataset.md)
# ----------------------------

CONTENT_NEUTRAL_PATTERNS: Tuple[str, ...] = (
    # locomotion
    r"\bwalk(?:ing|ed|s)?\b",
    r"\brun(?:ning|s)?\b",
    r"\bjog(?:ging|s)?\b",
    r"\bmove(?:s|d|ing)?\b",
    r"\bforward\b",
    r"\bbackward\b",
    r"\bsideways\b",
    # posture / transitions
    r"\bstand(?:ing|s)?\b",
    r"\bsit(?:ting|s)?\b",
    r"\blie(?:s|ing)?\b",
    r"\blaying\b",
    r"\bkneel(?:ing|s)?\b",
    r"\brise(?:s|d|ing)?\b",
    r"\bget\s+up\b",
    r"\bsit\s+down\b",
    r"\blay\s+down\b",
    # turns / navigation (still primary)
    r"\bturn(?:ing|ed|s)?\b",
    r"\brotate(?:s|d|ing)?\b",
    r"\bspin(?:ning|s)?\b",
    r"\bpivot(?:ing|ed|s)?\b",
    r"\bcircle\b",
    r"\bclockwise\b",
    r"\bcounterclockwise\b",
    # stairs / steps
    r"\bstairs?\b",
    r"\bstep(?:ping|s)?\b",
    r"\bstep\s+over\b",
    r"\bstep\s+up\b",
    r"\bstep\s+down\b",
    # daily tasks (low frequency dominant)
    r"\bwash(?:ing|es|ed)?\b",
    r"\bclean(?:ing|s|ed)?\b",
    r"\bsweep(?:ing|s|ed)?\b",
    r"\bwipe(?:ing|s|d)?\b",
    r"\bopen(?:ing|s|ed)?\b",
    r"\bclose(?:ing|s|d)?\b",
    # reach/carry/pick-place
    r"\breach(?:ing|es|ed)?\b",
    r"\bpick(?:ing|s|ed)?\b",
    r"\blift(?:ing|s|ed)?\b",
    r"\bcarry(?:ing|s|ed)?\b",
    r"\bhold(?:ing|s|held)?\b",
    r"\bplace(?:s|d|ing)?\b",
    r"\bput(?:s|ting)?\b",
)


LOCOMOTION_CLASS_PATTERNS: Dict[str, Tuple[str, ...]] = {
    # Priority intended: circle > turn > run > walk
    "circle": (
        r"\bcircle\b",
        r"\bclockwise\b",
        r"\bcounterclockwise\b",
    ),
    "turn": (
        r"\bturn(?:ing|ed|s)?\b",
        r"\brotate(?:s|d|ing)?\b",
        r"\bspin(?:ning|s)?\b",
        r"\bpivot(?:ing|ed|s)?\b",
    ),
    "run": (
        r"\brun(?:ning|s)?\b",
        r"\bjog(?:ging|s)?\b",
    ),
    "walk": (
        r"\bwalk(?:ing|ed|s)?\b",
    ),
}


# Strong style keywords used only for exclusion (same 4-class setup)
STYLE_STRONG_PATTERNS: Tuple[str, ...] = (
    # impact
    r"\bhit(?:s|ting)?\b",
    r"\bimpact(?:s|ing)?\b",
    r"\bkick(?:s|ing|ed)?\b",
    r"\bpunch(?:es|ing|ed)?\b",
    r"\bstomp(?:s|ing|ed)?\b",
    r"\bslam(?:s|ming|med)?\b",
    r"\bthump(?:s|ing|ed)?\b",
    r"\bstrike(?:s|ing|d)?\b",
    r"\battack(?:s|ing|ed)?\b",
    # balance
    r"\bslip(?:ping|ped|s)?\b",
    r"\btrip(?:ping|ped|s)?\b",
    r"\bstumbl(?:e|es|ed|ing)\b",
    r"\brecover(?:y|ing|ed|s)?\b",
    r"\bpush(?:ing|ed|es)?\b",
    r"\bpull(?:ing|ed|s)?\b",
    r"\blose(?:s|ing)?\s+balance\b",
    # tremor
    r"\bshake(?:s|d|n|ing)?\b",
    r"\btrembl(?:e|es|ed|ing)\b",
    r"\bvibrat(?:e|es|ed|ing)\b",
    r"\bjerk(?:s|ed|ing)?\b",
    r"\bquiver(?:s|ed|ing)?\b",
)


def _read_ids(p: Path) -> List[str]:
    if not p.is_file():
        return []
    out: List[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    return out


def _sorted_ids(ids: Iterable[str]) -> List[str]:
    # deterministic, mimic older convention: numeric part then mirror
    def k(s: str) -> Tuple[str, int]:
        return (s.lstrip("M").zfill(12), 1 if s.startswith("M") else 0)

    return sorted(set(ids), key=k)


def _write_ids(p: Path, ids: Iterable[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for x in _sorted_ids(ids):
            f.write(f"{x}\n")


def _iter_text_files(texts_dir: Path) -> Iterable[Path]:
    for p in sorted(texts_dir.glob("*.txt")):
        if p.is_file():
            yield p


def _extract_captions(text_file: Path) -> List[str]:
    # Each line: caption#tokens#f_tag#to_tag
    caps: List[str] = []
    with text_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cap = line.split("#", 1)[0].strip()
            if cap:
                caps.append(cap)
    return caps


def _compile_patterns(patterns: Sequence[str]) -> List[re.Pattern]:
    return [re.compile(p, flags=re.IGNORECASE) for p in patterns]


def _match_any(caps: Sequence[str], pats: Sequence[re.Pattern]) -> bool:
    for c in caps:
        for p in pats:
            if p.search(c) is not None:
                return True
    return False


def _assign_locomotion_class(
    caps: List[str],
    compiled_by_class: Dict[str, List[re.Pattern]],
) -> Optional[str]:
    # priority: circle > turn > run > walk
    for cls in ("circle", "turn", "run", "walk"):
        if _match_any(caps, compiled_by_class[cls]):
            return cls
    return None


@dataclass(frozen=True)
class ReportCounts:
    count: int
    by_split: Dict[str, int]


def build(
    dataset_root: Path,
    out_root: Path,
    style_exclude: bool = True,
) -> Dict:
    texts_dir = dataset_root / "texts"
    splits_dir = dataset_root / "splits"
    if not texts_dir.is_dir():
        raise FileNotFoundError(f"texts dir not found: {texts_dir}")
    if not splits_dir.is_dir():
        raise FileNotFoundError(f"splits dir not found: {splits_dir}")

    # Load base splits
    orig_splits: Dict[str, Set[str]] = {}
    for sp in ("train", "val", "test", "all", "train_val"):
        p = splits_dir / f"{sp}.txt"
        orig_splits[sp] = set(_read_ids(p)) if p.is_file() else set()
    if not orig_splits["all"]:
        orig_splits["all"] = set().union(orig_splits["train"], orig_splits["val"], orig_splits["test"])
    if not orig_splits["train_val"]:
        orig_splits["train_val"] = set().union(orig_splits["train"], orig_splits["val"])

    neutral_pats = _compile_patterns(CONTENT_NEUTRAL_PATTERNS)
    locomotion_compiled: Dict[str, List[re.Pattern]] = {
        k: _compile_patterns(v) for k, v in LOCOMOTION_CLASS_PATTERNS.items()
    }
    style_all_pats = _compile_patterns(STYLE_STRONG_PATTERNS)

    neutral_ids: Set[str] = set()
    locomotion_ids: Set[str] = set()
    locomotion_by_class: Dict[str, Set[str]] = {k: set() for k in ("walk", "run", "turn", "circle")}

    scanned = 0
    skipped_no_caption = 0
    skipped_style_exclude = 0

    # Scan all text files (one file per motion-id)
    for tf in _iter_text_files(texts_dir):
        mid = tf.stem
        scanned += 1
        caps = _extract_captions(tf)
        if not caps:
            skipped_no_caption += 1
            continue

        is_style = _match_any(caps, style_all_pats) if style_exclude else False
        if is_style:
            skipped_style_exclude += 1

        if _match_any(caps, neutral_pats) and (not is_style):
            neutral_ids.add(mid)

        cls = _assign_locomotion_class(caps, locomotion_compiled)
        if cls is not None and (not is_style):
            locomotion_by_class[cls].add(mid)
            locomotion_ids.add(mid)

    # Emit list-only dirs
    out_root.mkdir(parents=True, exist_ok=True)
    out_classes_root = out_root / "hfcontent_v0_4class"
    out_classes_root.mkdir(parents=True, exist_ok=True)

    def _create_list_only_dir(dst_root: Path, ids: Set[str]) -> None:
        dst_root.mkdir(parents=True, exist_ok=True)
        _write_ids(dst_root / "all.txt", ids)
        _write_ids(dst_root / "train.txt", ids.intersection(orig_splits["train"]))
        _write_ids(dst_root / "val.txt", ids.intersection(orig_splits["val"]))
        _write_ids(dst_root / "test.txt", ids.intersection(orig_splits["test"]))
        _write_ids(dst_root / "train_val.txt", ids.intersection(orig_splits["train_val"]))
        (dst_root / "README.md").write_text(
            "\n".join(
                [
                    "# FineMotion content subset (lists only)",
                    "",
                    "This directory contains subset split lists (TXT) only, without data files or symbolic links.",
                    f"source dataset_root: `{dataset_root}`",
                    f"generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    # unions
    _create_list_only_dir(out_root / "neutral", neutral_ids)
    _create_list_only_dir(out_root / "locomotion", locomotion_ids)
    for cls in ("walk", "run", "turn", "circle"):
        _create_list_only_dir(out_classes_root / cls, locomotion_by_class[cls])

    # report
    def _mk_counts(ids: Set[str]) -> ReportCounts:
        return ReportCounts(
            count=len(ids),
            by_split={
                sp: len(ids.intersection(orig_splits[sp])) for sp in ("train", "val", "test")
            },
        )

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_root": str(dataset_root),
        "out_root": str(out_root),
        "style_exclude": bool(style_exclude),
        "stats": {
            "scanned_text_files": scanned,
            "skipped_no_caption": skipped_no_caption,
            "skipped_style_exclude": skipped_style_exclude,
        },
        "content": {
            "neutral": _mk_counts(neutral_ids).__dict__,
            "locomotion": _mk_counts(locomotion_ids).__dict__,
            "locomotion_by_class": {k: _mk_counts(v).__dict__ for k, v in locomotion_by_class.items()},
            "locomotion_priority": ["circle", "turn", "run", "walk"],
        },
        "patterns": {
            "neutral": list(CONTENT_NEUTRAL_PATTERNS),
            "locomotion_by_class": {k: list(v) for k, v in LOCOMOTION_CLASS_PATTERNS.items()},
            "style_strong_exclude": list(STYLE_STRONG_PATTERNS),
        },
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_root",
        type=str,
        default="/home/randong/mydata/SOTAtest-HHT-Motion/datasets/FineMotion/humanml3d_20hz_for_Tranning263/finemotion_263_v3",
        help="FineMotion HumanML3D-style dataset root (has texts/ and splits/).",
    )
    ap.add_argument(
        "--out_root",
        type=str,
        default="/home/randong/mydata/SOTAtest-HHT-Motion/datasets/subsets/sem_v3-1/content",
        help="Output root under datasets/subsets/sem_v3-1.",
    )
    ap.add_argument(
        "--no_style_exclude",
        action="store_true",
        help="Disable strong style keyword exclusion.",
    )
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    report = build(dataset_root=dataset_root, out_root=out_root, style_exclude=not args.no_style_exclude)

    (out_root / "report_finemotion_content4_v0.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # small md
    md: List[str] = []
    md.append("# FineMotion content4 subset (from texts) - report\n")
    md.append(f"- generated_at: {report['generated_at']}")
    md.append(f"- dataset_root: `{report['dataset_root']}`")
    md.append(f"- out_root: `{report['out_root']}`")
    md.append(f"- style_exclude: `{report['style_exclude']}`\n")
    md.append("## Counts\n")
    md.append("| subset | total | train | val | test |")
    md.append("|---|---:|---:|---:|---:|")
    for k in ("neutral", "locomotion"):
        x = report["content"][k]
        md.append(f"| {k} | {x['count']} | {x['by_split']['train']} | {x['by_split']['val']} | {x['by_split']['test']} |")
    md.append("")
    md.append("## locomotion 4-class (mutually exclusive, priority: circle > turn > run > walk)\n")
    md.append("| class | total | train | val | test |")
    md.append("|---|---:|---:|---:|---:|")
    for cls in ("walk", "run", "turn", "circle"):
        x = report["content"]["locomotion_by_class"][cls]
        md.append(f"| {cls} | {x['count']} | {x['by_split']['train']} | {x['by_split']['val']} | {x['by_split']['test']} |")
    (out_root / "report_finemotion_content4_v0.md").write_text("\n".join(md), encoding="utf-8")

    print("[OK] wrote:")
    print(f"  - {out_root}/report_finemotion_content4_v0.json")
    print(f"  - {out_root}/report_finemotion_content4_v0.md")
    print("  - list-only dirs under:")
    print(f"      {out_root}/(neutral|locomotion|hfcontent_v0_4class/<class>/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

