#!/usr/bin/env python3
"""
Select a "clean" subset from Motion-X (already converted to SMPL24 npz).

Goal
----
Motion-X mixes multiple sources. Some clips have:
  - non Y-up coordinate system (X-up or Z-up)
  - unrealistic global translation: teleport jumps / extremely high offsets

This script scans Motion-X SMPL24 npz files and selects a conservative subset that
matches "normal" behavior (similar to a known-good clip like 2_Squats_2_Jacks_clip_1):
  - finite values
  - detected up-axis == Y
  - translation step spikes below threshold
  - mean translation along up-axis below threshold
  - (optional) exclude certain groups (e.g. aist)

It outputs:
  - a manifest text file of selected relative paths
  - a JSON report with per-group stats and drop reasons
  - optionally copies selected npz into an output root (mirroring structure)

Example
-------
source /home/randong/miniconda3/etc/profile.d/conda.sh
conda run -n lafan_smplx_cpu python tools/select_motionx_clean_subset.py \
  --in_root datasets/raw/Motion-X_smpl24 \
  --smpl_model_root third_party/eval-mcm-ldm/models \
  --exclude_group aist \
  --max_step 0.30 \
  --max_up_mean 2.0 \
  --out_manifest datasets/subsets/motionx_clean_v1.txt \
  --out_report outputs/motionx_clean_v1_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_npz(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.npz"))


def _group_key(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if rel.parts else "unknown"


def _load_motion_npz(p: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(p, allow_pickle=True)
    if "poses" not in z.files or "trans" not in z.files:
        raise KeyError(f"missing poses/trans (keys={list(z.files)})")
    poses = np.asarray(z["poses"], dtype=np.float32)
    trans = np.asarray(z["trans"], dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (24, 3):
        raise ValueError(f"bad poses shape: {poses.shape}")
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"bad trans shape: {trans.shape}")
    if poses.shape[0] != trans.shape[0]:
        raise ValueError(f"frame mismatch: poses={poses.shape[0]} trans={trans.shape[0]}")
    if "betas10" in z.files:
        betas10 = np.asarray(z["betas10"], dtype=np.float32).reshape(-1)[:10]
    elif "betas" in z.files:
        betas10 = np.asarray(z["betas"], dtype=np.float32).reshape(-1)[:10]
    else:
        betas10 = np.zeros((10,), dtype=np.float32)
    return poses, trans, betas10


@dataclass
class ClipDecision:
    rel: str
    group: str
    ok: bool
    reason: str
    T: int
    up_axis: Optional[int]
    max_step: float
    up_mean: float


def _detect_up_axis(
    poses: np.ndarray,
    trans: np.ndarray,
    betas10: np.ndarray,
    *,
    smpl_model_root: Path,
    device: str,
    axis_sample_frames: int,
) -> int:
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools.convert_motionx_smplx322_to_smpl24_npz import _detect_up_axis_from_smpl24  # noqa: WPS433

    T = int(poses.shape[0])
    return int(
        _detect_up_axis_from_smpl24(
            poses,
            trans,
            betas10,
            smpl_model_root=smpl_model_root,
            device=device,
            sample_frames=min(int(axis_sample_frames), T),
        )
    )


def main() -> int:
    repo_root = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", type=str, default=str(repo_root / "datasets" / "raw" / "Motion-X_smpl24"))
    ap.add_argument("--out_copy_root", type=str, default="", help="Optional: copy selected npz into this root.")
    ap.add_argument(
        "--out_manifest",
        type=str,
        default=str(repo_root / "datasets" / "subsets" / "motionx_clean_v1.txt"),
        help="Write selected relative paths (under in_root) to this text file.",
    )
    ap.add_argument(
        "--out_report",
        type=str,
        default=str(repo_root / "outputs" / "motionx_clean_v1_report.json"),
        help="Write a JSON report here.",
    )
    ap.add_argument("--limit", type=int, default=-1, help="Debug: only scan first N files.")
    ap.add_argument("--exclude_group", action="append", default=[], help="Exclude top-level groups (repeatable).")
    ap.add_argument("--min_T", type=int, default=20, help="Require at least this many frames.")
    ap.add_argument("--max_step", type=float, default=0.30, help="Max allowed per-frame translation step (L2).")
    ap.add_argument("--max_up_mean", type=float, default=2.0, help="Max allowed mean translation along up-axis.")
    ap.add_argument("--require_y_up", action="store_true", help="Require detected up-axis == Y (recommended).")
    ap.add_argument(
        "--smpl_model_root",
        type=str,
        default=str(repo_root / "third_party" / "eval-mcm-ldm" / "models"),
        help='Folder containing "smpl/SMPL_NEUTRAL.pkl".',
    )
    ap.add_argument("--device", type=str, default="cpu", help="cpu/cuda for SMPL forward.")
    ap.add_argument("--axis_sample_frames", type=int, default=80)
    ap.add_argument("--copy", action="store_true", help="Actually copy selected npz to out_copy_root.")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    if not in_root.is_dir():
        raise SystemExit(f"in_root not found: {in_root}")
    smpl_model_root = Path(args.smpl_model_root)
    exclude = set([str(x).strip() for x in (args.exclude_group or []) if str(x).strip()])

    files = list(_iter_npz(in_root))
    if int(args.limit) > 0:
        files = files[: int(args.limit)]

    out_manifest = Path(args.out_manifest)
    out_report = Path(args.out_report)

    out_copy_root = Path(args.out_copy_root) if str(args.out_copy_root).strip() else None
    if bool(args.copy) and out_copy_root is None:
        raise SystemExit("--copy requires --out_copy_root")

    decisions: List[ClipDecision] = []
    selected: List[str] = []
    reason_ctr: Counter[str] = Counter()
    by_group: Dict[str, Counter[str]] = defaultdict(Counter)
    by_group_sel: Counter[str] = Counter()

    for i, p in enumerate(files, 1):
        rel = str(p.relative_to(in_root)).replace("\\", "/")
        group = _group_key(p, in_root)
        if group in exclude:
            dec = ClipDecision(rel=rel, group=group, ok=False, reason="excluded_group", T=0, up_axis=None, max_step=0.0, up_mean=0.0)
            decisions.append(dec)
            reason_ctr[dec.reason] += 1
            by_group[group][dec.reason] += 1
            continue
        try:
            poses, trans, betas10 = _load_motion_npz(p)
            T = int(poses.shape[0])
            if T < int(args.min_T):
                reason = "too_short"
                dec = ClipDecision(rel=rel, group=group, ok=False, reason=reason, T=T, up_axis=None, max_step=0.0, up_mean=0.0)
            else:
                if (not np.isfinite(poses).all()) or (not np.isfinite(trans).all()):
                    reason = "nan_or_inf"
                    dec = ClipDecision(rel=rel, group=group, ok=False, reason=reason, T=T, up_axis=None, max_step=0.0, up_mean=0.0)
                else:
                    d = trans[1:] - trans[:-1]
                    max_step = float(np.linalg.norm(d, axis=1).max()) if d.shape[0] > 0 else 0.0
                    up_axis = _detect_up_axis(
                        poses,
                        trans,
                        betas10,
                        smpl_model_root=smpl_model_root,
                        device=str(args.device),
                        axis_sample_frames=int(args.axis_sample_frames),
                    )
                    up_mean = float(np.mean(trans[:, up_axis]))

                    if bool(args.require_y_up) and up_axis != 1:
                        reason = f"not_y_up:{up_axis}"
                        dec = ClipDecision(rel=rel, group=group, ok=False, reason=reason, T=T, up_axis=up_axis, max_step=max_step, up_mean=up_mean)
                    elif max_step > float(args.max_step):
                        reason = "teleport"
                        dec = ClipDecision(rel=rel, group=group, ok=False, reason=reason, T=T, up_axis=up_axis, max_step=max_step, up_mean=up_mean)
                    elif up_mean > float(args.max_up_mean):
                        reason = "float"
                        dec = ClipDecision(rel=rel, group=group, ok=False, reason=reason, T=T, up_axis=up_axis, max_step=max_step, up_mean=up_mean)
                    else:
                        dec = ClipDecision(rel=rel, group=group, ok=True, reason="ok", T=T, up_axis=up_axis, max_step=max_step, up_mean=up_mean)
        except Exception as e:
            dec = ClipDecision(rel=rel, group=group, ok=False, reason=f"exception:{type(e).__name__}", T=0, up_axis=None, max_step=0.0, up_mean=0.0)

        decisions.append(dec)
        reason_ctr[dec.reason] += 1
        by_group[group][dec.reason] += 1
        if dec.ok:
            selected.append(rel)
            by_group_sel[group] += 1
            if bool(args.copy) and out_copy_root is not None:
                src = p
                dst = out_copy_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        if i % 2000 == 0 or i == len(files):
            print(f"[scan] {i}/{len(files)} selected={len(selected)}")

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")

    report = {
        "generated_at": _now(),
        "in_root": str(in_root),
        "args": {
            "exclude_group": sorted(list(exclude)),
            "min_T": int(args.min_T),
            "max_step": float(args.max_step),
            "max_up_mean": float(args.max_up_mean),
            "require_y_up": bool(args.require_y_up),
            "smpl_model_root": str(smpl_model_root),
            "device": str(args.device),
            "axis_sample_frames": int(args.axis_sample_frames),
            "copy": bool(args.copy),
            "out_copy_root": str(out_copy_root) if out_copy_root is not None else "",
        },
        "counts": {
            "n_total": len(decisions),
            "n_selected": len(selected),
            "n_dropped": int(len(decisions) - len(selected)),
            "by_reason": dict(reason_ctr),
            "by_group_selected": dict(by_group_sel),
        },
        "by_group_reasons": {g: dict(c) for g, c in sorted(by_group.items(), key=lambda x: x[0])},
        "manifest_path": str(out_manifest),
    }

    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("[OK] wrote manifest:", out_manifest)
    print("[OK] wrote report:", out_report)
    print(f"[OK] selected {len(selected)}/{len(decisions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

