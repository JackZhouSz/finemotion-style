#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch convert FineDance motion npy files to SMPL npz (trans + poses),
matching the NPZ convention used by:
  - newBVHdata/Motorica_to_smpl/npz/*.npz
  - newBVHdata/LaFAN1_to_smpl/npz/*.npz
  - newBVHdata/Bandai_to_smpl/npz/*.npz

Output NPZ keys/shapes:
  - trans: (T, 3) float32   (meters, Y-up as FineDance)
  - poses: (T, 24, 3) float32  (SMPL axis-angle, 24 joints)

This script reuses the conversion logic in:
  newBVHdata/FineDance_to_smpl/finedance_to_smpl_npz.py
which already includes the correct FineDance/LODGE 6D rotation convention fix.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# Make sibling module importable when executed from anywhere.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from finedance_to_smpl_npz import finedance_motion_to_smpl_npz  # noqa: E402


@dataclass(frozen=True)
class Job:
    in_path: str
    out_path: str
    hand_mode: str
    add_height_y: float


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _iter_jobs(
    in_dir: Path,
    out_dir: Path,
    hand_mode: str,
    add_height_y: float,
    overwrite: bool,
    only: Optional[Iterable[str]] = None,
    limit: int = 0,
) -> List[Job]:
    only_set = None
    if only:
        only_set = {Path(x).stem for x in only}

    files = sorted(in_dir.glob("*.npy"))
    jobs: List[Job] = []
    for p in files:
        if only_set is not None and p.stem not in only_set:
            continue

        out_p = out_dir / f"{p.stem}.npz"
        if (not overwrite) and out_p.exists():
            continue

        jobs.append(
            Job(
                in_path=str(p),
                out_path=str(out_p),
                hand_mode=hand_mode,
                add_height_y=float(add_height_y),
            )
        )
        if limit and len(jobs) >= int(limit):
            break
    return jobs


def _run_one(job: Job) -> Dict:
    # Do imports inside the worker to keep spawn-friendly.
    in_path = job.in_path
    out_path = job.out_path
    T, fps_est = finedance_motion_to_smpl_npz(
        motion_npy_path=in_path,
        out_npz_path=out_path,
        add_height_y=float(job.add_height_y),
        hand_mode=str(job.hand_mode),
    )
    return {
        "name": Path(in_path).stem,
        "in_path": in_path,
        "out_path": out_path,
        "T": int(T),
        "fps_est": None if fps_est is None else float(fps_est),
        "hand_mode": str(job.hand_mode),
        "add_height_y": float(job.add_height_y),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch convert FineDance motion/*.npy -> SMPL npz (trans+poses)."
    )
    ap.add_argument(
        "--in-dir",
        type=str,
        default="finedance/motion",
        help="Input directory containing FineDance motion npy files (default: finedance/motion)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="newBVHdata/FineDance_to_smpl/npz",
        help="Output directory for SMPL npz files (default: newBVHdata/FineDance_to_smpl/npz)",
    )
    ap.add_argument(
        "--hand-mode",
        type=str,
        default="index1",
        choices=["zeros", "index1"],
        help="How to fill SMPL left_hand/right_hand joints when converting from SMPLH blocks.",
    )
    ap.add_argument(
        "--add-height-y",
        type=float,
        default=0.0,
        help="Optional y translation offset added to trans[:,1] (meters).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npz outputs. Default: skip existing for resume.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes. 1 = sequential (default).",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional list of motion ids / filenames to process (e.g. 001 050 100).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on how many files to process (0 = no limit).",
    )
    ap.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Optional path to write a JSON manifest of conversions.",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {in_dir}")
    _ensure_dir(out_dir)

    jobs = _iter_jobs(
        in_dir=in_dir,
        out_dir=out_dir,
        hand_mode=str(args.hand_mode),
        add_height_y=float(args.add_height_y),
        overwrite=bool(args.overwrite),
        only=args.only,
        limit=int(args.limit),
    )

    total_in_dir = len(list(in_dir.glob("*.npy")))
    print(f"[INFO] in_dir={in_dir}  (found {total_in_dir} *.npy)")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] jobs_to_run={len(jobs)}  (overwrite={bool(args.overwrite)})")
    print(f"[INFO] hand_mode={args.hand_mode}  add_height_y={float(args.add_height_y)}")

    results: List[Dict] = []
    if not jobs:
        print("[INFO] Nothing to do.")
    elif int(args.workers) <= 1:
        for i, job in enumerate(jobs):
            res = _run_one(job)
            results.append(res)
            if i == 0 or (i + 1) % 50 == 0 or (i + 1) == len(jobs):
                print(f"[PROGRESS] {i+1}/{len(jobs)}  last={res['name']}  T={res['T']}")
    else:
        workers = int(args.workers)
        print(f"[INFO] Using {workers} worker processes...")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run_one, j) for j in jobs]
            done = 0
            for fut in as_completed(futs):
                res = fut.result()
                results.append(res)
                done += 1
                if done == 1 or done % 50 == 0 or done == len(jobs):
                    print(f"[PROGRESS] {done}/{len(jobs)}  last={res['name']}  T={res['T']}")

    if args.manifest:
        man_path = Path(args.manifest)
        _ensure_dir(man_path.parent)
        payload = {
            "in_dir": str(in_dir),
            "out_dir": str(out_dir),
            "hand_mode": str(args.hand_mode),
            "add_height_y": float(args.add_height_y),
            "overwrite": bool(args.overwrite),
            "workers": int(args.workers),
            "count": len(results),
            "results": sorted(results, key=lambda x: x["name"]),
        }
        man_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"[OK] wrote manifest: {man_path}")

    print("[OK] done.")


if __name__ == "__main__":
    main()

