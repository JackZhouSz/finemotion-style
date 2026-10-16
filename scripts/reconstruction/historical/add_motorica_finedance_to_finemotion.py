#!/usr/bin/env python3
"""
Add Motorica and FineDance (already SMPL24 npz) into datasets/FineMotion at 30Hz.

Expected inputs
--------------
datasets/raw/Motorica_to_smpl/npz/*.npz
  - poses: (T,24,3)
  - trans: (T,3)
  - (assumed 120fps) -> downsample by factor=4

datasets/raw/FineDance_to_smpl/npz/*.npz
  - poses: (T,24,3)
  - trans: (T,3)
  - (assumed 30fps)  -> factor=1 (copy as-is)

Outputs
-------
datasets/FineMotion/motorica_30hz/**.npz
datasets/FineMotion/finedance_30hz/**.npz

Each output npz contains:
  - poses: (T',24,3) float32
  - trans: (T',3) float32
  - mocap_framerate: float32(30.0)
  - betas10: (10,) float32 if present in source

Writes atomically (.tmp.npz -> replace) to reduce corruption risk.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _iter_npz(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.npz"))


def _extract_betas10(z: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
    if "betas10" in z.files:
        b = np.asarray(z["betas10"], dtype=np.float32)
        return b.reshape(-1)[:10]
    if "betas" in z.files:
        b = np.asarray(z["betas"], dtype=np.float32)
        return b.reshape(-1)[:10]
    return None


def _load_motion_npz(p: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    z = np.load(p, allow_pickle=True)
    if "poses" not in z.files or "trans" not in z.files:
        raise KeyError(f"missing poses/trans in {p} (keys={list(z.files)})")
    poses = np.asarray(z["poses"], dtype=np.float32)
    trans = np.asarray(z["trans"], dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (24, 3):
        raise ValueError(f"bad poses shape in {p}: {poses.shape} (expected (T,24,3))")
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"bad trans shape in {p}: {trans.shape} (expected (T,3))")
    if poses.shape[0] != trans.shape[0]:
        raise ValueError(f"frame mismatch in {p}: poses={poses.shape[0]} trans={trans.shape[0]}")
    betas10 = _extract_betas10(z)
    return poses, trans, betas10


def _downsample(poses: np.ndarray, trans: np.ndarray, factor: int) -> Tuple[np.ndarray, np.ndarray]:
    if factor <= 1:
        return poses, trans
    return poses[::factor].copy(), trans[::factor].copy()


def _atomic_write_npz(dst: Path, payload: Dict[str, object], *, compress: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.npz")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass
    if compress:
        np.savez_compressed(tmp, **payload)  # type: ignore[arg-type]
    else:
        np.savez(tmp, **payload)  # type: ignore[arg-type]
    os.replace(tmp, dst)


@dataclass(frozen=True)
class Job:
    key: str
    in_root: Path
    out_root: Path
    factor: int
    assumed_src_fps: int


@dataclass
class Stats:
    n_total: int = 0
    n_done: int = 0
    n_skipped_exists: int = 0
    n_failed: int = 0


def _run_job(job: Job, *, overwrite: bool, compress: bool, limit: int) -> Stats:
    st = Stats()
    files = list(_iter_npz(job.in_root))
    if limit > 0:
        files = files[: int(limit)]
    st.n_total = len(files)

    for i, src in enumerate(files, 1):
        rel = src.relative_to(job.in_root)
        dst = job.out_root / rel
        if dst.exists() and (not overwrite):
            st.n_skipped_exists += 1
            continue
        try:
            poses, trans, betas10 = _load_motion_npz(src)
            poses2, trans2 = _downsample(poses, trans, int(job.factor))
            payload: Dict[str, object] = {
                "poses": poses2.astype(np.float32, copy=False),
                "trans": trans2.astype(np.float32, copy=False),
                "mocap_framerate": np.float32(30.0),
            }
            if betas10 is not None:
                payload["betas10"] = betas10.astype(np.float32, copy=False)
            _atomic_write_npz(dst, payload, compress=compress)
            st.n_done += 1
        except Exception:
            st.n_failed += 1
        if i % 200 == 0 or i == st.n_total:
            print(f"[{job.key}] {i}/{st.n_total} done={st.n_done} skip={st.n_skipped_exists} fail={st.n_failed}")
    return st


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--finemotion_root", type=str, default=str(repo_root / "datasets" / "FineMotion"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no_compress", action="store_true", help="Use np.savez (faster, larger) instead of np.savez_compressed.")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="Only run one dataset: motorica|finedance (default: run both).",
    )
    args = ap.parse_args()

    finemotion_root = Path(args.finemotion_root)
    finemotion_root.mkdir(parents=True, exist_ok=True)
    compress = not bool(args.no_compress)
    only = str(args.only).strip().lower()

    jobs: List[Job] = [
        Job(
            key="motorica_30hz",
            in_root=repo_root / "datasets" / "raw" / "Motorica_to_smpl" / "npz",
            out_root=finemotion_root / "motorica_30hz",
            factor=4,
            assumed_src_fps=120,
        ),
        Job(
            key="finedance_30hz",
            in_root=repo_root / "datasets" / "raw" / "FineDance_to_smpl" / "npz",
            out_root=finemotion_root / "finedance_30hz",
            factor=1,
            assumed_src_fps=30,
        ),
    ]

    if only:
        if only not in ("motorica", "finedance"):
            raise SystemExit("--only must be motorica or finedance")
        jobs = [j for j in jobs if j.key.startswith(only)]

    meta_all: Dict[str, object] = {
        "generated_at": _now(),
        "finemotion_root": str(finemotion_root),
        "compress": compress,
        "overwrite": bool(args.overwrite),
        "limit": int(args.limit),
        "jobs": [],
        "notes": [
            "Motorica assumed 120fps -> downsample by 4 to 30Hz.",
            "FineDance assumed 30fps -> copied as-is.",
            "Output mocap_framerate is set to 30.0 for consistency with other FineMotion datasets.",
        ],
    }

    for job in jobs:
        if not job.in_root.is_dir():
            raise SystemExit(f"in_root not found for {job.key}: {job.in_root}")
        print(f"\n=== Running: {job.key} ===")
        print(f"- in_root:  {job.in_root}")
        print(f"- out_root: {job.out_root}")
        print(f"- factor: {job.factor} (assumed src_fps={job.assumed_src_fps} -> 30)")
        st = _run_job(job, overwrite=bool(args.overwrite), compress=compress, limit=int(args.limit))
        # write per-dataset meta
        ds_meta = {
            "generated_at": meta_all["generated_at"],
            "dataset": job.key,
            "in_root": str(job.in_root),
            "out_root": str(job.out_root),
            "assumed_src_fps": int(job.assumed_src_fps),
            "downsample_factor": int(job.factor),
            "compress": compress,
            "overwrite": bool(args.overwrite),
            "limit": int(args.limit),
            "stats": {
                "n_total": st.n_total,
                "n_done": st.n_done,
                "n_skipped_exists": st.n_skipped_exists,
                "n_failed": st.n_failed,
            },
        }
        job.out_root.mkdir(parents=True, exist_ok=True)
        (job.out_root / "meta.json").write_text(json.dumps(ds_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        meta_all["jobs"].append(ds_meta)

    # write a small record at finemotion_root (does not replace the existing FineMotion/meta.json)
    # put under a different name to avoid overwriting user's global meta.json.
    extras_meta_path = finemotion_root / "meta_extras_motorica_finedance.json"
    extras_meta_path.write_text(json.dumps(meta_all, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[OK] wrote: {extras_meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

