#!/usr/bin/env python3
"""
Export HumanML3D *clip-level* SMPL24 motions that are 1-1 aligned with HumanML3D texts/<id>.txt,
and store them under datasets/FineMotion/humanml3d_30hz/ (or a user-chosen out_root).

CRITICAL CORRECTION (why previous outputs looked broken):
  - HumanML3D index.csv points to ./pose_data/***_poses.npy.
  - These pose_data files are *3D joint positions* (bone-length distances are constant),
    NOT SMPL axis-angle pose parameters.
  - Therefore we MUST NOT treat pose_data as SMPL poses.

What we do instead:
  - Use HumanML3D index.csv only as a *mapping* from clip id -> (source sequence path, start/end on 20fps timeline).
  - For SMPL parameters, we rely on our already-converted AMASS SMPL24 sequences:
      datasets/raw/HumanML3D_axisAngle_amass_smpl24_yup_m90_ry180/**.npz
    which contain:
      poses: (T,24,3) axis-angle
      trans: (T,3)
      mocap_framerate: native fps

Pipeline per clip:
  1) Map source_path (under pose_data/) -> relative AMASS npz path.
  2) Load the AMASS SMPL24 npz and read native fps.
  3) Resample native fps -> clip_fps (default 20fps, HumanML3D timeline).
  4) Slice [start_frame, end_frame) on the 20fps timeline (as defined in index.csv).
  5) Resample that slice 20fps -> 30fps.
  6) Save <clip_id>.npz with poses/trans at 30fps.

Limitation:
  - Some HumanML3D clips originate from HumanAct12 and do not have corresponding AMASS npz.
    Those clip ids will be skipped (optional skip list output).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _atomic_write_npz(dst: Path, data: Dict[str, object], *, compress: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.npz")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass
    if compress:
        np.savez_compressed(tmp, **data)  # type: ignore[arg-type]
    else:
        np.savez(tmp, **data)  # type: ignore[arg-type]
    os.replace(tmp, dst)


def _aa_to_quat(aa: np.ndarray) -> np.ndarray:
    """
    aa: (...,3) axis-angle
    returns quat (...,4) as [w,x,y,z]
    """
    aa = aa.astype(np.float32, copy=False)
    ang = np.linalg.norm(aa, axis=-1, keepdims=True)  # (...,1)
    half = 0.5 * ang
    w = np.cos(half)
    s = np.sin(half)
    axis = aa / (ang + 1e-8)
    axis = np.where(ang < 1e-8, 0.0, axis)
    xyz = axis * s
    return np.concatenate([w, xyz], axis=-1).astype(np.float32)


def _quat_to_aa(q: np.ndarray) -> np.ndarray:
    """
    q: (...,4) [w,x,y,z] normalized (best-effort)
    returns (...,3) axis-angle
    """
    q = q.astype(np.float32, copy=False)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / (n + 1e-8)
    w = np.clip(q[..., 0:1], -1.0, 1.0)
    xyz = q[..., 1:4]
    ang = 2.0 * np.arccos(w)  # (...,1)
    s = np.sqrt(np.maximum(1.0 - w * w, 0.0))  # (...,1)
    axis = xyz / (s + 1e-8)
    axis = np.where(s < 1e-6, 0.0, axis)
    return (axis * ang).astype(np.float32)


def _slerp(q0: np.ndarray, q1: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    q0,q1: (...,4) normalized-ish
    w: scalar or (...,1) interpolation factor in [0,1]
    returns: (...,4)
    """
    q0 = q0.astype(np.float32, copy=False)
    q1 = q1.astype(np.float32, copy=False)
    q0 = q0 / (np.linalg.norm(q0, axis=-1, keepdims=True) + 1e-8)
    q1 = q1 / (np.linalg.norm(q1, axis=-1, keepdims=True) + 1e-8)

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    flip = dot < 0.0
    q1 = np.where(flip, -q1, q1)
    dot = np.where(flip, -dot, dot)
    dot = np.clip(dot, -1.0, 1.0)

    close = dot > 0.9995
    w = np.asarray(w, dtype=np.float32)
    if w.ndim == 0:
        w = w.reshape(1)
    while w.ndim < dot.ndim:
        w = w[..., None]

    theta0 = np.arccos(dot)
    sin0 = np.sin(theta0)
    s0 = np.sin((1.0 - w) * theta0) / (sin0 + 1e-8)
    s1 = np.sin(w * theta0) / (sin0 + 1e-8)
    out = s0 * q0 + s1 * q1
    out = np.where(close, (1.0 - w) * q0 + w * q1, out)
    out = out / (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-8)
    return out.astype(np.float32)


def _resample_trans(trans: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    trans = trans.astype(np.float32, copy=False)
    T = trans.shape[0]
    if T <= 1 or abs(src_fps - dst_fps) < 1e-6:
        return trans.copy()
    t_src = np.arange(T, dtype=np.float32) / np.float32(src_fps)
    T_dst = int(np.floor((T - 1) * (dst_fps / src_fps) + 1.0))
    t_dst = np.arange(T_dst, dtype=np.float32) / np.float32(dst_fps)
    out = np.zeros((T_dst, 3), dtype=np.float32)
    for k in range(3):
        out[:, k] = np.interp(t_dst, t_src, trans[:, k]).astype(np.float32)
    return out


def _resample_poses_slerp(poses_aa: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    poses_aa = poses_aa.astype(np.float32, copy=False)
    T = poses_aa.shape[0]
    if T <= 1 or abs(src_fps - dst_fps) < 1e-6:
        return poses_aa.copy()
    q = _aa_to_quat(poses_aa)  # (T,24,4)
    ratio = float(dst_fps / src_fps)
    T_dst = int(np.floor((T - 1) * ratio + 1.0))
    src_pos = (np.arange(T_dst, dtype=np.float32) / np.float32(dst_fps)) * np.float32(src_fps)
    src_pos = np.clip(src_pos, 0.0, float(T - 1))
    i0 = np.floor(src_pos).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, T - 1)
    w = (src_pos - i0.astype(np.float32)).astype(np.float32)
    out_q = np.zeros((T_dst, 24, 4), dtype=np.float32)
    for t in range(T_dst):
        out_q[t] = _slerp(q[i0[t]], q[i1[t]], w[t])
    return _quat_to_aa(out_q).reshape(T_dst, 24, 3)


@dataclass(frozen=True)
class IndexRow:
    source_path: str
    start_frame: int
    end_frame: int
    clip_id: str


def _read_index_csv(index_csv: Path) -> List[IndexRow]:
    rows: List[IndexRow] = []
    with index_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sp = str(r["source_path"]).strip()
            sf = int(r["start_frame"])
            ef = int(r["end_frame"])
            nn = str(r["new_name"]).strip()
            cid = Path(nn).stem
            rows.append(IndexRow(source_path=sp, start_frame=sf, end_frame=ef, clip_id=cid))
    return rows


def _pose_data_rel_to_amass_npz_rel(source_path: str) -> str:
    s = source_path.strip().replace("\\", "/")
    if s.startswith("./"):
        s = s[2:]
    if s.startswith("pose_data/"):
        s = s[len("pose_data/") :]
    if s.endswith(".npy"):
        s = s[: -len(".npy")] + ".npz"
    else:
        s = s + ".npz"
    return s


def _load_smpl24_npz(p: Path) -> Tuple[np.ndarray, np.ndarray, float]:
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
    fps = float(z["mocap_framerate"]) if "mocap_framerate" in z.files else -1.0
    return poses, trans, fps


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--index_base_dir",
        type=str,
        default="/home/randong/mydata/MCM-LDM/HumanML3D-axisAngle-all",
        help="HumanML3D-axisAngle-all root containing index.csv (used only for mapping).",
    )
    ap.add_argument("--index_csv", type=str, default="", help="Optional: override index.csv path.")
    ap.add_argument(
        "--amass_smpl24_root",
        type=str,
        default=str(repo_root / "datasets" / "raw" / "HumanML3D_axisAngle_amass_smpl24_yup_m90_ry180"),
        help="Converted AMASS SMPL24 root (poses/trans/mocap_framerate).",
    )
    ap.add_argument(
        "--out_root",
        type=str,
        default=str(repo_root / "datasets" / "FineMotion" / "humanml3d_30hz"),
        help="Output folder for HumanML3D clip-level SMPL24 npz named by id (000000.npz).",
    )
    ap.add_argument("--clip_fps", type=float, default=20.0, help="HumanML3D clip fps for index.csv timeline.")
    ap.add_argument("--dst_fps", type=float, default=30.0, help="Target fps (default 30).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    ap.add_argument("--limit", type=int, default=-1, help="Debug: only export first N clips.")
    ap.add_argument("--no_compress", action="store_true", help="Write with np.savez (faster, larger).")
    ap.add_argument("--write_skip_list", action="store_true", help="Write skip_ids.txt for clips without AMASS mapping.")
    args = ap.parse_args()

    index_base_dir = Path(args.index_base_dir)
    index_csv = Path(args.index_csv) if str(args.index_csv).strip() else (index_base_dir / "index.csv")
    if not index_csv.is_file():
        raise SystemExit(f"index.csv not found: {index_csv}")

    amass_root = Path(args.amass_smpl24_root)
    if not amass_root.is_dir():
        raise SystemExit(f"amass_smpl24_root not found: {amass_root}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    compress = not bool(args.no_compress)
    clip_fps = float(args.clip_fps)
    dst_fps = float(args.dst_fps)

    rows = _read_index_csv(index_csv)
    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]

    meta = {
        "generated_at": _now(),
        "index_base_dir": str(index_base_dir),
        "index_csv": str(index_csv),
        "amass_smpl24_root": str(amass_root),
        "out_root": str(out_root),
        "clip_fps": clip_fps,
        "dst_fps": dst_fps,
        "compress": compress,
        "overwrite": bool(args.overwrite),
        "n_rows": len(rows),
        "notes": [
            "pose_data are joint positions; we map to AMASS SMPL24 npz for parameters.",
            "native_fps -> clip_fps -> dst_fps; poses uses quaternion SLERP; trans uses linear interpolation.",
            "clips without AMASS mapping (mostly HumanAct12) are skipped.",
        ],
    }
    (out_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    n_done = 0
    n_skip_exists = 0
    n_skip_nomap = 0
    n_fail = 0
    skip_ids: List[str] = []

    for i, r in enumerate(rows, 1):
        dst = out_root / f"{r.clip_id}.npz"
        if dst.exists() and (not args.overwrite):
            n_skip_exists += 1
            continue

        try:
            rel_npz = _pose_data_rel_to_amass_npz_rel(r.source_path)
            src_npz = amass_root / rel_npz
            if not src_npz.is_file():
                n_skip_nomap += 1
                skip_ids.append(r.clip_id)
                continue

            poses_src, trans_src, native_fps = _load_smpl24_npz(src_npz)
            if native_fps <= 0:
                native_fps = 120.0

            # native -> clip timeline
            poses_clip = _resample_poses_slerp(poses_src, src_fps=native_fps, dst_fps=clip_fps)
            trans_clip = _resample_trans(trans_src, src_fps=native_fps, dst_fps=clip_fps)
            T_clip = min(int(poses_clip.shape[0]), int(trans_clip.shape[0]))
            poses_clip = poses_clip[:T_clip]
            trans_clip = trans_clip[:T_clip]

            # slice on clip timeline
            start = max(0, int(r.start_frame))
            end = int(r.end_frame)
            if end < 0 or end > T_clip:
                end = T_clip
            if start >= end:
                raise ValueError(f"bad slice for {r.clip_id}: start={start} end={end} T_clip={T_clip}")
            poses_seg = poses_clip[start:end]
            trans_seg = trans_clip[start:end]

            # clip -> dst
            poses30 = _resample_poses_slerp(poses_seg, src_fps=clip_fps, dst_fps=dst_fps)
            trans30 = _resample_trans(trans_seg, src_fps=clip_fps, dst_fps=dst_fps)
            T2 = min(int(poses30.shape[0]), int(trans30.shape[0]))
            poses30 = poses30[:T2]
            trans30 = trans30[:T2]

            payload: Dict[str, object] = {
                "poses": poses30.astype(np.float32, copy=False),
                "trans": trans30.astype(np.float32, copy=False),
                "mocap_framerate": np.float32(dst_fps),
                "clip_id": r.clip_id,
                "src_amass_npz": str(src_npz),
                "src_native_fps": float(native_fps),
                "src_index_source": r.source_path,
                "src_index_slice": f"{r.start_frame}:{r.end_frame}",
            }
            _atomic_write_npz(dst, payload, compress=compress)
            n_done += 1
        except Exception:
            n_fail += 1

        if i % 1000 == 0 or i == len(rows):
            print(
                f"[humanml3d_30hz] {i}/{len(rows)} done={n_done} "
                f"skip_exists={n_skip_exists} skip_nomap={n_skip_nomap} fail={n_fail}"
            )

    print("[OK] export finished")
    print(f"- out_root: {out_root}")
    print(f"- done: {n_done}  skip_exists: {n_skip_exists}  skip_nomap: {n_skip_nomap}  failed: {n_fail}")
    if args.write_skip_list and skip_ids:
        p = out_root / "skip_ids.txt"
        p.write_text("\n".join(sorted(skip_ids)) + "\n", encoding="utf-8")
        print(f"[OK] wrote skip list: {p} (n={len(skip_ids)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

