#!/usr/bin/env python3
"""
Convert Motion-X (smplx_322/*.npy) into this repo's unified SMPL-24 npz format:
  - poses: (T,24,3) axis-angle
  - trans: (T,3)

Why:
  - In this repo, Bandai/LaFAN1 are already exported as SMPL-like npz with poses(T,24,3)+trans(T,3).
  - Motion-X provides SMPL-X parameters packed into a 322-D per-frame vector (T,322).
  - For "body-only" comparisons, we can map SMPL-X -> SMPL24 by:
      root_orient (3)
      + pose_body (63) -> 21 joints
      + pad 2 "hand" joints with zeros -> total 24 joints

Reference (Motion-X official README):
  root_orient: motion[:, :3]
  pose_body:   motion[:, 3:3+63]
  pose_hand:   motion[:, 66:66+90]
  pose_jaw:    motion[:, 66+90:66+93]
  face_expr:   motion[:, 159:159+50]
  face_shape:  motion[:, 209:209+100]
  trans:       motion[:, 309:309+3]
  betas:       motion[:, 312:]

This script is designed to be run by the user (full conversion can take long).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Compatibility patch for old deps (chumpy) that smplx may pull in.
# NumPy 1.24+ removed aliases like np.bool / np.int / np.unicode.
# -----------------------------------------------------------------------------
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "complex"):
    np.complex = complex  # type: ignore[attr-defined]
if not hasattr(np, "object"):
    np.object = object  # type: ignore[attr-defined]
if not hasattr(np, "str"):
    np.str = str  # type: ignore[attr-defined]
if not hasattr(np, "unicode"):
    np.unicode = str  # type: ignore[attr-defined]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _iter_npy(in_root: Path) -> Iterable[Path]:
    yield from sorted(in_root.rglob("*.npy"))


@dataclass(frozen=True)
class ConvertStats:
    n_total: int
    n_converted: int
    n_skipped_exists: int
    n_failed: int


def _aa_to_mat_batch(aa: np.ndarray) -> np.ndarray:
    aa = aa.astype(np.float32, copy=False)
    theta = np.linalg.norm(aa, axis=1, keepdims=True)
    k = aa / (theta + 1e-8)
    k = np.where(theta < 1e-8, 0.0, k)
    K = np.zeros((aa.shape[0], 3, 3), dtype=np.float32)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]
    I = np.eye(3, dtype=np.float32)[None, :, :]
    sin_t = np.sin(theta).astype(np.float32)[:, None]
    cos_t = np.cos(theta).astype(np.float32)[:, None]
    return I + sin_t * K + (1.0 - cos_t) * (K @ K)


def _mat_to_aa_batch(Rm: np.ndarray) -> np.ndarray:
    """
    Robust rotation-matrix -> axis-angle for a batch.
    Handles the near-pi case to avoid axis instability (sin(theta) ~ 0).
    """
    Rm = Rm.astype(np.float32, copy=False)
    tr = np.trace(Rm, axis1=1, axis2=2)
    cosang = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(cosang).astype(np.float32)

    aa = np.zeros((Rm.shape[0], 3), dtype=np.float32)
    m0 = ang < 1e-5
    mpi = ang > (np.pi - 1e-3)
    mmid = ~(m0 | mpi)

    if np.any(mmid):
        w = np.stack(
            [
                Rm[mmid, 2, 1] - Rm[mmid, 1, 2],
                Rm[mmid, 0, 2] - Rm[mmid, 2, 0],
                Rm[mmid, 1, 0] - Rm[mmid, 0, 1],
            ],
            axis=1,
        ).astype(np.float32)
        denom = (2.0 * np.sin(ang[mmid]) + 1e-8)[:, None]
        axis = w / denom
        aa[mmid] = axis * ang[mmid, None]

    if np.any(mpi):
        Rpi = Rm[mpi]
        diag = np.clip((np.diagonal(Rpi, axis1=1, axis2=2) + 1.0) / 2.0, 0.0, 1.0)
        axis = np.sqrt(diag).astype(np.float32)

        def sgn(x: np.ndarray) -> np.ndarray:
            return np.where(x >= 0.0, 1.0, -1.0).astype(np.float32)

        axis[:, 0] *= sgn(Rpi[:, 2, 1] - Rpi[:, 1, 2])
        axis[:, 1] *= sgn(Rpi[:, 0, 2] - Rpi[:, 2, 0])
        axis[:, 2] *= sgn(Rpi[:, 1, 0] - Rpi[:, 0, 1])

        n = np.linalg.norm(axis, axis=1, keepdims=True)
        axis = axis / (n + 1e-8)
        aa[mpi] = axis * ang[mpi, None]

    return aa.astype(np.float32)


def _detect_up_axis_from_smpl24(
    poses: np.ndarray,
    trans: np.ndarray,
    betas10: np.ndarray,
    *,
    smpl_model_root: Path,
    device: str,
    sample_frames: int = 80,
) -> int:
    """
    Detect whether the clip appears to be X/Y/Z-up by sampling SMPL joints and
    checking which axis has the largest root-relative vertical extent.
    """
    try:
        import torch
        import smplx
    except Exception as e:
        raise RuntimeError(
            "auto axis-fix requires torch+smplx. Please run in your mcmldm env."
        ) from e

    if poses.ndim != 3 or poses.shape[1:] != (24, 3):
        raise ValueError(f"expected poses (T,24,3), got {poses.shape}")
    if trans.ndim != 2 or trans.shape[1] != 3 or trans.shape[0] != poses.shape[0]:
        raise ValueError(f"expected trans (T,3) aligned with poses, got {trans.shape}")
    if betas10.shape != (10,):
        betas10 = betas10.reshape(-1)[:10].astype(np.float32, copy=False)

    # smplx.create expects model_root containing "smpl/SMPL_NEUTRAL.pkl"
    if not (smpl_model_root / "smpl" / "SMPL_NEUTRAL.pkl").exists():
        raise FileNotFoundError(f"Missing SMPL model under: {smpl_model_root}/smpl/SMPL_NEUTRAL.pkl")

    bm = smplx.create(
        model_path=str(smpl_model_root),
        model_type="smpl",
        gender="neutral",
        use_pca=False,
        batch_size=1,
    ).to(device)
    bm.eval()

    T = int(poses.shape[0])
    n = min(int(sample_frames), T)
    sel = np.linspace(0, T - 1, num=max(2, n), dtype=int)
    pp = torch.from_numpy(poses[sel]).to(device=device, dtype=torch.float32)
    tt = torch.from_numpy(trans[sel]).to(device=device, dtype=torch.float32)
    bb = torch.from_numpy(betas10[None, :]).to(device=device, dtype=torch.float32).repeat(pp.shape[0], 1)

    with torch.no_grad():
        out = bm(
            global_orient=pp[:, 0:1, :],
            body_pose=pp[:, 1 : 1 + bm.NUM_BODY_JOINTS, :],
            transl=tt,
            betas=bb,
        )
        j = out.joints[:, :22, :].detach().cpu().numpy().astype(np.float32)

    jr = j - j[:, 0:1, :]
    spans = jr.max(axis=(0, 1)) - jr.min(axis=(0, 1))
    return int(np.argmax(spans))


def _rotation_to_y_up(up_axis: int) -> np.ndarray:
    """
    Return a 3x3 rotation matrix R (world) such that applying it to the global
    transform makes the clip Y-up.
    """
    if up_axis == 1:
        return np.eye(3, dtype=np.float32)
    if up_axis == 2:
        # Z-up -> Y-up: rotate around X by -90deg
        return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    if up_axis == 0:
        # X-up -> Y-up: rotate around Z by +90deg
        return np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    raise ValueError(f"invalid up_axis={up_axis}")


def _convert_one(
    src_npy: Path,
    dst_npz: Path,
    *,
    trans_scale: float,
    global_rx_deg: float,
    global_ry_deg: float,
    auto_axis_fix: bool,
    smpl_model_root: Path,
    device: str,
    axis_sample_frames: int,
    overwrite: bool,
) -> Tuple[bool, str]:
    if dst_npz.exists() and (not overwrite):
        return False, "exists"
    try:
        arr = np.load(str(src_npy), allow_pickle=False)
        if arr.ndim != 2 or arr.shape[1] != 322:
            return False, f"unexpected_shape:{arr.shape}"

        T = int(arr.shape[0])
        root_aa = arr[:, 0:3].astype(np.float32, copy=False)
        body = arr[:, 3:66].astype(np.float32, copy=False).reshape(T, 21, 3)
        trans = (arr[:, 309:312].astype(np.float32, copy=False) * float(trans_scale))
        betas10 = arr[0, 312:322].astype(np.float32, copy=False)

        pad2 = np.zeros((T, 2, 3), dtype=np.float32)
        poses = np.concatenate([root_aa[:, None, :], body, pad2], axis=1)  # (T,24,3)

        axis_fix_meta: Dict[str, object] = {}
        if auto_axis_fix:
            up_axis = _detect_up_axis_from_smpl24(
                poses,
                trans,
                betas10,
                smpl_model_root=smpl_model_root,
                device=device,
                sample_frames=axis_sample_frames,
            )
            axis_fix_meta["detected_up_axis"] = int(up_axis)
            if up_axis != 1:
                R = _rotation_to_y_up(int(up_axis))
                Rgo = _aa_to_mat_batch(poses[:, 0, :])
                Rnew = (R[None, :, :] @ Rgo).astype(np.float32)
                poses[:, 0, :] = _mat_to_aa_batch(Rnew)
                trans = (trans @ R.T).astype(np.float32)
                axis_fix_meta["axis_fix_applied"] = True
            else:
                axis_fix_meta["axis_fix_applied"] = False

        # Optional global coordinate / facing fix (rotate root+trans only)
        if (abs(float(global_rx_deg)) > 1e-6) or (abs(float(global_ry_deg)) > 1e-6):
            rx = np.deg2rad(float(global_rx_deg)).astype(np.float32)
            ry = np.deg2rad(float(global_ry_deg)).astype(np.float32)
            cx, sx = float(np.cos(rx)), float(np.sin(rx))
            cy, sy = float(np.cos(ry)), float(np.sin(ry))
            Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
            Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
            R = (Ry @ Rx).astype(np.float32)  # X then Y

            Rgo = _aa_to_mat_batch(poses[:, 0, :])
            Rnew = (R[None, :, :] @ Rgo).astype(np.float32)
            poses[:, 0, :] = _mat_to_aa_batch(Rnew)
            trans = (trans @ R.T).astype(np.float32)

        out: Dict[str, object] = {
            "poses": poses,
            "trans": trans,
            "betas10": betas10,
            "src": str(src_npy),
            "export_mode": "smpl24_from_motionx_smplx322",
            # keep a consistent fps field for downstream
            "mocap_framerate": np.float32(30.0),
        }
        if axis_fix_meta:
            out.update(axis_fix_meta)
        dst_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst_npz, **out)  # type: ignore[arg-type]
        return True, "ok"
    except Exception as e:
        return False, f"exception:{type(e).__name__}:{e}"


def convert_many(
    *,
    in_root: Path,
    out_root: Path,
    trans_scale: float,
    global_rx_deg: float,
    global_ry_deg: float,
    auto_axis_fix: bool,
    smpl_model_root: Path,
    device: str,
    axis_sample_frames: int,
    overwrite: bool,
    limit: int,
    only_rel: str,
) -> ConvertStats:
    files = list(_iter_npy(in_root))
    if only_rel.strip():
        key = only_rel.strip()
        sel: List[Path] = []
        for p in files:
            rel = str(p.relative_to(in_root)).replace("\\", "/")
            if rel == key or key in rel:
                sel.append(p)
        files = sel
    if limit > 0:
        files = files[:limit]

    n_total = len(files)
    n_converted = 0
    n_skipped = 0
    n_failed = 0

    for src in files:
        rel = src.relative_to(in_root)
        dst = (out_root / rel).with_suffix(".npz")
        ok, reason = _convert_one(
            src,
            dst,
            trans_scale=trans_scale,
            global_rx_deg=global_rx_deg,
            global_ry_deg=global_ry_deg,
            auto_axis_fix=auto_axis_fix,
            smpl_model_root=smpl_model_root,
            device=device,
            axis_sample_frames=axis_sample_frames,
            overwrite=overwrite,
        )
        if ok:
            n_converted += 1
        else:
            if reason == "exists":
                n_skipped += 1
            else:
                n_failed += 1
    return ConvertStats(n_total=n_total, n_converted=n_converted, n_skipped_exists=n_skipped, n_failed=n_failed)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_conv = sub.add_parser("convert", help="Convert Motion-X smplx_322 npy to SMPL24 npz.")
    ap_conv.add_argument(
        "--in_root",
        type=str,
        default=str(repo_root / "datasets" / "raw" / "Motion-X" / "smplx_322"),
        help="Input folder containing Motion-X smplx_322 (nested) npy files.",
    )
    ap_conv.add_argument(
        "--out_root",
        type=str,
        default=str(repo_root / "datasets" / "raw" / "Motion-X_smpl24"),
        help="Output folder under this repo (mirrors in_root structure, but with .npz).",
    )
    ap_conv.add_argument(
        "--auto_axis_fix",
        action="store_true",
        help="Auto-detect X/Y/Z-up via SMPL joints and rotate root+trans to Y-up (recommended for Motion-X).",
    )
    ap_conv.add_argument(
        "--smpl_model_root",
        type=str,
        default="/home/randong/mydata/MCM-LDM/deps/motionclip",
        help="Folder containing smpl/SMPL_NEUTRAL.pkl (smplx.create model_path).",
    )
    ap_conv.add_argument("--device", type=str, default="cpu", help="Device for SMPL forward in auto_axis_fix (cpu/cuda).")
    ap_conv.add_argument("--axis_sample_frames", type=int, default=80, help="How many frames to sample for up-axis detection.")
    ap_conv.add_argument(
        "--trans_scale",
        type=float,
        default=1.0,
        help="Scale factor for trans. Motion-X is typically meters, so keep 1.0 unless you know otherwise.",
    )
    ap_conv.add_argument(
        "--global_rx_deg",
        type=float,
        default=0.0,
        help="Optional: rotate global transform around X (degrees).",
    )
    ap_conv.add_argument(
        "--global_ry_deg",
        type=float,
        default=0.0,
        help="Optional: rotate global transform around Y (degrees).",
    )
    ap_conv.add_argument("--only_rel", type=str, default="", help="Only convert files whose relative path matches/contains this.")
    ap_conv.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    ap_conv.add_argument("--limit", type=int, default=-1, help="For debugging: only convert first N files.")
    ap_conv.add_argument(
        "--write_meta",
        action="store_true",
        help="Write a meta.json next to out_root with parameters and counts.",
    )

    args = ap.parse_args()
    if args.cmd == "convert":
        in_root = Path(args.in_root)
        out_root = Path(args.out_root)
        if not in_root.is_dir():
            raise SystemExit(f"in_root not found: {in_root}")

        stats = convert_many(
            in_root=in_root,
            out_root=out_root,
            trans_scale=float(args.trans_scale),
            global_rx_deg=float(args.global_rx_deg),
            global_ry_deg=float(args.global_ry_deg),
            auto_axis_fix=bool(args.auto_axis_fix),
            smpl_model_root=Path(args.smpl_model_root),
            device=str(args.device),
            axis_sample_frames=int(args.axis_sample_frames),
            overwrite=bool(args.overwrite),
            limit=int(args.limit),
            only_rel=str(args.only_rel),
        )
        print("[OK] convert done")
        print(f"- in_root:  {in_root}")
        print(f"- out_root: {out_root}")
        print(f"- trans_scale: {args.trans_scale}")
        print(f"- global_rx_deg: {args.global_rx_deg}")
        print(f"- global_ry_deg: {args.global_ry_deg}")
        print(f"- auto_axis_fix: {bool(args.auto_axis_fix)}")
        if bool(args.auto_axis_fix):
            print(f"- smpl_model_root: {args.smpl_model_root}")
            print(f"- device: {args.device}")
            print(f"- axis_sample_frames: {args.axis_sample_frames}")
        print(f"- n_total: {stats.n_total}")
        print(f"- n_converted: {stats.n_converted}")
        print(f"- n_skipped_exists: {stats.n_skipped_exists}")
        print(f"- n_failed: {stats.n_failed}")

        if args.write_meta:
            meta = {
                "generated_at": _now(),
                "in_root": str(in_root),
                "out_root": str(out_root),
                "export_mode": "smpl24_from_motionx_smplx322",
                "trans_scale": float(args.trans_scale),
                "global_rx_deg": float(args.global_rx_deg),
                "global_ry_deg": float(args.global_ry_deg),
                "auto_axis_fix": bool(args.auto_axis_fix),
                "smpl_model_root": str(args.smpl_model_root),
                "device": str(args.device),
                "axis_sample_frames": int(args.axis_sample_frames),
                "overwrite": bool(args.overwrite),
                "limit": int(args.limit),
                "stats": {
                    "n_total": stats.n_total,
                    "n_converted": stats.n_converted,
                    "n_skipped_exists": stats.n_skipped_exists,
                    "n_failed": stats.n_failed,
                },
                "notes": [
                    "Output mirrors Motion-X smplx_322 folder structure under out_root (but .npz).",
                    "poses is axis-angle per joint: (T,24,3) as SMPL (root + 23).",
                    "Motion-X pose_body provides 21 body joints; we pad 2 hand joints with zeros for SMPL24 compatibility.",
                    "Only root orientation and translation are affected by global_rx_deg/global_ry_deg.",
                    "When auto_axis_fix is enabled, we run SMPL forward on sampled frames and rotate global root+trans to make clips Y-up.",
                ],
            }
            out_root.mkdir(parents=True, exist_ok=True)
            (out_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"[OK] wrote meta: {out_root / 'meta.json'}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

