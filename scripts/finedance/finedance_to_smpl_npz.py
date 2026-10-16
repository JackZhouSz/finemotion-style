#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FineDance motion (.npy) -> SMPL NPZ for `render_smpl_mesh.py`.

Input (FineDance dataset):
- `finedance/motion/XXX.npy`  shape (T, 315) typically:
    [trans(3), rot6d(52*6=312)]

Output:
- `.npz` with:
    - trans: (T, 3) float32
    - poses: (T, 24, 3) float32, axis-angle for SMPL (24 joints)

Notes
-----
FineDance stores SMPLH rotations (52 joints) in 6D representation.
We first convert 6D -> axis-angle (SMPLH layout: global(3)+body(63)+lhand(45)+rhand(45)=156),
then follow FineDance's own "smpl" visualization mapping to SMPL:
  body_pose_smpl = [body(63), left_hand(3), right_hand(3)]  -> 69 dims
so that SMPL params are:
  global_orient (3) + body_pose (69) = 72 dims = (24, 3).

Important:
- FineDance/LODGE use SMPLH/SMPLX hand blocks which start from finger joints (e.g. left_index1),
  while SMPL expects two extra joints: lhand/rhand.
- Different projects approximate these two joints differently. LODGE's training features often
  drop hand articulation entirely (hands set to zeros), which is usually more stable.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import wave
from typing import Tuple

import numpy as np


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation to rotation matrix.
    d6: (..., 6)
    returns: (..., 3, 3)
    """
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = _normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    # Match PyTorch3D/LODGE convention: 6D represents the first 2 rows.
    # Therefore we stack along axis=-2 (rows), not axis=-1 (columns).
    return np.stack([b1, b2, b3], axis=-2).astype(np.float32)


def matrix_to_axis_angle(R: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Convert rotation matrix to axis-angle vector.
    R: (..., 3, 3)
    returns: (..., 3) where magnitude is angle (radians).
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"R must have shape (...,3,3), got {R.shape}")

    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = (tr - 1.0) * 0.5
    cos = np.clip(cos, -1.0, 1.0)
    angle = np.arccos(cos)  # (...,)

    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    r = np.stack([rx, ry, rz], axis=-1)  # (...,3)

    aa = np.zeros_like(r, dtype=np.float32)

    small = angle < eps
    if np.any(small):
        # Small-angle approximation: aa ~= 0.5 * vee(R - R^T)
        aa_small = 0.5 * r
        aa = np.where(small[..., None], aa_small.astype(np.float32), aa)

    normal = ~small
    if np.any(normal):
        sin = np.sin(angle)
        denom = 2.0 * sin
        # Handle near-pi separately for numerical stability
        near_pi = normal & (np.abs(angle - math.pi) < 1e-3)
        not_pi = normal & (~near_pi)

        if np.any(not_pi):
            axis = r[not_pi] / (denom[not_pi][..., None] + 1e-8)
            aa[not_pi] = (axis * angle[not_pi][..., None]).astype(np.float32)

        if np.any(near_pi):
            Rp = R[near_pi]
            ang = angle[near_pi]
            # Compute axis from diagonal
            axis = np.empty((Rp.shape[0], 3), dtype=np.float32)
            axis[:, 0] = np.sqrt(np.maximum(0.0, (Rp[:, 0, 0] + 1.0) * 0.5))
            axis[:, 1] = np.sqrt(np.maximum(0.0, (Rp[:, 1, 1] + 1.0) * 0.5))
            axis[:, 2] = np.sqrt(np.maximum(0.0, (Rp[:, 2, 2] + 1.0) * 0.5))
            axis[:, 0] = np.copysign(axis[:, 0], Rp[:, 2, 1] - Rp[:, 1, 2])
            axis[:, 1] = np.copysign(axis[:, 1], Rp[:, 0, 2] - Rp[:, 2, 0])
            axis[:, 2] = np.copysign(axis[:, 2], Rp[:, 1, 0] - Rp[:, 0, 1])
            axis = _normalize(axis)
            aa[near_pi] = (axis * ang[:, None]).astype(np.float32)

    return aa


def finedance_motion_to_smpl_npz(
    motion_npy_path: str,
    out_npz_path: str,
    add_height_y: float = 0.0,
    hand_mode: str = "zeros",
) -> Tuple[int, float | None]:
    """
    Returns: (T, fps_estimate_or_None)
    """
    modata = np.load(motion_npy_path)
    if modata.ndim == 3 and modata.shape[0] == 1:
        modata = modata.squeeze(0)
    if modata.ndim != 2:
        raise ValueError(f"Expected 2D array, got {modata.shape} from {motion_npy_path}")

    # FineDance render.py handles several shapes; we mimic the relevant ones.
    if modata.shape[1] == 319:
        # Some derived formats prepend 4 dims
        modata = modata[:, 4:]

    T, C = modata.shape
    if C not in (315, 159, 69):
        raise ValueError(
            f"Unsupported motion dim C={C} from {motion_npy_path}. "
            "Expected 315 (trans+rot6d), 159 (trans+axis-angle SMPLH), or 69 (trans+axis-angle SMPL)."
        )

    trans = modata[:, :3].astype(np.float32)
    if add_height_y != 0.0:
        trans[:, 1] += float(add_height_y)

    if C == 315:
        rot6d = modata[:, 3:].astype(np.float32)  # (T, 312)
        if rot6d.shape[1] % 6 != 0:
            raise ValueError(f"rot6d dim must be multiple of 6, got {rot6d.shape}")
        J = rot6d.shape[1] // 6
        if J != 52:
            raise ValueError(f"Expected 52 joints for FineDance SMPLH, got J={J} from {motion_npy_path}")
        rot6d = rot6d.reshape(T, J, 6)
        R = rotation_6d_to_matrix(rot6d.reshape(-1, 6)).reshape(T, J, 3, 3)
        axis = matrix_to_axis_angle(R.reshape(-1, 3, 3)).reshape(T, J * 3)  # (T,156) for SMPLH
        axis = axis.astype(np.float32)
    elif C == 159:
        axis = modata[:, 3:].astype(np.float32)  # (T,156) SMPLH axis-angle
    else:  # C == 69
        # Already SMPL axis-angle (global(3)+body(63)=66?) — ambiguous; not used for FineDance raw.
        raise ValueError("C==69 is not expected for FineDance raw motion files.")

    # Build SMPL (24 joints) pose axis-angle
    # axis layout (FineDance): global(3) + body(63) + lhand(45) + rhand(45) = 156
    if axis.shape[1] != 156:
        raise ValueError(f"Expected SMPLH axis dim 156, got {axis.shape} from {motion_npy_path}")

    global_orient = axis[:, 0:3]
    body63 = axis[:, 3:66]
    hand_mode_ = str(hand_mode).strip().lower()
    if hand_mode_ == "index1":
        # Match FineDance/LODGE render.py SMPL approximation:
        # take first joint in each hand block (often left_index1/right_index1) as SMPL lhand/rhand.
        left_hand = axis[:, 66:69]
        right_hand = axis[:, 111:114]
    elif hand_mode_ == "zeros":
        # Match LODGE's common training representation (139 dims): drop hand articulation.
        left_hand = np.zeros((T, 3), dtype=np.float32)
        right_hand = np.zeros((T, 3), dtype=np.float32)
    else:
        raise ValueError("hand_mode must be one of: 'zeros', 'index1'")

    body69 = np.concatenate([body63, left_hand, right_hand], axis=1)  # (T,69)

    poses24 = np.concatenate([global_orient, body69], axis=1).reshape(T, 24, 3).astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(out_npz_path)), exist_ok=True)
    np.savez(out_npz_path, poses=poses24, trans=trans)

    # Estimate fps from matched wav+label_json if present
    fps_est: float | None = None
    stem = os.path.splitext(os.path.basename(motion_npy_path))[0]
    # Prefer label_json frames (they seem to be T-1), fall back to T.
    label_json = os.path.join(os.path.dirname(os.path.dirname(motion_npy_path)), "label_json", stem + ".json")
    wav_path = os.path.join(os.path.dirname(os.path.dirname(motion_npy_path)), "music_wav", stem + ".wav")
    if os.path.exists(wav_path):
        with contextlib.closing(wave.open(wav_path, "rb")) as w:
            dur = w.getnframes() / float(w.getframerate())
        frames = None
        if os.path.exists(label_json):
            try:
                with open(label_json, "r", encoding="utf-8") as f:
                    frames = int(json.load(f).get("frames"))
            except Exception:
                frames = None
        if frames is None:
            frames = int(T)
        if dur > 1e-6:
            fps_est = float(frames / dur)

    return int(T), fps_est


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert FineDance motion npy to SMPL npz for render_smpl_mesh.py")
    ap.add_argument("--in-npy", type=str, required=True, help="Input FineDance motion .npy (e.g. finedance/motion/001.npy)")
    ap.add_argument("--out-npz", type=str, required=True, help="Output .npz path (will contain poses/trans)")
    ap.add_argument(
        "--add-height-y",
        type=float,
        default=0.0,
        help="Optional height offset added to trans[:,1] (FineDance demo code uses +1.3).",
    )
    ap.add_argument(
        "--hand-mode",
        type=str,
        default="zeros",
        choices=["zeros", "index1"],
        help=(
            "How to fill SMPL lhand/rhand joints when converting from SMPLH. "
            "'zeros' (recommended): set them to identity. "
            "'index1': use first joint in each hand block (FineDance/LODGE render.py style)."
        ),
    )
    args = ap.parse_args()

    T, fps_est = finedance_motion_to_smpl_npz(
        args.in_npy,
        args.out_npz,
        add_height_y=args.add_height_y,
        hand_mode=args.hand_mode,
    )
    if fps_est is None:
        print(f"Saved: {args.out_npz}  (T={T})")
    else:
        # Keep output ASCII-only for Windows default encodings.
        print(f"Saved: {args.out_npz}  (T={T}, fps_est~{fps_est:.3f})")


if __name__ == "__main__":
    main()

