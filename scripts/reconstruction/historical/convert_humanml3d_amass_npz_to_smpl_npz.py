#!/usr/bin/env python3
"""
Convert HumanML3D-axisAngle-all AMASS npz files into a unified npz format under this repo,
and provide a check utility for HumanML3D splits/text/feature availability.

Why:
  - In /home/randong/mydata/MCM-LDM/HumanML3D-axisAngle-all/amass_data/**.npz,
    each file contains AMASS-style SMPL-family parameters:
      - poses: (T, 156) axis-angle (SMPL-H: 52 joints * 3)
      - trans: (T, 3)
  - This is close to a "raw parameter" representation and can be exported into a
    consistent, list-friendly dataset under datasets/raw/.

Important note about joint count:
  AMASS SMPL-H 'poses' is (T,156) = 52*3.
  Many pipelines only use the first 22 joints (root + 21 body joints) = 66 dims,
  and treat hands separately.
  This script supports exporting:
    - body22: poses_body (T,22,3) from the first 66 dims (default)
    - smpl24: poses (T,24,3) compatible with SMPL renderers (pad missing hand joints with zeros)
    - full52: poses_full (T,52,3) from all 156 dims

This script is designed to be run by the user because conversion may take a long time.
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


def _read_lines(p: Path) -> List[str]:
    if not p.is_file():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]


def _iter_amass_npz(amass_root: Path) -> Iterable[Path]:
    # AMASS npz are in nested folders; keep deterministic order
    yield from sorted(amass_root.rglob("*.npz"))


@dataclass(frozen=True)
class ConvertStats:
    n_total: int
    n_converted: int
    n_skipped_exists: int
    n_failed: int


def _convert_one(
    src_npz: Path,
    dst_npz: Path,
    *,
    export_mode: str,
    trans_scale: float,
    global_rx_deg: float,
    global_ry_deg: float,
    overwrite: bool,
) -> Tuple[bool, str]:
    if dst_npz.exists() and (not overwrite):
        return False, "exists"

    try:
        z = np.load(src_npz, allow_pickle=False)
        if "poses" not in z.files or "trans" not in z.files:
            return False, "missing_poses_or_trans"

        poses = z["poses"]  # (T,156)
        trans = z["trans"]  # (T,3)
        fps = float(z["mocap_framerate"]) if "mocap_framerate" in z.files else None
        gender = str(z["gender"]) if "gender" in z.files else None
        betas = z["betas"] if "betas" in z.files else None

        if poses.ndim != 2 or poses.shape[1] != 156:
            return False, f"unexpected_poses_shape:{poses.shape}"
        if trans.ndim != 2 or trans.shape[1] != 3:
            return False, f"unexpected_trans_shape:{trans.shape}"

        poses_aa = poses.reshape(poses.shape[0], 52, 3).astype(np.float32, copy=False)
        trans_m = (trans.astype(np.float32, copy=False) * float(trans_scale))

        # Optional coordinate fix (common AMASS->SMPL viewer mismatch):
        # rotate the *global* transform around X/Y axis by global_rx_deg/global_ry_deg.
        # This only affects:
        #   - root global orientation (joint 0)
        #   - translation
        # Body joint rotations remain in local frames and are unchanged.
        if (abs(float(global_rx_deg)) > 1e-6) or (abs(float(global_ry_deg)) > 1e-6):
            rx = np.deg2rad(float(global_rx_deg)).astype(np.float32)
            ry = np.deg2rad(float(global_ry_deg)).astype(np.float32)

            cx, sx = float(np.cos(rx)), float(np.sin(rx))
            cy, sy = float(np.cos(ry)), float(np.sin(ry))

            Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
            Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
            # Apply X first then Y in world space: R = Ry @ Rx
            R = (Ry @ Rx).astype(np.float32)

            def aa_to_mat_batch(aa: np.ndarray) -> np.ndarray:
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

            def mat_to_aa_batch(Rm: np.ndarray) -> np.ndarray:
                """
                Robust rotation-matrix -> axis-angle for a batch.
                Handles the near-pi case to avoid axis instability (sin(theta) ~ 0).
                """
                Rm = Rm.astype(np.float32, copy=False)
                tr = np.trace(Rm, axis1=1, axis2=2)
                cosang = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
                ang = np.arccos(cosang).astype(np.float32)

                aa = np.zeros((Rm.shape[0], 3), dtype=np.float32)
                # Near zero
                m0 = ang < 1e-5
                # Near pi
                mpi = ang > (np.pi - 1e-3)
                mmid = ~(m0 | mpi)

                # Mid-range: standard formula
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

                # Near-pi: compute axis from diagonal, with sign from off-diagonals
                if np.any(mpi):
                    Rpi = Rm[mpi]
                    diag = np.clip((np.diagonal(Rpi, axis1=1, axis2=2) + 1.0) / 2.0, 0.0, 1.0)
                    axis = np.sqrt(diag).astype(np.float32)

                    # Choose signs using off-diagonals (avoid 0/0)
                    # If axis component is ~0, sign doesn't matter.
                    def sgn(x: np.ndarray) -> np.ndarray:
                        return np.where(x >= 0.0, 1.0, -1.0).astype(np.float32)

                    axis[:, 0] *= sgn(Rpi[:, 2, 1] - Rpi[:, 1, 2])
                    axis[:, 1] *= sgn(Rpi[:, 0, 2] - Rpi[:, 2, 0])
                    axis[:, 2] *= sgn(Rpi[:, 1, 0] - Rpi[:, 0, 1])

                    # Normalize (just in case numerical noise)
                    n = np.linalg.norm(axis, axis=1, keepdims=True)
                    axis = axis / (n + 1e-8)
                    aa[mpi] = axis * ang[mpi, None]

                # Near zero already stays 0
                return aa.astype(np.float32)

            aa_root = poses_aa[:, 0, :]  # (T,3)
            Rgo = aa_to_mat_batch(aa_root)
            Rnew = (R[None, :, :] @ Rgo).astype(np.float32)
            poses_aa[:, 0, :] = mat_to_aa_batch(Rnew)
            trans_m = (trans_m @ R.T).astype(np.float32)

        out: Dict[str, object] = {
            "trans": trans_m,
            "mocap_framerate": fps if fps is not None else -1.0,
            "gender": gender if gender is not None else "",
            "src": str(src_npz),
            "export_mode": export_mode,
        }
        if betas is not None:
            b = betas.astype(np.float32, copy=False)
            out["betas"] = b
            # Convenience: SMPL commonly uses 10 betas; keep a consistent view.
            if b.ndim == 1 and b.shape[0] >= 10:
                out["betas10"] = b[:10]

        if export_mode == "body22":
            out["poses"] = poses_aa[:, :22, :]  # (T,22,3) = first 66 dims
        elif export_mode == "smpl24":
            # AMASS SMPL-H poses layout is typically:
            #   root(3) + body_pose(63) + left_hand(45) + right_hand(45) = 156
            # where body joints cover up to wrists (22 joints total incl. root).
            #
            # Our renderer `tools/render_smpl_mesh.py` expects SMPL axis-angle in shape (T,24,3):
            #   global_orient (1 joint) + body_pose (23 joints) = 24 joints.
            #
            # To bridge SMPL-H -> SMPL(24):
            #   - take the first 22 joints (up to wrists)
            #   - append 2 "hand" joints with zero rotation as a neutral default
            # This yields a SMPL-compatible (T,24,3) sequence for visualization/baselines.
            t = poses_aa.shape[0]
            pad = np.zeros((t, 2, 3), dtype=np.float32)
            out["poses"] = np.concatenate([poses_aa[:, :22, :], pad], axis=1)
        elif export_mode == "full52":
            out["poses"] = poses_aa  # (T,52,3)
        else:
            return False, f"bad_export_mode:{export_mode}"

        dst_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst_npz, **out)  # type: ignore[arg-type]
        return True, "ok"
    except Exception as e:
        return False, f"exception:{type(e).__name__}:{e}"


def convert_amass(
    *,
    amass_root: Path,
    out_root: Path,
    export_mode: str,
    trans_scale: float,
    global_rx_deg: float,
    global_ry_deg: float,
    overwrite: bool,
    limit: int,
    only_rel: str,
) -> ConvertStats:
    files = list(_iter_amass_npz(amass_root))
    if only_rel.strip():
        key = only_rel.strip()
        sel: List[Path] = []
        for p in files:
            rel = str(p.relative_to(amass_root))
            rel_posix = rel.replace("\\", "/")
            if rel == key or rel_posix == key or key in rel or key in rel_posix:
                sel.append(p)
        files = sel
    if limit > 0:
        files = files[:limit]

    n_total = len(files)
    n_converted = 0
    n_skipped = 0
    n_failed = 0

    for src in files:
        rel = src.relative_to(amass_root)
        dst = out_root / rel
        ok, reason = _convert_one(
            src,
            dst,
            export_mode=export_mode,
            trans_scale=trans_scale,
            global_rx_deg=global_rx_deg,
            global_ry_deg=global_ry_deg,
            overwrite=overwrite,
        )
        if ok:
            n_converted += 1
        else:
            if reason == "exists":
                n_skipped += 1
            else:
                n_failed += 1

    return ConvertStats(
        n_total=n_total,
        n_converted=n_converted,
        n_skipped_exists=n_skipped,
        n_failed=n_failed,
    )


def check_humanml3d_root(humanml3d_root: Path) -> Dict[str, object]:
    """
    Verify that ids referenced by train/val/test lists exist in texts/ and new_joint_vecs/.
    This does NOT require any extra datasets (e.g., KIT-ML).
    """
    texts = humanml3d_root / "texts"
    feats = humanml3d_root / "new_joint_vecs"
    report: Dict[str, object] = {
        "humanml3d_root": str(humanml3d_root),
        "generated_at": _now(),
        "texts_dir_exists": texts.is_dir(),
        "new_joint_vecs_dir_exists": feats.is_dir(),
        "splits": {},
    }

    for sp in ("train", "val", "test"):
        ids = _read_lines(humanml3d_root / f"{sp}.txt")
        missing_text = []
        missing_feat = []
        for mid in ids:
            if not (texts / f"{mid}.txt").is_file():
                missing_text.append(mid)
            if not (feats / f"{mid}.npy").is_file():
                missing_feat.append(mid)
        report["splits"][sp] = {
            "n_ids": len(ids),
            "missing_text": len(missing_text),
            "missing_feat": len(missing_feat),
            "missing_text_examples": missing_text[:20],
            "missing_feat_examples": missing_feat[:20],
        }
    return report


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_check = sub.add_parser("check", help="Check HumanML3D split/text/feat consistency.")
    ap_check.add_argument(
        "--humanml3d_root",
        type=str,
        default=str(repo_root.parent / "MCM-LDM" / "HumanML3D-axisAngle-all" / "HumanML3D"),
        help="HumanML3D root containing train/val/test.txt, texts/, new_joint_vecs/.",
    )
    ap_check.add_argument(
        "--out_json",
        type=str,
        default=str(repo_root / "outputs" / "humanml3d_split_check.json"),
        help="Where to write the JSON report.",
    )

    ap_conv = sub.add_parser("convert", help="Convert AMASS npz to unified npz under datasets/raw/.")
    ap_conv.add_argument(
        "--amass_root",
        type=str,
        default=str(repo_root.parent / "MCM-LDM" / "HumanML3D-axisAngle-all" / "amass_data"),
        help="Path to HumanML3D-axisAngle-all/amass_data",
    )
    ap_conv.add_argument(
        "--out_root",
        type=str,
        default=str(repo_root / "datasets" / "raw" / "HumanML3D_axisAngle_amass_smplh"),
        help="Output folder under this repo (will mirror AMASS folder structure).",
    )
    ap_conv.add_argument(
        "--export_mode",
        type=str,
        default="body22",
        choices=["body22", "smpl24", "full52"],
        help="body22 exports (T,22,3); smpl24 exports (T,24,3); full52 exports (T,52,3).",
    )
    ap_conv.add_argument(
        "--trans_scale",
        type=float,
        default=1.0,
        help="Scale factor for trans. Use 1.0 if trans is already meters (AMASS typically is).",
    )
    ap_conv.add_argument(
        "--global_rx_deg",
        type=float,
        default=0.0,
        help="Optional: rotate global transform around X (degrees). "
        "Useful if AMASS appears Z-up when rendering with Y-up viewers (try -90 or +90).",
    )
    ap_conv.add_argument(
        "--global_ry_deg",
        type=float,
        default=0.0,
        help="Optional: rotate global transform around Y (degrees). "
        "Useful for flipping facing direction (try 180).",
    )
    ap_conv.add_argument(
        "--only_rel",
        type=str,
        default="",
        help="Optional: only convert files whose relative path matches/contains this string "
        "(relative to amass_root). Useful for small tests.",
    )
    ap_conv.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    ap_conv.add_argument("--limit", type=int, default=-1, help="For debugging: only convert first N files.")
    ap_conv.add_argument(
        "--write_meta",
        action="store_true",
        help="Write a meta.json next to out_root with parameters and counts.",
    )

    args = ap.parse_args()

    if args.cmd == "check":
        root = Path(args.humanml3d_root)
        out_json = Path(args.out_json)
        rep = check_humanml3d_root(root)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] wrote: {out_json}")
        # Also print a concise summary
        for sp, d in rep.get("splits", {}).items():
            print(f"- {sp}: n={d['n_ids']} missing_text={d['missing_text']} missing_feat={d['missing_feat']}")
        return 0

    if args.cmd == "convert":
        amass_root = Path(args.amass_root)
        out_root = Path(args.out_root)
        if not amass_root.is_dir():
            raise SystemExit(f"amass_root not found: {amass_root}")

        stats = convert_amass(
            amass_root=amass_root,
            out_root=out_root,
            export_mode=str(args.export_mode),
            trans_scale=float(args.trans_scale),
            global_rx_deg=float(args.global_rx_deg),
            global_ry_deg=float(args.global_ry_deg),
            overwrite=bool(args.overwrite),
            limit=int(args.limit),
            only_rel=str(args.only_rel),
        )
        print("[OK] convert done")
        print(f"- amass_root: {amass_root}")
        print(f"- out_root:   {out_root}")
        print(f"- export_mode: {args.export_mode}")
        print(f"- trans_scale: {args.trans_scale}")
        print(f"- global_rx_deg: {args.global_rx_deg}")
        print(f"- global_ry_deg: {args.global_ry_deg}")
        print(f"- n_total: {stats.n_total}")
        print(f"- n_converted: {stats.n_converted}")
        print(f"- n_skipped_exists: {stats.n_skipped_exists}")
        print(f"- n_failed: {stats.n_failed}")

        if args.write_meta:
            meta = {
                "generated_at": _now(),
                "amass_root": str(amass_root),
                "out_root": str(out_root),
                "export_mode": str(args.export_mode),
                "trans_scale": float(args.trans_scale),
                "overwrite": bool(args.overwrite),
                "limit": int(args.limit),
                "stats": {
                    "n_total": stats.n_total,
                    "n_converted": stats.n_converted,
                    "n_skipped_exists": stats.n_skipped_exists,
                    "n_failed": stats.n_failed,
                },
                "notes": [
                    "Output mirrors AMASS folder structure under out_root.",
                    "poses are axis-angle per joint: (T,J,3). J=22 for body22; J=52 for full52.",
                    "trans is scaled by trans_scale (default 1.0).",
                ],
            }
            (out_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"[OK] wrote meta: {out_root / 'meta.json'}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

