#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: Convert MotionBuilder-retargeted SMPL BVH to canonical SMPL BVH + NPZ.

Input:
  d:\\retargetBVH\\newBVHdata\\LaFAN1_to_smpl\\bvhForC\\output\\*Re.bvh

Output:
  d:\\retargetBVH\\newBVHdata\\LaFAN1_to_smpl\\bvhSMPL\\*.bvh
  d:\\retargetBVH\\newBVHdata\\LaFAN1_to_smpl\\npz\\*.npz
"""

import os
import sys

WORK_DIR = os.path.dirname(os.path.abspath(__file__))  # LaFAN1_to_smpl

# Add tempo-changing-music2motion repo root to sys.path so we can import:
#   from dataset.python import smpl_bvh_to_smpl_npz
workspace_root = os.path.abspath(os.path.join(WORK_DIR, "..", ".."))  # d:\retargetBVH
tempo_repo_root = os.path.join(workspace_root, "tempo-changing-music2motion")
sys.path.insert(0, tempo_repo_root)

from dataset.python import smpl_bvh_to_smpl_npz  # type: ignore


def main() -> None:
    mb_out_dir = os.path.join(WORK_DIR, "bvhForC", "output")
    if not os.path.isdir(mb_out_dir):
        print(f"[Error] MotionBuilder output folder not found:\n  {mb_out_dir}")
        print("Please run MotionBuilder script first.")
        return

    mb_files = [f for f in os.listdir(mb_out_dir) if f.lower().endswith(".bvh")]
    if not mb_files:
        print(f"[Error] No .bvh files found in:\n  {mb_out_dir}")
        return

    # Reference SMPL T-pose BVH header
    ref_bvh = os.path.join(
        tempo_repo_root, "dataset", "motionbuilder", "smpl-T.bvh"
    )
    if not os.path.exists(ref_bvh):
        print(f"[Warning] Reference smpl-T.bvh not found, fallback to first MB output file.")
        ref_bvh = os.path.join(mb_out_dir, mb_files[0])

    print(f"Found {len(mb_files)} MotionBuilder BVH files.")
    print(" -> Standardizing SMPL BVH...")
    smpl_bvh_to_smpl_npz.build_smpl_bvh(
        root=WORK_DIR,
        smpl_t_bvh_path=ref_bvh,
        input_bvh_dir="bvhForC/output",
        out_bvh_smpl_dir="bvhSMPL",
    )

    print(" -> Converting to NPZ...")
    smpl_bvh_to_smpl_npz.build_smpl_npz(
        root=WORK_DIR,
        in_bvh_smpl_dir="bvhSMPL",
        out_npz_dir="npz",
        # MotionBuilder BVH translation is typically in centimeters.
        # Convert to meters for downstream SMPL tooling.
        trans_scale=0.01,
    )
    print("Done.")


if __name__ == "__main__":
    main()

