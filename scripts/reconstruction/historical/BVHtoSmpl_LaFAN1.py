#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MotionBuilder script (run inside Autodesk MotionBuilder) for retargeting
LaFAN1 BVH clips to the provided SMPL skeleton.

What it does (per BVH clip):
1) Open SMPL T-pose FBX (with a Character already set up)
2) Import BVH clip (expects a prepended T-pose as first frame)
3) Create & characterize a new Character for the BVH skeleton (standard HIK slots)
4) Retarget + bake animation onto SMPL skeleton
5) Export SMPL BVH to: <bvhForC>/output/<clip>Re.bvh

Input folder expected (created by the Python Step1 script):
  d:\\retargetBVH\\newBVHdata\\LaFAN1_to_smpl\\bvhForC
"""

import os

import pyfbsdk
from pyfbsdk import (
    FBApplication,
    FBCharacter,
    FBCharacterInputType,
    FBCharacterPlotWhere,
    FBModelSkeleton,
    FBPlotOptions,
    FBSystem,
)


def _find_repo_root(start_dir):
    """Walk up parents until we find 'tempo-changing-music2motion'."""
    cur = os.path.abspath(start_dir)
    while True:
        candidate = os.path.join(cur, "tempo-changing-music2motion", "dataset", "motionbuilder", "smpl-male-T-pose.fbx")
        if os.path.exists(candidate):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = _find_repo_root(SCRIPT_DIR)

FOLDER_PATH = os.path.join(SCRIPT_DIR, "bvhForC")  # LaFAN1_to_smpl/bvhForC
OUTPUT_DIR_NAME = "output"

# Batch mode: leave empty to process every BVH in bvhForC.
# To test one file, set its filename, for example "walk4_subject1.bvh".
ONLY_BVH_NAME = ""

# Set True to clear the output directory before each batch.
CLEAR_OUTPUT_DIR = True

if REPO_ROOT is None:
    msg = (
        "[Error] Could not locate repo root by searching for:\n"
        "  tempo-changing-music2motion/dataset/motionbuilder/smpl-male-T-pose.fbx\n"
        "Please keep this script under d:\\retargetBVH\\newBVHdata\\LaFAN1_to_smpl\\"
    )
    print(msg)
    pyfbsdk.FBMessageBox("Script Error", msg, "OK")
    FBX_T_POSE_PATH = None
else:
    FBX_T_POSE_PATH = os.path.join(
        REPO_ROOT, "tempo-changing-music2motion", "dataset", "motionbuilder", "smpl-male-T-pose.fbx"
    )

print(f"DEBUG: FOLDER_PATH = {FOLDER_PATH}")
print(f"DEBUG: FBX_T_POSE_PATH = {FBX_T_POSE_PATH}")

if not os.path.exists(FOLDER_PATH):
    msg = f"[Error] Input folder not found:\n{FOLDER_PATH}\n\nPlease run the Python Step1 script first."
    print(msg)
    pyfbsdk.FBMessageBox("Script Error", msg, "OK")
    FOLDER_PATH = None

if FBX_T_POSE_PATH is not None and not os.path.exists(FBX_T_POSE_PATH):
    msg = f"[Error] SMPL FBX not found:\n{FBX_T_POSE_PATH}"
    print(msg)
    pyfbsdk.FBMessageBox("Script Error", msg, "OK")
    FBX_T_POSE_PATH = None


# Standard LaFAN1 BVH joint naming -> MotionBuilder Character slots
# (slot_name + "Link" must exist on FBCharacter.PropertyList)
HIK_SLOT_TO_JOINT = {
    "Hips": "Hips",
    "LeftUpLeg": "LeftUpLeg",
    "LeftLeg": "LeftLeg",
    "LeftFoot": "LeftFoot",
    "LeftToeBase": "LeftToeBase",
    "RightUpLeg": "RightUpLeg",
    "RightLeg": "RightLeg",
    "RightFoot": "RightFoot",
    "RightToeBase": "RightToeBase",
    "Spine": "Spine",
    "Spine1": "Spine1",
    "Spine2": "Spine2",
    "Neck": "Neck",
    "Head": "Head",
    "LeftShoulder": "LeftShoulder",
    "LeftArm": "LeftArm",
    "LeftForeArm": "LeftForeArm",
    "LeftHand": "LeftHand",
    "RightShoulder": "RightShoulder",
    "RightArm": "RightArm",
    "RightForeArm": "RightForeArm",
    "RightHand": "RightHand",
}


def _find_joint_by_name(joint_name, num, file_base=None):
    """
    BVH import often namespaces joints like:
      'BVH 1:Hips'
    This helper tries a few common patterns, then falls back to suffix match.
    """
    # Try direct API lookup first (some MB versions provide it)
    try:
        if hasattr(pyfbsdk, "FBFindModelByName"):
            obj = pyfbsdk.FBFindModelByName(joint_name)
            if obj:
                return obj
    except Exception:
        pass

    candidates = [
        f"BVH {num}:{joint_name}",
        f"BVH {num}::{joint_name}",
        f"BVH:{num}:{joint_name}",
        f"BVH:{joint_name}",
        joint_name,
    ]
    if file_base:
        candidates.extend(
            [
                f"{file_base}:{joint_name}",
                f"{file_base}::{joint_name}",
                f"BVH {num}:{file_base}:{joint_name}",
                f"BVH {num}:{file_base}::{joint_name}",
            ]
        )

    for label in candidates:
        try:
            obj = pyfbsdk.FBFindModelByLabelName(label)
            if obj:
                return obj
        except Exception:
            pass

    # Fallback: scan skeletons and match suffix / exact name
    for skel in FBSystem().Scene.ModelSkeletons:
        try:
            name = getattr(skel, "Name", "")
            label = getattr(skel, "LabelName", "")
            long_name = getattr(skel, "LongName", "")

            if name == joint_name or label == joint_name or long_name == joint_name:
                return skel

            for s in (name, label, long_name):
                if not s:
                    continue
                if s.endswith(":" + joint_name) or s.endswith("::" + joint_name):
                    return skel
                if ":" in s and s.split(":")[-1] == joint_name:
                    return skel
        except Exception:
            continue
    return None


def _debug_print_some_skeleton_names(limit=30):
    """Print a small sample of skeleton naming for debugging."""
    try:
        skels = list(FBSystem().Scene.ModelSkeletons)
    except Exception:
        return
    print(f"DEBUG: ModelSkeletons count = {len(skels)}")
    for i, skel in enumerate(skels[:limit]):
        try:
            print(
                "DEBUG: skel[%d] Name='%s' Label='%s' LongName='%s'"
                % (
                    i,
                    getattr(skel, "Name", ""),
                    getattr(skel, "LabelName", ""),
                    getattr(skel, "LongName", ""),
                )
            )
        except Exception:
            continue


def characterize_bvh_character(character_name, num, file_base=None):
    """Create and characterize a Character for the imported BVH skeleton."""
    new_character = FBCharacter(f"{character_name}{num}")

    missing = []
    for slot_name, joint_name in HIK_SLOT_TO_JOINT.items():
        mapping_slot = new_character.PropertyList.Find(slot_name + "Link")
        # IMPORTANT: do not use truthiness here; empty slots can evaluate False.
        if mapping_slot is None:
            continue
        joint_obj = _find_joint_by_name(joint_name, num=num, file_base=file_base)
        if joint_obj:
            mapping_slot.append(joint_obj)
            try:
                print(
                    "DEBUG: mapped slot '%s' -> Name='%s' Label='%s' LongName='%s'"
                    % (
                        slot_name,
                        getattr(joint_obj, "Name", ""),
                        getattr(joint_obj, "LabelName", ""),
                        getattr(joint_obj, "LongName", ""),
                    )
                )
            except Exception:
                pass
        else:
            missing.append(slot_name)

    if missing:
        print(f"[Error] Could not find BVH joints for slots: {missing}")
        _debug_print_some_skeleton_names(limit=40)
        pyfbsdk.FBMessageBox(
            "Characterize Error",
            "Could not find required BVH joints. See console for details.\n\nMissing slots:\n- "
            + "\n- ".join(missing),
            "OK",
        )
        return new_character, False

    characterized = new_character.SetCharacterizeOn(True)
    if not characterized:
        print(new_character.GetCharacterizeError())
    else:
        FBApplication().CurrentCharacter = new_character
    return new_character, bool(characterized)


def plot_character_to_skeleton():
    """Bake the current Character's animation onto its skeleton."""
    l_character = FBApplication().CurrentCharacter

    plot_options = FBPlotOptions()
    plot_options.ConstantKeyReducerKeepOneKey = False
    plot_options.PlotAllTakes = False
    plot_options.PlotOnFrame = True
    plot_options.PlotPeriod = pyfbsdk.FBTime(0, 0, 0, 1)
    plot_options.PlotTranslationOnRootOnly = False
    plot_options.PreciseTimeDiscontinuities = False
    plot_options.RotationFilterToApply = pyfbsdk.FBRotationFilter.kFBRotationFilterUnroll
    plot_options.UseConstantKeyReducer = False

    l_character.PlotAnimation(
        FBCharacterPlotWhere.kFBCharacterPlotOnSkeleton,
        plot_options,
    )


def activate_smpl_character(character_name="Character"):
    """Activate the SMPL Character from the loaded FBX (typically named 'Character')."""
    for l_char in FBSystem().Scene.Characters:
        if l_char.Name == character_name:
            l_char.InputCharacter = FBApplication().CurrentCharacter
            l_char.InputType = FBCharacterInputType.kFBCharacterInputCharacter
            l_char.ActiveInput = True
            FBApplication().CurrentCharacter = l_char


def select_descendants(skeleton):
    skeleton.Selected = True
    for child in skeleton.Children:
        if isinstance(child, FBModelSkeleton):
            select_descendants(child)


def export_bvh(file_path):
    return FBApplication().FileExport(file_path)


def retarget_and_export(bvh_file_path, out_bvh_path, file_base):
    app = FBApplication()

    # Start clean per clip
    app.FileNew()

    # Open SMPL T-pose FBX
    if not app.FileOpen(FBX_T_POSE_PATH):
        raise RuntimeError(f"Could not open FBX: {FBX_T_POSE_PATH}")

    # Import BVH
    if not app.FileImport(bvh_file_path):
        raise RuntimeError(f"Could not import BVH: {bvh_file_path}")

    # Characterize BVH skeleton (namespace usually 'BVH 1:')
    bvh_char, characterized = characterize_bvh_character("BVHCharacter", num=1, file_base=file_base)
    if not characterized:
        raise RuntimeError("BVH Characterization failed (missing joint mapping).")

    # Retarget to SMPL character and bake
    activate_smpl_character("Character")
    plot_character_to_skeleton()

    # Select SMPL skeleton (Pelvis root) for clean export
    for skeleton in FBSystem().Scene.ModelSkeletons:
        if skeleton.Name == "Pelvis":
            select_descendants(skeleton)
            break

    if not export_bvh(out_bvh_path):
        raise RuntimeError(f"Could not export BVH: {out_bvh_path}")


def main():
    if FOLDER_PATH is None or FBX_T_POSE_PATH is None:
        print("Script execution aborted due to path errors.")
        return

    output_dir = os.path.join(FOLDER_PATH, OUTPUT_DIR_NAME)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif CLEAR_OUTPUT_DIR:
        # Best-effort cleanup: remove existing BVH outputs to avoid mixing runs
        try:
            for f in os.listdir(output_dir):
                if f.lower().endswith(".bvh"):
                    os.remove(os.path.join(output_dir, f))
            print(f"DEBUG: Cleared existing BVH files in: {output_dir}")
        except Exception as e:
            print(f"[Warning] Failed to clear output dir: {e}")

    all_files = os.listdir(FOLDER_PATH)
    bvh_files = [f for f in all_files if f.lower().endswith(".bvh")]
    if not bvh_files:
        msg = f"No .bvh files found in:\n{FOLDER_PATH}"
        print(msg)
        pyfbsdk.FBMessageBox("Warning", msg, "OK")
        return

    if ONLY_BVH_NAME:
        bvh_files = [f for f in bvh_files if f == ONLY_BVH_NAME]
        if not bvh_files:
            msg = f"[Error] ONLY_BVH_NAME not found in bvhForC:\n{ONLY_BVH_NAME}"
            print(msg)
            pyfbsdk.FBMessageBox("Warning", msg, "OK")
            return

    success_count = 0
    for fname in bvh_files:
        file_base = os.path.splitext(fname)[0]
        bvh_path = os.path.join(FOLDER_PATH, fname)
        out_bvh_path = os.path.join(output_dir, file_base + "Re.bvh")
        try:
            print(f"Processing: {fname}")
            retarget_and_export(bvh_path, out_bvh_path, file_base=file_base)
            print(f"Exported: {out_bvh_path}")
            success_count += 1
        except Exception as e:
            print(f"[Error] {fname}: {e}")

    pyfbsdk.FBMessageBox(
        "Process Completed",
        f"Successfully processed {success_count}/{len(bvh_files)} files.\nCheck output folder:\n{output_dir}",
        "OK",
    )


main()

