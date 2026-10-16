#!/usr/bin/env python3
"""
Convert FineMotion-style SMPL npz motions to HumanML3D "new_joint_vecs" (263-dim) format
used by `third_party/mcm-ldm-official`.

Input npz (FineMotion):
  - poses: (T, 24, 3) axis-angle (SMPL 24 joints)
  - trans: (T, 3)
  - mocap_framerate: scalar (usually 30.0)
  - betas10: (10,) optional

Output:
  <out_root>/
    new_joint_vecs/<id>.npy   # (F, 263) float32, typically F = T20 - 1
    texts/<id>.txt            # one-line caption in HumanML3D text format
    viz/<id>.mp4              # optional sanity-check video from recovered joints

Why this script:
  HumanML3D 263-dim representation is not SMPL params; it is computed from 22-joint
  3D joint trajectories after canonicalization (floor, origin, facing Z+, etc).

Notes:
  - This script needs PyTorch + smplx. Use your training env, e.g.:
      conda activate mcmldm
      python tools/convert_finemotion_smpl_npz_to_humanml3d_vecs.py ...
  - It also needs SMPL model files. By default it tries:
      /home/randong/mydata/MCM-LDM/deps/motionclip   (expects smpl/SMPL_NEUTRAL.pkl inside)
    You can override with --smpl_model_root.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# Avoid spamming warnings in large conversions
_CUDA_FALLBACK_WARNED = False


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


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    mcmldm_root: Path
    humanml3d_root: Path


def _resolve_paths() -> Paths:
    repo_root = Path(__file__).resolve().parents[1]
    mcmldm_root = repo_root / "third_party" / "mcm-ldm-official"
    humanml3d_root = mcmldm_root / "datasets" / "humanml3d"
    return Paths(repo_root=repo_root, mcmldm_root=mcmldm_root, humanml3d_root=humanml3d_root)


def _slug(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_\\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "motion"


def caption_from_stem(stem: str) -> str:
    """
    Heuristic caption from filename stem.
    Keeps semantics, drops common metadata suffixes.
    """
    s = stem
    # Common patterns across FineMotion
    s = re.sub(r"_clip_\\d+$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"subject\\d+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"Re$", "", s)
    s = s.replace("__", "_").strip("_")
    s = s.replace("_", " ").strip()
    s = re.sub(r"\\s+", " ", s)
    return s if s else stem


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def linear_resample_positions(
    positions: np.ndarray, src_fps: float, tgt_fps: float
) -> np.ndarray:
    """
    positions: (T, J, 3)
    returns: (T2, J, 3) with linear interpolation in time.
    """
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"positions must be (T,J,3), got {positions.shape}")
    if src_fps <= 0 or tgt_fps <= 0:
        raise ValueError(f"invalid fps src={src_fps} tgt={tgt_fps}")

    T = int(positions.shape[0])
    if T <= 1:
        return positions.copy()

    duration = (T - 1) / float(src_fps)
    T2 = int(np.floor(duration * float(tgt_fps))) + 1
    T2 = max(2, T2)

    t_src = np.arange(T, dtype=np.float64) / float(src_fps)
    t_tgt = np.arange(T2, dtype=np.float64) / float(tgt_fps)

    out = np.empty((T2, positions.shape[1], 3), dtype=np.float32)
    for j in range(positions.shape[1]):
        for c in range(3):
            out[:, j, c] = np.interp(t_tgt, t_src, positions[:, j, c]).astype(np.float32)
    return out


def _detect_up_axis_from_joints(joints22: np.ndarray) -> int:
    """
    Heuristic: pick the joint that is farthest from pelvis on average (often a foot),
    then take the axis of the mean pelvis->that_joint vector with largest abs value.

    Returns:
      0 for X-up, 1 for Y-up, 2 for Z-up
    """
    if joints22.ndim != 3 or joints22.shape[1:] != (22, 3):
        raise ValueError(f"expected joints22 (T,22,3), got {joints22.shape}")
    v = joints22 - joints22[:, 0:1, :]  # root-relative
    d = np.linalg.norm(v, axis=2)  # (T,22)
    idx = int(np.argmax(d.mean(axis=0)))
    mean_vec = v[:, idx, :].mean(axis=0)  # (3,)
    return int(np.argmax(np.abs(mean_vec)))


def _rotate_joints_to_y_up(joints: np.ndarray, up_axis: int) -> np.ndarray:
    """
    Rotate joints positions so that the detected up axis becomes Y.
    This fixes mixed coordinate conventions in Motion-X subsets.
    """
    if up_axis == 1:
        return joints

    # We treat points as row-vectors, so apply: p' = p @ R^T
    if up_axis == 2:
        # Z-up -> Y-up : rotate around X by -90deg
        R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    elif up_axis == 0:
        # X-up -> Y-up : rotate around Z by +90deg (y' = x)
        R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    else:
        raise ValueError(f"invalid up_axis={up_axis}")

    return (joints @ R.T).astype(np.float32)


def find_reference_vec(paths: Paths) -> Optional[Path]:
    vec_dir = paths.humanml3d_root / "new_joint_vecs"
    if not vec_dir.exists():
        return None
    # Prefer canonical id if present
    p = vec_dir / "000021.npy"
    if p.exists():
        return p
    # Else pick any
    for cand in sorted(vec_dir.glob("*.npy"))[:1]:
        return cand
    return None


def init_motion_process(paths: Paths):
    """
    Import mcm-ldm's motion_process and initialize its globals so we can call process_file().
    We derive tgt_offsets from an existing HumanML3D feature file if available, to match
    official skeleton proportions.
    """
    sys.path.insert(0, str(paths.mcmldm_root))
    # pylint: disable=import-error
    import torch  # noqa: WPS433
    from mld.data.humanml.scripts import motion_process as mp  # noqa: WPS433

    # Initialize globals expected by mp.process_file / mp.uniform_skeleton
    mp.joints_num = 22
    mp.l_idx1, mp.l_idx2 = 5, 8
    mp.fid_r, mp.fid_l = [8, 11], [7, 10]
    mp.face_joint_indx = [2, 1, 17, 16]
    mp.r_hip, mp.l_hip = 2, 1
    mp.n_raw_offsets = torch.from_numpy(mp.t2m_raw_offsets)
    mp.kinematic_chain = mp.t2m_kinematic_chain

    ref_vec = find_reference_vec(paths)
    if ref_vec is not None:
        ref = np.load(ref_vec).astype(np.float32)
        # (1, F, 263)
        ref_t = torch.from_numpy(ref).unsqueeze(0)
        # (1, F, 22, 3)
        rec = mp.recover_from_ric(ref_t, joints_num=22).squeeze(0).cpu().numpy()
        example0 = torch.from_numpy(rec[0]).float()
        tgt_skel = mp.Skeleton(mp.n_raw_offsets, mp.kinematic_chain, "cpu")
        mp.tgt_offsets = tgt_skel.get_offsets_joints(example0)
    else:
        # Fallback: unit-length target skeleton (will still work, but proportions differ).
        mp.tgt_offsets = torch.from_numpy(mp.t2m_raw_offsets).float()

    return mp


def smpl_npz_to_joints22(
    npz_path: Path,
    *,
    smpl_model_root: Path,
    device: str = "cpu",
    use_betas10: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Returns:
      joints22: (T, 22, 3) float32, SMPL world joints
      fps: source fps
    """
    import torch  # noqa: WPS433
    import smplx  # noqa: WPS433

    # If user requested CUDA but this environment doesn't support it, fall back to CPU.
    # This keeps correctness; only speed changes.
    global _CUDA_FALLBACK_WARNED  # noqa: PLW0603
    if str(device).startswith("cuda") and (not torch.cuda.is_available()):
        if not _CUDA_FALLBACK_WARNED:
            print(
                f"[WARN] device={device} requested, but torch.cuda.is_available() is False "
                f"(torch.version.cuda={getattr(torch.version, 'cuda', None)}). Using CPU."
            )
            _CUDA_FALLBACK_WARNED = True
        device = "cpu"

    z = np.load(npz_path, allow_pickle=False)
    poses = z["poses"].astype(np.float32)  # (T,24,3)
    trans = z["trans"].astype(np.float32)  # (T,3)
    fps = float(z["mocap_framerate"]) if "mocap_framerate" in z.files else 30.0
    betas10 = z["betas10"].astype(np.float32) if (use_betas10 and "betas10" in z.files) else None

    if poses.ndim != 3 or poses.shape[1:] != (24, 3):
        raise ValueError(f"{npz_path}: expected poses (T,24,3), got {poses.shape}")
    if trans.ndim != 2 or trans.shape[1] != 3 or trans.shape[0] != poses.shape[0]:
        raise ValueError(f"{npz_path}: expected trans (T,3) aligned with poses, got {trans.shape}")

    # smplx.create expects model_root containing "smpl/SMPL_NEUTRAL.pkl"
    bm = smplx.create(
        model_path=str(smpl_model_root),
        model_type="smpl",
        gender="neutral",
        use_pca=False,
        batch_size=1,  # we'll chunk manually
    ).to(device)
    bm.eval()

    T = poses.shape[0]
    joints_out = np.empty((T, 22, 3), dtype=np.float32)

    chunk = 256
    for start in range(0, T, chunk):
        end = min(T, start + chunk)
        p = torch.from_numpy(poses[start:end]).to(device=device, dtype=torch.float32)  # (B,24,3)
        t = torch.from_numpy(trans[start:end]).to(device=device, dtype=torch.float32)  # (B,3)
        B = p.shape[0]

        root = p[:, 0:1, :]  # (B,1,3)
        body = p[:, 1 : 1 + bm.NUM_BODY_JOINTS, :]  # (B,23,3)

        if betas10 is None:
            betas = torch.zeros((B, 10), device=device, dtype=torch.float32)
        else:
            betas = torch.from_numpy(betas10[None, :]).to(device=device, dtype=torch.float32).repeat(B, 1)

        with torch.no_grad():
            out = bm(global_orient=root, body_pose=body, transl=t, betas=betas)
            joints = out.joints[:, :22, :].detach().cpu().numpy().astype(np.float32)  # (B,22,3)
        joints_out[start:end] = joints

    return joints_out, fps


def write_text_file(txt_path: Path, caption: str) -> None:
    """
    HumanML3D text format: caption#tokens#f_tag#to_tag
    To avoid vocabulary issues, we put tokens as 'unk/OTHER'.
    """
    ensure_dir(txt_path.parent)
    line = f"{caption}#unk/OTHER#0.0#0.0\n"
    txt_path.write_text(line, encoding="utf-8")


def _copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copyfile(src, dst)


def _read_first_nonempty_line(p: Path) -> str:
    txt = p.read_text(encoding="utf-8", errors="replace").splitlines()
    for ln in txt:
        ln = ln.strip()
        if ln:
            return ln
    return ""


def write_text_auto(
    *,
    npz_path: Path,
    out_txt: Path,
    caption_fallback: str,
    repo_root: Path,
    humanml3d_text_root: Path,
    motionx_text_root: Path,
    text_mode: str,
) -> None:
    """
    text_mode:
      - filename: always generate from filename (caption_fallback) with unk/OTHER tokens
      - auto: for known datasets, copy/convert original texts when possible
    """
    ensure_dir(out_txt.parent)
    if text_mode == "filename":
        write_text_file(out_txt, caption=caption_fallback)
        return

    parts = set(npz_path.parts)

    # HumanML3D: copy the official caption+tokens file
    if "humanml3d_30hz" in parts:
        src = humanml3d_text_root / f"{npz_path.stem}.txt"
        if src.is_file():
            _copy_file(src, out_txt)
            return

    # Motion-X / Motion-X clean: text is a plain sentence in motionx_seq_text_v1.1
    # We map by relative path under the motionx_* root and wrap into HumanML3D text format.
    for motionx_root_name in ("motionx_clean_30hz", "motionx_30hz"):
        if motionx_root_name in parts:
            # Find the motionx root index in the path
            idx = npz_path.parts.index(motionx_root_name)
            rel = Path(*npz_path.parts[idx + 1 :]).with_suffix(".txt")
            src = motionx_text_root / rel
            if src.is_file():
                caption = _read_first_nonempty_line(src)
                caption = caption if caption else caption_fallback
                write_text_file(out_txt, caption=caption)
                return

    # Fallback
    write_text_file(out_txt, caption=caption_fallback)


def render_joints22_mp4(
    joints: np.ndarray,
    out_mp4: Path,
    *,
    fps: float,
    title: str,
    kinematic_chain: List[List[int]],
    figsize: Tuple[int, int] = (3, 3),
) -> None:
    """
    Robust visualization that does NOT rely on matplotlib.animation.
    We render PNG frames with matplotlib and encode them via system ffmpeg.
    This avoids "empty mp4" issues caused by matplotlib animation writer defaults.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise ValueError(f"expected joints (F,22,3), got {joints.shape}")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH; cannot encode mp4")

    data = joints.astype(np.float32, copy=True)
    # match humanml visual scaling used in their scripts
    data *= 1.3
    # put on floor
    data[:, :, 1] -= float(data[:, :, 1].min())
    # root-relative for stable view
    data[:, :, 0] -= data[:, 0:1, 0]
    data[:, :, 2] -= data[:, 0:1, 2]

    mins = data.min(axis=(0, 1))
    maxs = data.max(axis=(0, 1))
    # generous radius to keep motion in view
    rad = float(max((maxs - mins).max(), 1.0))
    xlim = (-rad * 0.6, rad * 0.6)
    ylim = (0.0, rad * 1.2)
    zlim = (-rad * 0.6, rad * 0.6)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hml_viz_") as tmpdir:
        tmp = Path(tmpdir)
        pattern = tmp / "frame_%05d.png"

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        colors = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]

        for i in range(data.shape[0]):
            ax.cla()
            ax.set_title(title, fontsize=10)
            ax.set_xlim3d(xlim)
            ax.set_ylim3d(ylim)
            ax.set_zlim3d(zlim)
            ax.view_init(elev=120, azim=-90)
            ax.grid(False)
            ax.set_axis_off()

            for ci, chain in enumerate(kinematic_chain):
                c = colors[ci % len(colors)]
                pts = data[i, chain, :]
                ax.plot3D(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=3.0 if ci < 5 else 2.0, color=c)

            fig.savefig(tmp / f"frame_{i:05d}.png", dpi=100, bbox_inches="tight", pad_inches=0.02)

        plt.close(fig)

        # Encode frames -> mp4
        # Ensure even dimensions for h264 by padding in filter.
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(float(fps)),
            "-i",
            str(pattern),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True)


def render_joints22_mp4_with_traj_inset(
    joints: np.ndarray,
    out_mp4: Path,
    *,
    fps: float,
    title: str,
    kinematic_chain: List[List[int]],
    figsize: Tuple[int, int] = (3, 3),
    center_mode: str = "per_frame",
    traj_inset: bool = True,
    traj_inset_rect: Tuple[float, float, float, float] = (0.64, 0.64, 0.33, 0.33),
) -> None:
    """
    Optional experimental preview: body-centered skeleton + 2D root trajectory.
    Current public conversion and dataset-building entrypoints leave this helper unused.
    For the maintained ground-reference example, see docs/RENDERING.md.

    center_mode:
      - per_frame: subtract root XZ for each frame (best for HF/secondary action visibility)
      - start: subtract root XZ of the first frame (keeps some global drift; may go out of view for long clips)
      - none: no centering (not recommended for long clips)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise ValueError(f"expected joints (T,22,3), got {joints.shape}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH; cannot encode mp4")

    # Keep a copy for trajectory evidence (before centering).
    joints0 = joints.astype(np.float32, copy=True)
    traj = joints0[:, 0, [0, 2]].copy()  # (T,2) = (x,z)
    traj -= traj[:1]

    # Main skeleton view (scaled + floored + optionally centered)
    data = joints0.copy()
    data *= 1.3
    data[:, :, 1] -= float(data[:, :, 1].min())

    cm = str(center_mode).strip().lower()
    if cm == "per_frame":
        data[:, :, 0] -= data[:, 0:1, 0]
        data[:, :, 2] -= data[:, 0:1, 2]
    elif cm == "start":
        x0 = float(data[0, 0, 0])
        z0 = float(data[0, 0, 2])
        data[:, :, 0] -= x0
        data[:, :, 2] -= z0
    elif cm == "none":
        pass
    else:
        raise ValueError(f"Unknown center_mode={center_mode!r}. Use per_frame/start/none.")

    mins = data.min(axis=(0, 1))
    maxs = data.max(axis=(0, 1))
    rad = float(max((maxs - mins).max(), 1.0))
    xlim = (-rad * 0.6, rad * 0.6)
    ylim = (0.0, rad * 1.2)
    zlim = (-rad * 0.6, rad * 0.6)

    # Trajectory inset limits (robust even for near-static clips)
    tmins = traj.min(axis=0)
    tmaxs = traj.max(axis=0)
    tspan = np.maximum(1e-3, tmaxs - tmins)
    tpad = np.maximum(0.20, 0.10 * tspan)
    txlim = (float(tmins[0] - tpad[0]), float(tmaxs[0] + tpad[0]))
    tzlim = (float(tmins[1] - tpad[1]), float(tmaxs[1] + tpad[1]))

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hml_viz_traj_") as tmpdir:
        tmp = Path(tmpdir)
        pattern = tmp / "frame_%05d.png"

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax_traj = fig.add_axes(list(traj_inset_rect)) if bool(traj_inset) else None

        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        if ax_traj is not None:
            ax_traj.set_facecolor((1.0, 1.0, 1.0, 0.85))

        colors = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]

        for i in range(data.shape[0]):
            ax.cla()
            ax.set_title(title, fontsize=10)
            ax.set_xlim3d(xlim)
            ax.set_ylim3d(ylim)
            ax.set_zlim3d(zlim)
            ax.view_init(elev=120, azim=-90)
            ax.grid(False)
            ax.set_axis_off()

            for ci, chain in enumerate(kinematic_chain):
                c = colors[ci % len(colors)]
                pts = data[i, chain, :]
                ax.plot3D(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    linewidth=3.0 if ci < 5 else 2.0,
                    color=c,
                )

            # Trajectory inset (xz)
            if ax_traj is not None:
                ax_traj.cla()
                ax_traj.plot(traj[:, 0], traj[:, 1], color="#333333", linewidth=1.0, alpha=0.75)
                ax_traj.scatter([traj[i, 0]], [traj[i, 1]], c="#d62728", s=18)
                ax_traj.set_xlim(*txlim)
                ax_traj.set_ylim(*tzlim)
                ax_traj.set_aspect("equal", adjustable="box")
                ax_traj.set_xticks([])
                ax_traj.set_yticks([])
                for spine in ax_traj.spines.values():
                    spine.set_linewidth(0.6)
                    spine.set_color("#777777")

            fig.savefig(tmp / f"frame_{i:05d}.png", dpi=100, facecolor="white")

        plt.close(fig)

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(float(fps)),
            "-i",
            str(pattern),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=str, nargs="+", required=True, help="Input FineMotion SMPL npz paths")
    ap.add_argument(
        "--out_root",
        type=str,
        default="datasets/FineMotion/humanml3d_20hz_test",
        help="Output root folder",
    )
    ap.add_argument(
        "--smpl_model_root",
        type=str,
        default="/home/randong/mydata/MCM-LDM/deps/motionclip",
        help="Folder containing smpl/SMPL_NEUTRAL.pkl (smplx.create model_path)",
    )
    ap.add_argument("--target_fps", type=float, default=20.0)
    ap.add_argument("--feet_thre", type=float, default=0.002)
    ap.add_argument("--max_frames", type=int, default=220, help="Cap frames before resampling (0 disables)")
    ap.add_argument("--viz", action="store_true", help="Generate mp4 visualization via recovered joints")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--no_axis_fix",
        action="store_true",
        help="Disable auto coordinate fix (e.g., Motion-X Z-up -> Y-up).",
    )
    ap.add_argument(
        "--text_mode",
        type=str,
        default="auto",
        choices=["auto", "filename"],
        help="How to write texts/*.txt. 'auto' will copy/convert original texts for known datasets when possible.",
    )
    ap.add_argument(
        "--humanml3d_text_root",
        type=str,
        default="/home/randong/mydata/MCM-LDM/HumanML3D-axisAngle-all/HumanML3D/texts",
        help="Folder containing HumanML3D texts/*.txt (caption#tokens#...).",
    )
    ap.add_argument(
        "--motionx_text_root",
        type=str,
        default="datasets/raw/Motion-X/motionx_seq_text_v1.1",
        help="Folder containing Motion-X sequence texts (plain sentence per file).",
    )
    return ap.parse_args()


def _normalize_device(device: str) -> str:
    d = str(device).strip()
    if d == "cuda:":
        return "cuda:0"
    if d.lower() in {"gpu"}:
        return "cuda"
    if d.isdigit():
        return f"cuda:{d}"
    return d


def main() -> int:
    args = parse_args()
    args.device = _normalize_device(args.device)
    paths = _resolve_paths()

    out_root = Path(args.out_root)
    out_vec_dir = out_root / "new_joint_vecs"
    out_txt_dir = out_root / "texts"
    out_viz_dir = out_root / "viz"
    ensure_dir(out_vec_dir)
    ensure_dir(out_txt_dir)
    ensure_dir(out_viz_dir)

    smpl_model_root = Path(args.smpl_model_root)
    if not (smpl_model_root / "smpl" / "SMPL_NEUTRAL.pkl").exists():
        raise FileNotFoundError(
            f"SMPL model not found under {smpl_model_root}. Expected {smpl_model_root}/smpl/SMPL_NEUTRAL.pkl"
        )

    mp = init_motion_process(paths)

    # Optional visualizer (robust ffmpeg-based)
    kinematic_chain: Optional[List[List[int]]] = None
    if args.viz:
        sys.path.insert(0, str(paths.mcmldm_root))
        from mld.data.humanml.utils import paramUtil as _paramUtil  # noqa: WPS433

        kinematic_chain = list(_paramUtil.t2m_kinematic_chain)

    for in_path_str in args.inputs:
        npz_path = Path(in_path_str)
        if not npz_path.is_file():
            raise FileNotFoundError(f"missing input: {npz_path}")

        stem = npz_path.stem
        out_id = _slug(stem)
        caption = caption_from_stem(stem)

        joints22_30, src_fps = smpl_npz_to_joints22(
            npz_path, smpl_model_root=smpl_model_root, device=args.device
        )
        if args.max_frames and args.max_frames > 0:
            joints22_30 = joints22_30[: int(args.max_frames)]

        joints22_20 = linear_resample_positions(joints22_30, src_fps=src_fps, tgt_fps=float(args.target_fps))

        # Motion-X (and some other sources) can mix coordinate conventions (e.g. Z-up).
        # HumanML3D feature extraction assumes Y-up for "put on floor".
        if not bool(args.no_axis_fix):
            up_axis = _detect_up_axis_from_joints(joints22_20)
            if up_axis != 1:
                print(f"[axis-fix] {npz_path.name}: detected up_axis={up_axis} -> rotate to Y-up")
            joints22_20 = _rotate_joints_to_y_up(joints22_20, up_axis=up_axis)

        # HumanML3D feature extraction (263)
        data_263, _, _, _ = mp.process_file(joints22_20, float(args.feet_thre))
        data_263 = data_263.astype(np.float32)
        if data_263.ndim != 2 or data_263.shape[1] != 263:
            raise ValueError(f"unexpected feature shape {data_263.shape} for {npz_path}")

        out_vec = out_vec_dir / f"{out_id}.npy"
        np.save(out_vec, data_263)

        out_txt = out_txt_dir / f"{out_id}.txt"
        write_text_auto(
            npz_path=npz_path,
            out_txt=out_txt,
            caption_fallback=caption,
            repo_root=paths.repo_root,
            humanml3d_text_root=Path(args.humanml3d_text_root),
            motionx_text_root=(paths.repo_root / args.motionx_text_root).resolve()
            if not Path(args.motionx_text_root).is_absolute()
            else Path(args.motionx_text_root),
            text_mode=str(args.text_mode),
        )

        if kinematic_chain is not None:
            import torch  # noqa: WPS433

            rec = mp.recover_from_ric(torch.from_numpy(data_263).unsqueeze(0).float(), joints_num=22)
            rec = rec.squeeze(0).cpu().numpy()  # (F,22,3)
            out_mp4 = out_viz_dir / f"{out_id}.mp4"
            render_joints22_mp4(
                rec,
                out_mp4,
                fps=float(args.target_fps),
                title=caption,
                kinematic_chain=kinematic_chain,
            )

        print(f"[OK] {npz_path} -> {out_vec}")

    print(f"Done. Output root: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

