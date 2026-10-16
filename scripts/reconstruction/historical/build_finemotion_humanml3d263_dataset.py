#!/usr/bin/env python3
"""
Build a HumanML3D-style dataset (263-dim) for MCM-LDM training.

Public metadata adaptation: the embedded LaFAN1 README theme prose is removed.
LaFAN1 inputs require --lafan-theme-map pointing to user-acquired local source
metadata. It is not bundled or relicensed here. See ../README.md and the public
manifest for the original and adapted hashes. Motion processing is unchanged.

Goal:
  Convert multiple FineMotion SMPL-24 npz sources into HumanML3D motion features:
    - new_joint_vecs/*.npy  (F,263) at 20Hz, with F <= 196 (safe for loader which drops >=200)
    - texts/*.txt           (line format: caption#tokens#f_tag#to_tag)
    - train/val/test splits (split_root/*.txt listing ids without extension)
    - Mean.npy / Std.npy    (dataset-level normalization stats)

Datasets supported (6 sources):
  1) humanml3d_30hz:
     - text source: /home/randong/mydata/MCM-LDM/HumanML3D-axisAngle-all/HumanML3D/texts/<id>.txt
     - if we have to segment (rare), we take the FIRST line, keep its tokens, and set tags to 0.0#0.0.
  2) motionx_clean_30hz:
     - text source: datasets/raw/Motion-X/motionx_seq_text_v1.1/<relpath>.txt (plain sentence)
     - we wrap as: caption#unk/OTHER#0.0#0.0
  3) bandai_30hz:
     - derive caption from filename: "<content> in a <style> manner"
     - tokens: unk/OTHER
  4) lafan1_30hz:
     - derive caption from the explicitly supplied local source theme map
     - tokens: unk/OTHER
  5) motorica_30hz:
     - derive caption from filename stem (best-effort; mostly metadata)
     - tokens: unk/OTHER
  6) finedance_30hz:
     - derive caption from filename stem (usually numeric id)
     - tokens: unk/OTHER

Why segment during 263 conversion:
  HumanML3D loader drops any motion with len>=200. It's easiest/safer to segment
  after resampling to 20Hz but before feature extraction, so generated features
  always satisfy length constraints.

Run (recommended env):
  conda activate mcmldm

Example:
  python tools/build_finemotion_humanml3d263_dataset.py \\
    --out_root datasets/FineMotion/humanml3d_20hz_for_Tranning263/finemotion_263_v1 \\
    --include humanml3d_30hz bandai_30hz lafan1_30hz motionx_clean_30hz \\
    --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


# Reuse robust conversion utilities (SMPL forward, resample, axis fix, renderer if needed)
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from convert_finemotion_smpl_npz_to_humanml3d_vecs import (  # noqa: E402
    _detect_up_axis_from_joints,
    _rotate_joints_to_y_up,
    _resolve_paths as _conv_resolve_paths,
    init_motion_process,
    linear_resample_positions,
    smpl_npz_to_joints22,
)


@dataclass(frozen=True)
class SourceCfg:
    key: str
    npz_root: Path
    # text handling
    text_kind: str  # humanml3d_copy | motionx_copy | bandai_name | lafan_theme | motorica_name | finedance_name
    # optional motionx text root (relative path mapping)
    motionx_text_root: Optional[Path] = None
    # optional humanml3d text root
    humanml3d_text_root: Optional[Path] = None


def _load_lafan_theme_map(path: Path) -> Dict[str, str]:
    """Read user-local {normalized_theme_key: upstream_description} metadata."""
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or not value:
        raise ValueError("LaFAN1 theme map must be a nonempty JSON object")
    result: Dict[str, str] = {}
    for key, description in value.items():
        if not isinstance(key, str) or re.fullmatch(r"[a-z]+", key) is None:
            raise ValueError("LaFAN1 theme keys must be lowercase letters")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"LaFAN1 theme {key!r} requires a source description")
        if any(character in description for character in "#\r\n"):
            raise ValueError(f"LaFAN1 theme {key!r} contains a separator or line break")
        result[key] = description
    return result


def _preflight_lafan_metadata(include: Sequence[str], repo: Path,
                             map_path: Optional[Path], limit_per_source: int = -1) -> Tuple[Dict[str, str], Optional[str]]:
    """Resolve every selected LaFAN1 sequence before writing or SMPL loading."""
    if "lafan1_30hz" not in include:
        return {}, None
    if map_path is None:
        raise ValueError("LaFAN1 inputs require --lafan-theme-map with user-acquired metadata")
    source_root = repo / "datasets" / "FineMotion" / "lafan1_30hz"
    if not source_root.is_dir():
        raise ValueError(f"Missing source root: {source_root}")
    theme_map = _load_lafan_theme_map(map_path)
    selected = sorted(source_root.rglob("*.npz"))
    if limit_per_source > 0:
        selected = selected[:limit_per_source]
    for source in selected:
        lafan_caption_from_stem(source.stem, theme_map=theme_map)
    return theme_map, hashlib.sha256(map_path.read_bytes()).hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _stable_hash01(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(16**12)


def split_ids(ids: Sequence[str], *, seed: int, ratios=(0.8, 0.1, 0.1)) -> Dict[str, List[str]]:
    a, b, c = ratios
    assert abs((a + b + c) - 1.0) < 1e-6
    out = {"train": [], "val": [], "test": []}
    for mid in ids:
        x = _stable_hash01(f"{seed}:{mid}")
        if x < a:
            out["train"].append(mid)
        elif x < a + b:
            out["val"].append(mid)
        else:
            out["test"].append(mid)
    for k in out:
        out[k] = sorted(out[k])
    return out


def write_split_files(split_root: Path, splits: Dict[str, List[str]]) -> None:
    _ensure_dir(split_root)
    for k, ids in splits.items():
        (split_root / f"{k}.txt").write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    all_ids = sorted(dict.fromkeys(splits["train"] + splits["val"] + splits["test"]))
    (split_root / "all.txt").write_text("\n".join(all_ids) + ("\n" if all_ids else ""), encoding="utf-8")


def make_motion_id(src_key: str, rel: Path) -> str:
    """
    Ensure unique ids across sources.
    HumanML3D numeric ids are already unique: keep stem.
    Others: include relpath to avoid collisions, plus source key.
    """
    if src_key == "humanml3d_30hz":
        return rel.stem
    stem = rel.with_suffix("").as_posix().replace("/", "__")
    return f"{src_key}__{stem}"


def segment_ranges(total_frames: int, *, window_frames: int, stride_frames: int, min_frames: int) -> List[Tuple[int, int]]:
    """
    Segment [0, total_frames) into windows [s, e) in joint-frame domain.
    Prefer sliding windows. Include tail window if long enough.
    """
    if total_frames < min_frames:
        return []
    # If it already fits in one window, do NOT create duplicate tail segments.
    if total_frames <= window_frames:
        return [(0, total_frames)]
    ranges: List[Tuple[int, int]] = []
    s = 0
    while s + window_frames <= total_frames:
        ranges.append((s, s + window_frames))
        s += stride_frames
    # tail: only add if there is a meaningful uncovered remainder
    if ranges:
        last_s, last_e = ranges[-1]
        remainder = total_frames - last_e
        # If remainder is tiny (e.g., 1-2 frames), skip tail to avoid near-duplicate windows.
        if remainder >= max(1, stride_frames // 4):
            tail_s = total_frames - window_frames
            tail_e = total_frames
            if tail_s >= 0 and (tail_e - tail_s) >= min_frames and (tail_s, tail_e) != ranges[-1]:
                ranges.append((tail_s, tail_e))
    return ranges


def _max_joint_frames_from_max_feat_len(max_feat_len: int) -> int:
    """
    mp.process_file returns feature length F = (segment_joint_frames - 1).
    Therefore, to enforce F <= max_feat_len we must have:
      segment_joint_frames <= max_feat_len + 1
    """
    return int(max_feat_len) + 1


def _read_first_nonempty_line(p: Path) -> str:
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if ln:
            return ln
    return ""


def bandai_caption_from_stem(stem: str) -> str:
    """
    Examples:
      dataset-2_wave-left-hand_youthful_013Re
      dataset-1_bow_active_001Re
    We want content + style, remove dataset id and trailing numeric id.
    """
    s = stem
    s = re.sub(r"Re$", "", s)
    m = re.match(r"dataset-(\d+)_(.+)_([^_]+)_(\d+)$", s, flags=re.IGNORECASE)
    if not m:
        # fallback: just prettify
        return re.sub(r"_+", " ", s.replace("-", " ")).strip()
    content = m.group(2).replace("-", " ").strip()
    style = m.group(3).replace("-", " ").strip()
    return f"{content} in a {style} manner"


def lafan_caption_from_stem(stem: str, *, theme_map: Dict[str, str]) -> str:
    """
    Examples: obstacles5_subject4Re, fallAndGetUp1_subject1Re, multipleActions1_subject3Re
    """
    s = re.sub(r"Re$", "", stem, flags=re.IGNORECASE)
    # Extract theme prefix (before first digit / _subject)
    m = re.match(r"([A-Za-z]+)(?:\\d+)?_subject\\d+", s)
    theme = m.group(1) if m else s.split("_")[0]
    # Normalize camelCase -> flat key
    theme_key = re.sub(r"[^a-zA-Z]", "", theme).lower()
    if theme_key not in theme_map:
        raise ValueError(f"No user-supplied LaFAN1 metadata for theme {theme_key!r}")
    return theme_map[theme_key]


def motorica_caption_from_stem(stem: str) -> str:
    """
    Motorica filenames are mostly metadata-heavy. We keep a best-effort short caption:
      "<prefix> motorica motion"
    where prefix is the first token before '_' (e.g., kthjazz/kthstreet/kthmisc).
    """
    s = re.sub(r"Re$", "", stem, flags=re.IGNORECASE)
    prefix = s.split("_", 1)[0].strip()
    prefix = prefix or "motorica"
    return f"{prefix} motorica motion"


def finedance_caption_from_stem(stem: str) -> str:
    """
    FineDance filenames are often numeric ids like '001'. Use a stable placeholder caption.
    """
    s = re.sub(r"Re$", "", stem, flags=re.IGNORECASE).strip()
    s = s or "unknown"
    return f"finedance dance sequence {s}"


def write_text_line(out_txt: Path, caption: str, tokens: str = "unk/OTHER", f_tag: float = 0.0, to_tag: float = 0.0) -> None:
    _ensure_dir(out_txt.parent)
    out_txt.write_text(f"{caption}#{tokens}#{f_tag:.1f}#{to_tag:.1f}\n", encoding="utf-8")


def write_text_for_item(
    *,
    src: SourceCfg,
    rel: Path,
    out_txt: Path,
    caption_fallback: str,
    tokens_fallback: str = "unk/OTHER",
    lafan_theme_map: Optional[Dict[str, str]] = None,
) -> None:
    if src.text_kind == "humanml3d_copy":
        assert src.humanml3d_text_root is not None
        p = src.humanml3d_text_root / f"{rel.stem}.txt"
        if p.is_file():
            out_txt.write_text(p.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            return
        # fallback
        write_text_line(out_txt, caption_fallback, tokens_fallback)
        return

    if src.text_kind == "motionx_copy":
        assert src.motionx_text_root is not None
        # motionx text mirrors the relative path under motionx root
        p = (src.motionx_text_root / rel).with_suffix(".txt")
        if p.is_file():
            caption = _read_first_nonempty_line(p) or caption_fallback
            write_text_line(out_txt, caption, "unk/OTHER", 0.0, 0.0)
            return
        write_text_line(out_txt, caption_fallback, tokens_fallback)
        return

    if src.text_kind == "bandai_name":
        write_text_line(out_txt, bandai_caption_from_stem(rel.stem), "unk/OTHER", 0.0, 0.0)
        return

    if src.text_kind == "lafan_theme":
        write_text_line(out_txt, lafan_caption_from_stem(rel.stem, theme_map=lafan_theme_map or {}), "unk/OTHER", 0.0, 0.0)
        return

    if src.text_kind == "motorica_name":
        write_text_line(out_txt, motorica_caption_from_stem(rel.stem), "unk/OTHER", 0.0, 0.0)
        return

    if src.text_kind == "finedance_name":
        write_text_line(out_txt, finedance_caption_from_stem(rel.stem), "unk/OTHER", 0.0, 0.0)
        return

    raise ValueError(f"Unknown text_kind={src.text_kind}")


def compute_mean_std(feature_paths: Sequence[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Streaming mean/std over all frames across all clips.
    """
    n = 0
    mean = None
    M2 = None
    for p in feature_paths:
        x = np.load(p)  # (F,263)
        x = x.astype(np.float64, copy=False)
        if mean is None:
            mean = np.zeros((x.shape[1],), dtype=np.float64)
            M2 = np.zeros((x.shape[1],), dtype=np.float64)
        # Welford over rows
        for i in range(x.shape[0]):
            n += 1
            delta = x[i] - mean
            mean += delta / n
            delta2 = x[i] - mean
            M2 += delta * delta2
    if mean is None or M2 is None or n < 2:
        raise RuntimeError("Not enough data to compute mean/std")
    var = M2 / (n - 1)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def iter_npz(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.npz"))


def parse_args() -> argparse.Namespace:
    repo = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_root",
        type=str,
        default=str(repo / "datasets" / "FineMotion" / "humanml3d_20hz_for_Tranning263" / "finemotion_263_v1"),
        help="Output dataset root (will create subfolders: new_joint_vecs, texts, splits, Mean.npy, Std.npy)",
    )
    ap.add_argument(
        "--include",
        type=str,
        nargs="+",
        default=["humanml3d_30hz", "bandai_30hz", "lafan1_30hz", "motionx_clean_30hz"],
        help="Which sources to include",
    )
    ap.add_argument("--device", type=str, default="cpu", help="cpu/cuda/cuda:0 or just GPU index like 0")
    ap.add_argument(
        "--smpl_model_root",
        type=str,
        default="/home/randong/mydata/MCM-LDM/deps/motionclip",
        help="Folder containing smpl/SMPL_NEUTRAL.pkl (smplx.create model_path)",
    )
    # segmentation in joint-space at 20Hz
    ap.add_argument(
        "--max_feat_len",
        type=int,
        default=196,
        help="Max HumanML3D feature length F. Default 196 to match MCM-LDM HumanML3D config.",
    )
    ap.add_argument(
        "--window_frames",
        type=int,
        default=197,
        help=(
            "Segment length in 20Hz joint frames. Feature length will be (window_frames-1). "
            "To match MCM-LDM MAX_LEN=196 feats, the safe default is 197 joint frames -> 196 feats."
        ),
    )
    ap.add_argument("--stride_frames", type=int, default=80, help="Stride in 20Hz joint frames for long sequences")
    ap.add_argument("--min_frames", type=int, default=41, help="Min segment length in 20Hz joint frames (41 -> 40 feats)")
    ap.add_argument("--seed", type=int, default=1234, help="Split seed")
    ap.add_argument("--lafan-theme-map", type=Path,
                    help="Required when including LaFAN1: user-local JSON {theme_key: source description}.")
    ap.add_argument("--limit_per_source", type=int, default=-1, help="Debug: limit number of input npz per source")
    ap.add_argument(
        "--axis_fix_motionx",
        action="store_true",
        help="Apply auto X/Y/Z-up -> Y-up fix for motionx_clean_30hz (recommended).",
    )
    ap.add_argument(
        "--humanml3d_text_root",
        type=str,
        default="/home/randong/mydata/MCM-LDM/HumanML3D-axisAngle-all/HumanML3D/texts",
        help="HumanML3D official texts folder (caption#tokens#...).",
    )
    ap.add_argument(
        "--motionx_text_root",
        type=str,
        default=str(repo / "datasets" / "raw" / "Motion-X" / "motionx_seq_text_v1.1"),
        help="Motion-X sequence texts root (plain sentence).",
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
    # allow "cuda" (no index) and "cuda:0" etc.
    return d


def main() -> int:
    args = parse_args()
    repo = _repo_root()
    try:
        lafan_theme_map, lafan_metadata_sha256 = _preflight_lafan_metadata(
            args.include, repo, args.lafan_theme_map, int(args.limit_per_source))
    except (ValueError, OSError, UnicodeError) as exc:
        raise SystemExit(f"LaFAN1 metadata preflight failed: {exc}")
    args.device = _normalize_device(args.device)

    # Guard: user may request CUDA in a CPU-only torch build (e.g. lafan_smplx_cpu).
    # In that case, fall back to CPU with a clear warning (correctness is unchanged; only speed).
    if str(args.device).startswith("cuda"):
        try:
            import torch  # noqa: WPS433

            if not torch.cuda.is_available():
                print(
                    f"[WARN] Requested --device={args.device}, but torch.cuda.is_available() is False "
                    f"(torch.version.cuda={getattr(torch.version, 'cuda', None)}). Falling back to CPU. "
                    "If you want GPU acceleration, run this script in an environment with CUDA-enabled PyTorch."
                )
                args.device = "cpu"
        except Exception as e:
            print(f"[WARN] Could not validate CUDA availability ({type(e).__name__}: {e}). Using device={args.device}.")

    out_root = Path(args.out_root)
    vec_dir = out_root / "new_joint_vecs"
    txt_dir = out_root / "texts"
    split_root = out_root / "splits"
    _ensure_dir(vec_dir)
    _ensure_dir(txt_dir)
    _ensure_dir(split_root)

    # motion_process initialization (HumanML3D 263 feature extraction)
    mp = init_motion_process(_conv_resolve_paths())

    # Build source configs
    sources_all: Dict[str, SourceCfg] = {
        "humanml3d_30hz": SourceCfg(
            key="humanml3d_30hz",
            npz_root=repo / "datasets" / "FineMotion" / "humanml3d_30hz",
            text_kind="humanml3d_copy",
            humanml3d_text_root=Path(args.humanml3d_text_root),
        ),
        "bandai_30hz": SourceCfg(
            key="bandai_30hz",
            npz_root=repo / "datasets" / "FineMotion" / "bandai_30hz",
            text_kind="bandai_name",
        ),
        "lafan1_30hz": SourceCfg(
            key="lafan1_30hz",
            npz_root=repo / "datasets" / "FineMotion" / "lafan1_30hz",
            text_kind="lafan_theme",
        ),
        "motionx_clean_30hz": SourceCfg(
            key="motionx_clean_30hz",
            npz_root=repo / "datasets" / "FineMotion" / "motionx_clean_30hz",
            text_kind="motionx_copy",
            motionx_text_root=Path(args.motionx_text_root),
        ),
        "motorica_30hz": SourceCfg(
            key="motorica_30hz",
            npz_root=repo / "datasets" / "FineMotion" / "motorica_30hz",
            text_kind="motorica_name",
        ),
        "finedance_30hz": SourceCfg(
            key="finedance_30hz",
            npz_root=repo / "datasets" / "FineMotion" / "finedance_30hz",
            text_kind="finedance_name",
        ),
    }

    include = list(dict.fromkeys(args.include))
    for k in include:
        if k not in sources_all:
            raise SystemExit(f"Unknown source: {k}. Supported: {sorted(sources_all.keys())}")
        if not sources_all[k].npz_root.is_dir():
            raise SystemExit(f"Missing source root: {sources_all[k].npz_root}")

    # Collect ids and build
    built_ids: List[str] = []
    built_feat_paths: List[Path] = []
    stats: Dict[str, Dict[str, int]] = {}

    max_feat_len = int(args.max_feat_len)
    max_joint_frames = _max_joint_frames_from_max_feat_len(max_feat_len)
    window_frames = int(args.window_frames)
    if window_frames > max_joint_frames:
        print(
            f"[WARN] --window_frames={window_frames} too large for --max_feat_len={max_feat_len}. "
            f"Clamping to {max_joint_frames} (so feature_len <= {max_feat_len})."
        )
        window_frames = max_joint_frames

    for key in include:
        src = sources_all[key]
        files = list(iter_npz(src.npz_root))
        if args.limit_per_source and int(args.limit_per_source) > 0:
            files = files[: int(args.limit_per_source)]
        stats[key] = {"n_in": len(files), "n_out": 0, "n_seg": 0, "n_skip_short": 0, "n_skip_badlen": 0}

        for f in files:
            rel = f.relative_to(src.npz_root)
            base_id = make_motion_id(src.key, rel)

            # SMPL forward -> joints22 at source fps, then resample to 20Hz joints
            joints22, src_fps = smpl_npz_to_joints22(
                f,
                smpl_model_root=Path(args.smpl_model_root),
                device=str(args.device),
            )
            joints20 = linear_resample_positions(joints22, src_fps=float(src_fps), tgt_fps=20.0)

            # optional axis fix (mainly for Motion-X variants)
            if src.key.startswith("motionx") and bool(args.axis_fix_motionx):
                up_axis = _detect_up_axis_from_joints(joints20)
                joints20 = _rotate_joints_to_y_up(joints20, up_axis=up_axis)

            segs = segment_ranges(
                joints20.shape[0],
                window_frames=window_frames,
                stride_frames=int(args.stride_frames),
                min_frames=int(args.min_frames),
            )
            if not segs:
                stats[key]["n_skip_short"] += 1
                continue

            for si, (s, e) in enumerate(segs):
                seg_id = base_id if (len(segs) == 1 and s == 0 and e == joints20.shape[0]) else f"{base_id}__s{s:06d}_e{e:06d}"
                out_feat = vec_dir / f"{seg_id}.npy"
                out_txt = txt_dir / f"{seg_id}.txt"

                # Extract 263 features
                feat, _, _, _ = mp.process_file(joints20[s:e], 0.002)
                feat = feat.astype(np.float32)

                # Enforce length constraints: F <= max_feat_len and F >= 40
                # process_file returns F = (segment_len - 1)
                if feat.shape[0] < 40 or feat.shape[0] > max_feat_len:
                    stats[key]["n_skip_badlen"] += 1
                    continue

                np.save(out_feat, feat)
                built_feat_paths.append(out_feat)
                built_ids.append(seg_id)
                stats[key]["n_out"] += 1
                if len(segs) > 1:
                    stats[key]["n_seg"] += 1

                # Write texts
                # HumanML3D: if we didn't segment and id == stem, copy full file.
                # Otherwise, best-effort: first line tokens.
                if src.text_kind == "humanml3d_copy":
                    p = (src.humanml3d_text_root or Path(args.humanml3d_text_root)) / f"{rel.stem}.txt"
                    if p.is_file() and seg_id == rel.stem:
                        out_txt.write_text(p.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
                    elif p.is_file():
                        first = _read_first_nonempty_line(p)
                        if "#" in first:
                            parts = first.split("#")
                            caption = parts[0].strip()
                            tokens = parts[1].strip() if len(parts) > 1 else "unk/OTHER"
                            write_text_line(out_txt, caption, tokens, 0.0, 0.0)
                        else:
                            write_text_line(out_txt, first or rel.stem)
                    else:
                        write_text_line(out_txt, rel.stem)
                else:
                    write_text_for_item(
                        src=src,
                        rel=rel,
                        out_txt=out_txt,
                        caption_fallback=rel.stem,
                        tokens_fallback="unk/OTHER",
                        lafan_theme_map=lafan_theme_map,
                    )

    # Write splits (ids refer to new_joint_vecs/<id>.npy and texts/<id>.txt)
    all_ids = sorted(dict.fromkeys(built_ids))
    splits = split_ids(all_ids, seed=int(args.seed))
    write_split_files(split_root, splits)

    # Compute stats and save Mean/Std (required by mcm-ldm loader)
    mean, std = compute_mean_std(built_feat_paths)
    np.save(out_root / "Mean.npy", mean)
    np.save(out_root / "Std.npy", std)

    meta = {
        "out_root": str(out_root),
        "include": include,
        "lafan_theme_metadata": ({
            "input_sha256": lafan_metadata_sha256,
            "role": "user-acquired upstream metadata; not bundled or relicensed",
        } if lafan_metadata_sha256 else None),
        "device": str(args.device),
        "smpl_model_root": str(args.smpl_model_root),
        "segmentation": {
            "target_fps": 20.0,
            "window_frames": int(window_frames),
            "stride_frames": int(args.stride_frames),
            "min_frames": int(args.min_frames),
            "max_feat_len": int(max_feat_len),
            "feature_len_range": [40, int(max_feat_len)],
        },
        "axis_fix_motionx": bool(args.axis_fix_motionx),
        "stats": stats,
        "n_total_out": len(all_ids),
        "splits": {k: len(v) for k, v in splits.items()},
        "paths": {
            "new_joint_vecs": str(vec_dir),
            "texts": str(txt_dir),
            "split_root": str(split_root),
        },
    }
    (out_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("[OK] Built dataset:", out_root)
    print("[OK] Total clips:", len(all_ids))
    print("[OK] Mean/Std:", out_root / "Mean.npy", out_root / "Std.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

