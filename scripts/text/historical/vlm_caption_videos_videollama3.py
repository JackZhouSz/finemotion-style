#!/usr/bin/env python3
"""
Batch caption MP4 videos using VideoLLaMA3 (Transformers) and write JSONL.

Typical usage (GPU env, e.g. mcmldm):
  conda run -n mcmldm python tools/vlm_caption_videos_videollama3.py \\
    --model_path /home/randong/mydata/models/VideoLLaMA3-7B \\
    --videos_dir datasets/VLM_text_label/smoketest_mesh_batch_v3 \\
    --out_jsonl datasets/VLM_text_label/captions/smoketest_mesh_batch_v3.videollama3.jsonl \\
    --video_fps 4 --max_frames 128 --max_new_tokens 128 --temperature 0.2

Notes:
  - This script is designed for long runs: it appends per-video results to JSONL.
  - Use --shard_idx/--shard_count to parallelize across multiple GPUs/processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import gc
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Mitigate CUDA allocator fragmentation for long runs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


DEFAULT_SYSTEM = "You are a helpful assistant."
DEFAULT_QUESTION = (
    "You are given a short video clip of a single person performing a motion.\n"
    "Describe ONLY the human motion in English for text-to-motion generation / motion style transfer.\n"
    "Do NOT mention the video itself, the renderer, a 3D model/mesh, silhouette, background, camera, lighting, or missing facial features.\n"
    "Style constraints:\n"
    "- Use simple words and short sentences.\n"
    "- Prefer starting with \"The person ...\".\n"
    "- Each line MUST be a single sentence with at most 20 words.\n"
    "Return exactly two lines:\n"
    "MAIN: <one concise sentence describing the main action and overall style (fast/slow, smooth/shaky, strong/weak, balanced/unsteady)>\n"
    "DETAILS: <one concise sentence focusing on subtle/high-frequency/secondary motions (e.g., jitter, tremor, impacts, balance corrections), and contact events>\n"
)

SEG_SUFFIX_RE = re.compile(r"__s\d{6}_e\d{6}$")


@dataclass
class GenConfig:
    max_new_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iter_videos(videos_dir: Path, pattern: str) -> List[Path]:
    vids = sorted([p for p in videos_dir.glob(pattern) if p.is_file()])
    return vids


def _read_done_ids(out_jsonl: Path) -> set:
    done = set()
    if not out_jsonl.is_file():
        return done
    with out_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                mid = obj.get("id")
                if mid:
                    done.add(str(mid))
            except Exception:
                continue
    return done


def _apply_shard(items: List[Path], shard_idx: int, shard_count: int) -> List[Path]:
    shard_count = int(shard_count)
    shard_idx = int(shard_idx)
    if shard_count <= 1:
        return items
    if shard_idx < 0 or shard_idx >= shard_count:
        raise ValueError(f"Bad shard params: shard_idx={shard_idx} shard_count={shard_count}")
    return [p for i, p in enumerate(items) if (i % shard_count) == shard_idx]


_MAIN_RE = re.compile(r"^\s*MAIN\s*:\s*(.+?)\s*$", re.IGNORECASE)
_DETAILS_RE = re.compile(r"^\s*DETAILS\s*:\s*(.+?)\s*$", re.IGNORECASE)


def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def _base_id(mid: str) -> str:
    m = SEG_SUFFIX_RE.search(mid or "")
    return (mid[: m.start()] if m else (mid or "")).strip()


def _load_context_jsonl(path: str) -> Dict[str, Dict]:
    """
    Load per-video context JSONL into a dict keyed by `id`.
    Expected keys (best-effort):
      - id
      - main_anchor
      - prompt_context
      - required_keywords
      - prompt_context_sha1 (optional)
    """
    p = Path(path) if path else None
    if not p or (not p.is_file()):
        return {}
    out: Dict[str, Dict] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            mid = obj.get("id")
            if not mid:
                continue
            out[str(mid)] = obj
    return out


def _norm_words(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return [w for w in s.split() if w]


def _match_required_keywords(text: str, required: Iterable[str]) -> bool:
    """
    Best-effort keyword consistency check between VLM MAIN and official labels.
    We allow small morphological variants via `startswith`.
    """
    t_words = _norm_words(text)
    t_compact = "".join(t_words)
    if not t_words:
        return False
    for kw in required or []:
        kw = str(kw or "").strip()
        if not kw:
            continue
        kw_words = _norm_words(kw)
        if not kw_words:
            continue
        kw_compact = "".join(kw_words)
        if kw_compact and (kw_compact in t_compact):
            continue
        for tok in kw_words:
            if not any(w.startswith(tok) for w in t_words):
                return False
    return True


def _build_question(base_question: str, *, prompt_context: str, main_anchor: str, main_mode: str) -> str:
    q = (base_question or "").rstrip()
    pc = (prompt_context or "").strip()
    ma = (main_anchor or "").strip()
    mm = (main_mode or "").strip().lower()
    if pc:
        q += "\n\n" + pc
    if ma:
        q += "\n\nMAIN_ANCHOR: " + ma
        if mm == "anchor_only":
            q += (
                "\nIMPORTANT: MAIN is fixed.\n"
                "- You MUST copy MAIN_ANCHOR EXACTLY as the MAIN line.\n"
                "- Do NOT rewrite MAIN.\n"
                "- DETAILS must be consistent with MAIN_ANCHOR.\n"
            )
        elif mm == "vlm_or_anchor":
            q += "\nIMPORTANT: MAIN must be consistent with MAIN_ANCHOR (official labels)."
    return q.strip() + "\n"


def _strip_role_prefixes(s: str) -> str:
    """Remove leading MAIN:/DETAILS: prefixes (sometimes repeated)."""
    s = (s or "").strip()
    for _ in range(3):
        s2 = re.sub(r"^\s*(MAIN|DETAILS)\s*:\s*", "", s, flags=re.IGNORECASE).strip()
        if s2 == s:
            break
        s = s2
    return s.strip()


def _first_sentence(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", s)
    return (parts[0] if parts else s).strip()


def _parse_two_lines(text: str) -> Tuple[str, str]:
    s = (text or "").strip()
    if not s:
        return "", ""

    main = ""
    details = ""

    # 1) Preferred: two labeled lines
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _MAIN_RE.match(line)
        if m:
            main = m.group(1).strip()
            continue
        m = _DETAILS_RE.match(line)
        if m:
            details = m.group(1).strip()
            continue

    # 2) If MAIN captured but DETAILS is embedded on the same line, split it.
    if main and (not details) and re.search(r"\bDETAILS\s*:", main, flags=re.IGNORECASE):
        parts = re.split(r"\bDETAILS\s*:\s*", main, maxsplit=1, flags=re.IGNORECASE)
        main = (parts[0] if parts else "").strip()
        details = (parts[1] if len(parts) > 1 else "").strip()

    # 3) If the model omitted MAIN:/DETAILS: formatting but used inline "DETAILS:", split by it.
    if not main and not details and re.search(r"\bDETAILS\s*:", s, flags=re.IGNORECASE):
        parts = re.split(r"\bDETAILS\s*:\s*", s, maxsplit=1, flags=re.IGNORECASE)
        main = (parts[0] if parts else "").strip()
        details = (parts[1] if len(parts) > 1 else "").strip()

    # 4) Fallback: first sentence as main, rest as details.
    if not main and not details:
        parts = re.split(r"(?<=[.!?])\s+", s)
        main = (parts[0] if parts else s).strip()
        details = " ".join([p.strip() for p in parts[1:] if p.strip()]).strip()

    main = _strip_role_prefixes(main)
    details = _strip_role_prefixes(details)

    # If main contains multiple sentences and details is empty, split them.
    if main and not details:
        parts = re.split(r"(?<=[.!?])\s+", main)
        if len(parts) > 1:
            main = parts[0].strip()
            details = " ".join([p.strip() for p in parts[1:] if p.strip()]).strip()
            details = _strip_role_prefixes(details)

    # Keep one sentence per field (prompt requires it).
    main = _first_sentence(main)
    details = _first_sentence(details)
    return main, details


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _parse_scale_to(s: str) -> Optional[Tuple[int, int]]:
    s = (s or "").strip().lower()
    if not s:
        return None
    m = re.match(r"^(?P<w>\d+)x(?P<h>\d+)$", s)
    if not m:
        raise ValueError(f"Bad --scale_to format: {s} (expected like 512x384)")
    w = int(m.group("w"))
    h = int(m.group("h"))
    if w <= 0 or h <= 0:
        raise ValueError(f"Bad --scale_to: {s}")
    return w, h


def _maybe_scale_video(vp: Path, scale_to: Optional[Tuple[int, int]], cache_dir: Optional[Path]) -> Tuple[Path, Optional[Dict]]:
    """
    Returns (video_path_to_use, scale_meta or None).
    We use ffmpeg to downscale high-res videos to prevent VLM OOM on long clips.
    """
    if scale_to is None:
        return vp, None
    if cache_dir is None:
        raise ValueError("scale_to set but cache_dir is None")
    cache_dir.mkdir(parents=True, exist_ok=True)

    w, h = int(scale_to[0]), int(scale_to[1])
    out = cache_dir / vp.name
    if out.is_file() and out.stat().st_mtime >= vp.stat().st_mtime:
        return out, {"scaled": True, "src": str(vp), "dst": str(out), "to": f"{w}x{h}", "cached": True}

    vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=white"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(vp),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "veryfast",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        raise RuntimeError(f"ffmpeg scale failed for {vp} -> {out}: {e}")
    return out, {"scaled": True, "src": str(vp), "dst": str(out), "to": f"{w}x{h}", "cached": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--videos_dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="*.mp4")
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--system", type=str, default=DEFAULT_SYSTEM)
    ap.add_argument("--question", type=str, default=DEFAULT_QUESTION)
    ap.add_argument(
        "--context_jsonl",
        type=str,
        default="",
        help="Optional per-video context JSONL (id -> prompt_context/main_anchor/required_keywords).",
    )
    ap.add_argument(
        "--main_mode",
        type=str,
        default="vlm",
        choices=["vlm", "anchor_only", "vlm_or_anchor"],
        help="How to set caption_main when context provides main_anchor.",
    )
    ap.add_argument("--video_fps", type=float, default=4.0, help="Sampling fps for VLM video encoder.")
    ap.add_argument("--max_frames", type=int, default=128)
    ap.add_argument(
        "--scale_to",
        type=str,
        default="",
        help="Optional downscale videos before captioning, e.g. 512x384 (recommended for long 640x540 clips).",
    )
    ap.add_argument(
        "--scale_cache_dir",
        type=str,
        default="",
        help="Where to store scaled videos (default: <videos_dir>/_scaled_<WxH>/).",
    )

    ap.add_argument("--attn_impl", type=str, default="sdpa", help="e.g. sdpa / flash_attention_2 (if installed).")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument(
        "--device_map",
        type=str,
        default="",
        help="HuggingFace device_map (e.g. auto). If set, may require accelerate.",
    )
    ap.add_argument("--device", type=str, default="", help="Force device for single-device load (e.g. cuda / cpu).")

    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--no_sample", dest="do_sample", action="store_false")
    ap.set_defaults(do_sample=True)

    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--shard_count", type=int, default=1)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    model_path = Path(args.model_path)
    videos_dir = Path(args.videos_dir)
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    videos = _iter_videos(videos_dir, args.pattern)
    videos = _apply_shard(videos, int(args.shard_idx), int(args.shard_count))
    if int(args.limit) > 0:
        videos = videos[: int(args.limit)]
    if not videos:
        raise SystemExit("No videos found.")

    done = _read_done_ids(out_jsonl) if bool(args.skip_existing) else set()
    scale_to = _parse_scale_to(args.scale_to)
    scale_cache_dir: Optional[Path]
    if scale_to is None:
        scale_cache_dir = None
    else:
        if args.scale_cache_dir:
            scale_cache_dir = Path(args.scale_cache_dir)
        else:
            scale_cache_dir = videos_dir / f"_scaled_{scale_to[0]}x{scale_to[1]}"


    ctx_map = _load_context_jsonl(str(args.context_jsonl)) if str(args.context_jsonl).strip() else {}
    if ctx_map:
        print(f"[ctx] loaded: {len(ctx_map)} from {args.context_jsonl}")

    # dtype
    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Decide device for the common single-GPU case.
    if args.device:
        device = torch.device(str(args.device))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[load] model={model_path} dtype={args.dtype} attn={args.attn_impl} device_map={args.device_map or '(none)'} device={device}")
    load_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        attn_implementation=str(args.attn_impl),
    )
    if args.device_map:
        load_kwargs["device_map"] = str(args.device_map)
    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
    except ValueError as e:
        # Common in minimal envs: `device_map` requires accelerate.
        if args.device_map and "requires `accelerate`" in str(e):
            print("[warn] accelerate missing; falling back to single-device load (no device_map).")
            load_kwargs.pop("device_map", None)
            model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
        else:
            raise

    # If not using device_map, move whole model to one device.
    use_device_map = bool(getattr(model, "hf_device_map", None))
    if not use_device_map:
        model = model.to(device)
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)

    gen_cfg = GenConfig(
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        do_sample=bool(args.do_sample),
    )

    torch.set_grad_enabled(False)
    model.eval()

    with out_jsonl.open("a", encoding="utf-8") as fw:
        for i, vp in enumerate(videos):
            mid = vp.stem
            if mid in done:
                continue

            inputs = None
            output_ids = None

            try:
                ctx = ctx_map.get(mid) or ctx_map.get(_base_id(mid))
                ctx_id = str(ctx.get("id")) if isinstance(ctx, dict) and ctx.get("id") else ""
                prompt_context = str(ctx.get("prompt_context", "")) if isinstance(ctx, dict) else ""
                main_anchor = str(ctx.get("main_anchor", "")) if isinstance(ctx, dict) else ""
                required_keywords = ctx.get("required_keywords") if isinstance(ctx, dict) else []
                if not isinstance(required_keywords, list):
                    required_keywords = []

                question_final = _build_question(
                    str(args.question),
                    prompt_context=prompt_context,
                    main_anchor=main_anchor,
                    main_mode=str(args.main_mode),
                )

                vp_use, scale_meta = _maybe_scale_video(vp, scale_to=scale_to, cache_dir=scale_cache_dir)
                conversation = [
                    {"role": "system", "content": str(args.system)},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": {
                                    "video_path": str(vp_use.resolve()),
                                    "fps": float(args.video_fps),
                                    "max_frames": int(args.max_frames),
                                },
                            },
                            {"type": "text", "text": question_final},
                        ],
                    },
                ]
                # Match official inference defaults for better formatting.
                inputs = processor(
                    conversation=conversation,
                    add_system_prompt=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                # Move tensors to the chosen device (works for single-device and most sharded setups).
                inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
                if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch_dtype)

                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=int(gen_cfg.max_new_tokens),
                    temperature=float(gen_cfg.temperature),
                    top_p=float(gen_cfg.top_p),
                    do_sample=bool(gen_cfg.do_sample),
                )
                response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                cap_main_vlm, cap_details_vlm = _parse_two_lines(response)

                cap_main = cap_main_vlm
                cap_details = cap_details_vlm
                main_mode_used = str(args.main_mode).strip().lower()
                main_was_overridden = False
                if main_anchor and main_mode_used == "anchor_only":
                    cap_main = main_anchor
                    main_was_overridden = (cap_main_vlm.strip() != main_anchor.strip())
                elif main_anchor and main_mode_used == "vlm_or_anchor":
                    if (not cap_main_vlm) or (required_keywords and (not _match_required_keywords(cap_main_vlm, required_keywords))):
                        cap_main = main_anchor
                        main_was_overridden = True
                else:
                    main_mode_used = "vlm"

                rec = {
                    "time_utc": _utc_now(),
                    "id": mid,
                    "video_path": str(vp_use),
                    "model_path": str(model_path),
                    "video_fps": float(args.video_fps),
                    "video_max_frames": int(args.max_frames),
                    "question": str(args.question),
                    "question_final_sha1": _sha1(question_final),
                    "gen": asdict(gen_cfg),
                    "response": response,
                    "caption_main_vlm": cap_main_vlm,
                    "caption_details_vlm": cap_details_vlm,
                    "caption_main": cap_main,
                    "caption_details": cap_details,
                    "labelaware": {
                        "context_jsonl": str(args.context_jsonl) if str(args.context_jsonl).strip() else "",
                        "context_id": ctx_id,
                        "has_context": bool(ctx is not None),
                        "prompt_context_sha1": str(ctx.get("prompt_context_sha1", "")) if isinstance(ctx, dict) else "",
                        "main_anchor": main_anchor,
                        "required_keywords": required_keywords,
                        "main_mode": main_mode_used,
                        "main_was_overridden": bool(main_was_overridden),
                    },
                }
                if scale_meta is not None:
                    rec["scale"] = scale_meta
            except Exception as e:
                rec = {
                    "time_utc": _utc_now(),
                    "id": mid,
                    "video_path": str(vp),
                    "model_path": str(model_path),
                    "error": f"{type(e).__name__}: {e}",
                }
            finally:
                # Best-effort cleanup between videos to avoid fragmentation / gradual OOM.
                try:
                    del output_ids
                    del inputs
                except Exception:
                    pass
                _cleanup_cuda()

            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fw.flush()
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(videos)}] done (last={mid})")

    print(f"[OK] wrote: {out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

