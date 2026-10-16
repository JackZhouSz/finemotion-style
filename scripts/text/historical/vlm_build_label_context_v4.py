#!/usr/bin/env python3
from __future__ import annotations

"""
Build per-video label context JSONL for label-aware VLM captioning (v4 dataset).

Public adaptation of the historical project script: the embedded LaFAN1 README
theme-description table is removed. LaFAN1 processing requires --lafan-theme-map
pointing to metadata acquired locally by the user under its upstream terms.
That metadata and generated anchors retain source provenance; they are not
relicensed as project-authored MIT prose. The original private snapshot is kept
separately, with its hash recorded in manifest.json.

This script does NOT run any VLM model. It only extracts/normalizes *official* label
information (from filenames / dataset-provided README/JSON) into:

  - main_anchor: a clean HumanML-like MAIN sentence (no noisy codes)
  - prompt_context: richer context to be appended to the VLM prompt
  - required_keywords: tokens that must appear in MAIN (for audit / main override rules)

Supported sources (by id prefix):
  - bandai_30hz__*
  - lafan1_30hz__*
  - motorica_30hz__*
  - finedance_30hz__*

Typical usage:
  python tools/vlm_build_label_context_v4.py \\
    --videos_dir datasets/VLM_text_label/v4_full_20260212_133514/mesh_videos_v4 \\
    --out_jsonl  datasets/VLM_text_label/v5_labelaware/label_context.jsonl
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SEG_SUFFIX_RE = re.compile(r"__s\d{6}_e\d{6}$")


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _safe(s: str) -> str:
    s = (s or "").replace("\t", " ").replace("\r", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _safe_multiline(s: str) -> str:
    s = (s or "").replace("\t", " ").replace("\r", "")
    lines: List[str] = []
    for ln in s.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        ln = re.sub(r"\s+", " ", ln)
        lines.append(ln)
    return "\n".join(lines).strip()


def _base_id(mid: str) -> str:
    m = SEG_SUFFIX_RE.search(mid)
    return mid[: m.start()] if m else mid


def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def _a_or_an(s: str) -> str:
    s = (s or "").strip().lower()
    if not s:
        return "a"
    return "an" if s[0] in "aeiou" else "a"


# -----------------------------------------------------------------------------
# Bandai adapter
# -----------------------------------------------------------------------------


def _bandai_parse_stem(stem: str) -> Optional[Dict[str, str]]:
    """
    Examples (v4 ids typically end with Re):
      dataset-1_bow_angry_001Re
      dataset-2_wave-right-hand_youthful_029Re
      dataset-1_walk-back_not-confident_001Re
    """
    s = re.sub(r"Re$", "", stem, flags=re.IGNORECASE)
    parts = s.split("_")
    if len(parts) < 4 or not parts[0].lower().startswith("dataset-"):
        return None
    dataset = parts[0]
    take = parts[-1]
    style = parts[-2]
    content = "_".join(parts[1:-2])  # may contain underscores
    return {"dataset": dataset, "content": content, "style": style, "take": take}


_BANDAI_ACTION_GLOSSARY = {
    "bow": "bow = bending the upper body forward.",
    "wave": "wave = moving the hand/arm side to side.",
    "kick": "kick = striking with the foot/leg.",
    "punch": "punch = striking with the fist/arm.",
}

_BANDAI_STYLE_GLOSSARY = {
    "angry": "angry style = forceful, tense, sharp movements.",
    "active": "active style = energetic and lively movements.",
    "sad": "sad style = slower, lower energy movements.",
    "proud": "proud style = upright posture, confident movements.",
    "childish": "childish style = playful, bouncy movements.",
    "not confident": "not-confident style = hesitant, cautious movements.",
}


def _bandai_label_context(mid: str) -> Optional[Dict]:
    base = _base_id(mid)
    _, rest = base.split("__", 1)
    info = _bandai_parse_stem(rest)
    if info is None:
        return None
    content_raw = info["content"]
    style_raw = info["style"]
    content_phrase = _safe(content_raw.replace("-", " ").replace("_", " "))
    style_phrase = _safe(style_raw.replace("-", " ").replace("_", " "))
    # Clean MAIN anchor (keep short and label-faithful)
    main_anchor = _safe(f"The person performs {content_phrase} in {_a_or_an(style_phrase)} {style_phrase} manner.")
    required = [w for w in [content_phrase.split(" ")[0], style_phrase.split(" ")[0]] if w]

    # Prompt context (can be richer; not written into final training text)
    gloss = []
    if content_phrase:
        k = content_phrase.split(" ")[0].lower()
        if k in _BANDAI_ACTION_GLOSSARY:
            gloss.append(_BANDAI_ACTION_GLOSSARY[k])
    if style_phrase:
        k = style_phrase.lower()
        if k in _BANDAI_STYLE_GLOSSARY:
            gloss.append(_BANDAI_STYLE_GLOSSARY[k])
    gloss_txt = ("\n".join([f"- {g}" for g in gloss]) if gloss else "").strip()
    prompt_context = _safe_multiline(
        "\n".join(
            [
                "OFFICIAL LABELS (Bandai; derived from filename):",
                f"- action/content: {content_phrase}",
                f"- style: {style_phrase}",
                "Treat these labels as ground truth.",
                (("Glossary:\n" + gloss_txt) if gloss_txt else ""),
                f"Full id: {mid}",
            ]
        )
    )
    return {
        "id": mid,
        "base_id": base,
        "src_key": "bandai_30hz",
        "main_anchor": main_anchor,
        "required_keywords": required,
        "prompt_context": prompt_context,
        "meta": {"dataset": info["dataset"], "take": info["take"], "content_raw": content_raw, "style_raw": style_raw},
    }


# -----------------------------------------------------------------------------
# LaFAN1 adapter
# -----------------------------------------------------------------------------


def _load_lafan_theme_map(path: Path) -> Dict[str, str]:
    """Read a user-local {normalized_theme_key: source_description} JSON map.

    Source descriptions are deliberately not embedded or downloaded by this
    tool. Obtain the dataset metadata separately under the upstream terms.
    """
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or not value:
        raise ValueError("LaFAN1 theme map must be a nonempty JSON object")
    themes: Dict[str, str] = {}
    for key, description in value.items():
        if not isinstance(key, str) or re.fullmatch(r"[a-z]+", key) is None:
            raise ValueError("LaFAN1 theme keys must be lowercase letters, e.g. a filename theme")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"LaFAN1 theme {key!r} needs a nonempty source description")
        if any(character in description for character in "#\r\n"):
            raise ValueError(f"LaFAN1 theme {key!r} contains a separator or line break")
        themes[key] = description
    return themes

_LAFAN_REQUIRED = {
    "obstacles": ["uneven", "terrain"],
    "walk": ["walk"],
    "dance": ["dance"],
    "fallandgetup": ["fall", "get", "up"],
    "aiming": ["aim", "gun"],
    "ground": ["crawl", "crouch"],
    "multipleactions": ["multiple", "misc"],
    "run": ["run", "jog"],
    "fight": ["fight"],
    "jumps": ["jump", "hop", "leap"],
    "fightandsports": ["fight", "sport"],
    "pushandstumble": ["push", "stumble", "recover"],
    "pushandfall": ["push", "fall", "get", "up"],
    "sprint": ["sprint"],
    "push": ["push"],
}


def _lafan_theme_of_stem(stem: str, *, theme_map: Dict[str, str]) -> Tuple[str, str]:
    s = re.sub(r"Re$", "", stem, flags=re.IGNORECASE)
    m = re.match(r"([A-Za-z]+)(?:\d+)?_subject\d+", s)
    theme = m.group(1) if m else s.split("_")[0]
    theme_key = re.sub(r"[^a-zA-Z]", "", theme).lower()
    if theme_key not in theme_map:
        raise ValueError(f"No user-supplied LaFAN1 metadata for theme {theme_key!r}")
    desc = theme_map[theme_key]
    return theme_key, desc


def _lafan_label_context(mid: str, *, theme_map: Dict[str, str]) -> Optional[Dict]:
    base = _base_id(mid)
    _, rest = base.split("__", 1)
    theme_key, desc = _lafan_theme_of_stem(rest, theme_map=theme_map)
    desc_clean = _safe(desc)
    # Retain the historical construction, now using explicit user-local source
    # metadata. Its wording/provenance is not newly authored by this script.
    main_anchor = _safe(f"The person performs {desc_clean.lower()}.")
    required = list(_LAFAN_REQUIRED.get(theme_key, []))
    prompt_context = _safe_multiline(
        "\n".join(
            [
                "OFFICIAL LABELS (LaFAN1 theme):",
                f"- theme: {theme_key}",
                f"- description: {desc_clean}",
                "Treat this theme description as ground truth.",
                f"Full id: {mid}",
            ]
        )
    )
    return {
        "id": mid,
        "base_id": base,
        "src_key": "lafan1_30hz",
        "main_anchor": main_anchor,
        "required_keywords": required,
        "prompt_context": prompt_context,
        "meta": {"theme": theme_key, "description": desc_clean},
    }


# -----------------------------------------------------------------------------
# Motorica adapter
# -----------------------------------------------------------------------------


_MOTORICA_CODE_TO_STYLE = {
    "gLH": "Hiphop",
    "gKR": "Krumping",
    "gPO": "Popping",
    "gLO": "Locking",
    "gJZ": "Jazz",
    "gCH": "Charleston",
    "gTP": "Tapping",
    "gCA": "Casual",
}


def _motorica_label_context(mid: str) -> Optional[Dict]:
    base = _base_id(mid)
    _, rest = base.split("__", 1)
    # Example: kthstreet_gPO_sFM_cAll_d02_mPO_ch01_justus2_002Re
    parts = rest.split("_")
    if len(parts) < 2:
        return None
    prefix = parts[0]
    code = parts[1]
    style = _MOTORICA_CODE_TO_STYLE.get(code, code)
    style_clean = _safe(style).lower()
    main_anchor = _safe(f"The person performs a {style_clean} dance.")
    required = [style_clean.split(" ")[0]] if style_clean else []
    prompt_context = _safe_multiline(
        "\n".join(
            [
                "OFFICIAL LABELS (Motorica; derived from filename + README code table):",
                f"- style_code: {code}",
                f"- style_name: {style}",
                f"- prefix_tag: {prefix}",
                "Treat the style_name as ground truth.",
                f"Full id: {mid}",
            ]
        )
    )
    return {
        "id": mid,
        "base_id": base,
        "src_key": "motorica_30hz",
        "main_anchor": main_anchor,
        "required_keywords": required,
        "prompt_context": prompt_context,
        "meta": {"style_code": code, "style_name": style, "prefix": prefix},
    }


# -----------------------------------------------------------------------------
# FineDance adapter
# -----------------------------------------------------------------------------


def _finedance_label_context(mid: str, *, label_dir: Path) -> Optional[Dict]:
    base = _base_id(mid)
    _, rest = base.split("__", 1)
    # Example: finedance_30hz__001__s000000_e000197
    dance_id = rest.split("__", 1)[0]
    dance_id_z3 = str(dance_id).zfill(3)
    jp = label_dir / f"{dance_id}.json"
    if not jp.is_file():
        jp = label_dir / f"{dance_id_z3}.json"
    meta = {}
    style_words: List[str] = []
    if jp.is_file():
        try:
            meta = json.loads(_read_text(jp))
        except Exception:
            meta = {}
    if isinstance(meta, dict):
        for k in ["style1", "style2"]:
            v = str(meta.get(k, "")).strip()
            if v:
                style_words.append(v)
    style_phrase = _safe(" ".join(style_words)).lower()
    if not style_phrase:
        style_phrase = "dance"
    main_anchor = _safe(f"The person performs a {style_phrase} dance.")
    required = [w.lower() for w in style_words if w.strip()]
    prompt_context = _safe_multiline(
        "\n".join(
            [
                "OFFICIAL LABELS (FineDance; from label_json):",
                f"- dance_id: {dance_id}",
                f"- style1: {meta.get('style1', '') if isinstance(meta, dict) else ''}",
                f"- style2: {meta.get('style2', '') if isinstance(meta, dict) else ''}",
                f"- name: {meta.get('name', '') if isinstance(meta, dict) else ''}",
                "Treat these style labels as ground truth.",
                f"Full id: {mid}",
            ]
        )
    )
    return {
        "id": mid,
        "base_id": base,
        "src_key": "finedance_30hz",
        "main_anchor": main_anchor,
        "required_keywords": required,
        "prompt_context": prompt_context,
        "meta": {"dance_id": dance_id, "label_json": str(jp) if jp.is_file() else "", "label": meta},
    }


# -----------------------------------------------------------------------------


def _iter_videos(videos_dir: Path, pattern: str) -> List[Path]:
    return sorted([p for p in videos_dir.glob(pattern) if p.is_file()])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="*.mp4")
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument(
        "--only_prefixes",
        type=str,
        default="bandai_30hz__,lafan1_30hz__,motorica_30hz__,finedance_30hz__",
        help="Comma-separated id prefixes to include (match on mp4 stem). Empty = include all supported.",
    )
    ap.add_argument(
        "--finedance_label_dir",
        type=str,
        default="datasets/raw/finedance/label_json",
        help="Directory containing FineDance label JSONs (e.g., 001.json).",
    )
    ap.add_argument(
        "--lafan-theme-map", type=Path,
        help="Required for LaFAN1 inputs: user-local JSON {theme_key: source description}; upstream metadata is not bundled.",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir)
    out_jsonl = Path(args.out_jsonl)
    if out_jsonl.exists() and (not bool(args.overwrite)):
        raise SystemExit(f"Refusing to overwrite existing: {out_jsonl} (use --overwrite)")

    label_dir = Path(args.finedance_label_dir)

    only_prefixes = [p.strip() for p in str(args.only_prefixes).split(",") if p.strip()]
    vids = _iter_videos(videos_dir, str(args.pattern))
    lafan_videos = [vp for vp in vids if vp.stem.startswith("lafan1_30hz__")
                   and (not only_prefixes or any(vp.stem.startswith(p) for p in only_prefixes))]
    lafan_theme_map: Dict[str, str] = {}
    lafan_map_sha256 = None
    if lafan_videos:
        if args.lafan_theme_map is None:
            ap.error("LaFAN1 inputs require --lafan-theme-map with user-acquired source metadata")
        try:
            lafan_theme_map = _load_lafan_theme_map(args.lafan_theme_map)
            # Resolve every selected theme before opening any output file, so
            # an absent theme cannot silently skip a clip or invent a caption.
            for vp in lafan_videos:
                _lafan_label_context(vp.stem, theme_map=lafan_theme_map)
            lafan_map_sha256 = hashlib.sha256(args.lafan_theme_map.read_bytes()).hexdigest()
        except (ValueError, OSError, UnicodeError) as exc:
            ap.error(str(exc))
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    n_skip = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        for vp in vids:
            mid = vp.stem
            if only_prefixes and not any(mid.startswith(p) for p in only_prefixes):
                n_skip += 1
                continue

            rec: Optional[Dict] = None
            try:
                if mid.startswith("bandai_30hz__"):
                    rec = _bandai_label_context(mid)
                elif mid.startswith("lafan1_30hz__"):
                    rec = _lafan_label_context(mid, theme_map=lafan_theme_map)
                    rec["meta"]["theme_metadata"] = {
                        "role": "user-acquired upstream metadata; not bundled or relicensed",
                        "input_sha256": lafan_map_sha256,
                    }
                elif mid.startswith("motorica_30hz__"):
                    rec = _motorica_label_context(mid)
                elif mid.startswith("finedance_30hz__"):
                    rec = _finedance_label_context(mid, label_dir=label_dir)
                else:
                    rec = None
            except Exception:
                rec = None

            if rec is None:
                n_skip += 1
                continue

            # Add stable hashes for audit/debug.
            rec["prompt_context_sha1"] = _sha1(str(rec.get("prompt_context", "")))
            rec["main_anchor_sha1"] = _sha1(str(rec.get("main_anchor", "")))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 1000 == 0:
                print(f"[OK] wrote {n} (last={mid})")

    print(f"[DONE] wrote={n} skipped={n_skip} out={out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

