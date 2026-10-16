#!/usr/bin/env python3
"""
Convert VideoLLaMA3 caption JSONL to HumanML3D-style text files:
  <caption>#<word/POS tokens>#0.0#0.0

This script updates only ids found in the JSONL; other text files are untouched.

Example:
  conda run -n mcmldm python tools/vlm_jsonl_to_humanml_texts.py \\
    --jsonl datasets/VLM_text_label/captions/smoketest_mesh_batch_v3.videollama3.jsonl \\
    --texts_dir datasets/FineMotion/humanml3d_20hz_for_Tranning263/finemotion_263_v3/texts \\
    --backup_dir datasets/FineMotion/humanml3d_20hz_for_Tranning263/finemotion_263_v3/texts_backup_before_vlm \\
    --include_details
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _safe_caption(s: str) -> str:
    # Text files use '#' as separator.
    s = (s or "").replace("#", " ").strip()
    # collapse spaces
    s = re.sub(r"\s+", " ", s)
    return s


def _strip_role_prefixes(s: str) -> str:
    """Remove leading MAIN:/DETAILS: prefixes (sometimes repeated)."""
    s = _safe_caption(s)
    for _ in range(3):
        s2 = re.sub(r"^\s*(MAIN|DETAILS)\s*:\s*", "", s, flags=re.IGNORECASE).strip()
        if s2 == s:
            break
        s = s2
    return s.strip()


def _split_inline_details(s: str) -> Tuple[str, str]:
    """
    Some VLM outputs put both fields on one line, e.g.:
      "The person ... DETAILS: The person ..."
    """
    s = _safe_caption(s)
    if not s:
        return "", ""
    parts = re.split(r"\bDETAILS\s*:\s*", s, maxsplit=1, flags=re.IGNORECASE)
    main = (parts[0] if parts else "").strip()
    det = (parts[1] if len(parts) > 1 else "").strip()
    return main, det


def _load_spacy():
    try:
        import spacy  # type: ignore
    except Exception:
        return None, False
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp, True
    except Exception:
        # fallback tokenizer only
        nlp = spacy.blank("en")
        return nlp, False


def _tokenize(caption: str, nlp, has_pos: bool) -> str:
    caption = _safe_caption(caption)
    if not caption:
        return "unk/OTHER"

    if nlp is None:
        # regex fallback
        words = re.findall(r"[A-Za-z0-9']+", caption.lower())
        if not words:
            return "unk/OTHER"
        return " ".join([f"{w}/OTHER" for w in words])

    doc = nlp(caption)
    toks: List[str] = []
    for t in doc:
        if t.is_space or t.is_punct:
            continue
        word = (t.lemma_ if has_pos else t.text).lower().strip()
        if not word:
            continue
        if word == "-pron-":
            word = t.text.lower().strip()
        pos = (t.pos_ if has_pos else "OTHER") or "OTHER"
        toks.append(f"{word}/{pos}")
    if not toks:
        return "unk/OTHER"
    return " ".join(toks)


def _read_jsonl_latest(jsonl_path: Path) -> Dict[str, Dict]:
    latest: Dict[str, Dict] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            mid = obj.get("id")
            if not mid:
                continue
            latest[str(mid)] = obj
    return latest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True)
    ap.add_argument("--texts_dir", type=str, required=True)
    ap.add_argument("--backup_dir", type=str, default="")
    ap.add_argument("--include_details", action="store_true")
    ap.add_argument("--skip_if_exists", action="store_true", help="Skip ids whose text file already exists.")
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    texts_dir = Path(args.texts_dir)
    texts_dir.mkdir(parents=True, exist_ok=True)

    backup_dir = Path(args.backup_dir) if args.backup_dir else None
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)

    nlp, has_pos = _load_spacy()
    if nlp is None:
        print("[warn] spacy not available; using regex tokenizer + OTHER POS.")
    elif not has_pos:
        print("[warn] en_core_web_sm not found; using tokenizer-only + OTHER POS.")

    latest = _read_jsonl_latest(jsonl_path)
    ids = sorted(latest.keys())
    if int(args.limit) > 0:
        ids = ids[: int(args.limit)]

    n_ok = 0
    n_skip = 0
    n_err = 0
    n_fix_main_from_details = 0
    for mid in ids:
        obj = latest[mid]
        if "error" in obj:
            n_err += 1
            continue
        cap_main_raw = str(obj.get("caption_main", "")).strip()
        cap_det_raw = str(obj.get("caption_details", "")).strip()

        # Clean legacy/edge cases:
        # - details accidentally embedded in caption_main (single-line "DETAILS:" output)
        # - caption_details still starting with "DETAILS:" (fallback parser path)
        cap_main = _strip_role_prefixes(cap_main_raw)
        cap_det = _strip_role_prefixes(cap_det_raw)
        if (not cap_det) and re.search(r"\bDETAILS\s*:", cap_main_raw, flags=re.IGNORECASE):
            m, d = _split_inline_details(cap_main_raw)
            cap_main = _strip_role_prefixes(m)
            cap_det = _strip_role_prefixes(d)

        # If MAIN is missing but DETAILS exists, promote DETAILS to MAIN.
        # This avoids leaving placeholder texts untouched due to a single formatting failure.
        if not cap_main:
            if cap_det:
                cap_main = cap_det
                cap_det = ""
                n_fix_main_from_details += 1
            else:
                n_err += 1
                continue

        out_path = texts_dir / f"{mid}.txt"
        if bool(args.skip_if_exists) and out_path.is_file():
            n_skip += 1
            continue

        # backup existing file once
        if backup_dir is not None and out_path.is_file():
            bpath = backup_dir / out_path.name
            if not bpath.is_file():
                bpath.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")

        tok_main = _tokenize(cap_main, nlp, has_pos)
        lines = [f"{cap_main}#{tok_main}#0.0#0.0"]
        if bool(args.include_details) and cap_det:
            tok_det = _tokenize(cap_det, nlp, has_pos)
            lines.append(f"{cap_det}#{tok_det}#0.0#0.0")

        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_ok += 1

    print(
        f"[OK] updated={n_ok} skipped={n_skip} errors={n_err} "
        f"fix_main_from_details={n_fix_main_from_details} texts_dir={texts_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

