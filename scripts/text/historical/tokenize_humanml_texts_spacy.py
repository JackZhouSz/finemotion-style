#!/usr/bin/env python3
"""
Tokenize HumanML3D-style text files with spaCy (en_core_web_sm) and write tokens in-place.

Target format per line:
  <caption>#<word/POS tokens>#f_tag#to_tag

This script is intended to upgrade placeholder tokens like:
  caption#unk/OTHER#0.0#0.0
into:
  caption#the/DET person/NOUN ...#0.0#0.0

Typical usage (MotionX only):
  conda run -n mcmldm python tools/tokenize_humanml_texts_spacy.py \\
    --texts_dir datasets/FineMotion/humanml3d_20hz_for_Tranning263/finemotion_263_v4/texts \\
    --prefix motionx_ \\
    --backup_dir datasets/FineMotion/humanml3d_20hz_for_Tranning263/finemotion_263_v4/texts_backup_before_motionx_tokenize

If caption is empty, we try to recover from MotionX raw text root based on the id,
otherwise we fall back to a caption derived from the id stem.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple


SEG_RE = re.compile(r"__s(?P<s>\d{6})_e(?P<e>\d{6})$")


def _base_id(mid: str) -> str:
    m = SEG_RE.search(mid)
    return mid[: m.start()] if m else mid


def _safe_caption(s: str) -> str:
    s = (s or "").replace("#", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _load_spacy():
    try:
        import spacy  # type: ignore
    except Exception:
        return None, False
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp, True
    except Exception:
        nlp = spacy.blank("en")
        return nlp, False


def _tokenize(caption: str, nlp, has_pos: bool) -> str:
    caption = _safe_caption(caption)
    if not caption:
        return "unk/OTHER"

    if nlp is None:
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


def parse_humanml_line(line: str) -> Optional[Tuple[str, str, str, str]]:
    parts = line.rstrip("\n").split("#")
    if len(parts) < 2:
        return None
    caption = parts[0].strip()
    tokens = parts[1].strip()
    f_tag = parts[2].strip() if len(parts) >= 3 else "0.0"
    to_tag = parts[3].strip() if len(parts) >= 4 else "0.0"
    return caption, tokens, f_tag, to_tag


def _safe_tag(v: str) -> str:
    s = (v or "").strip()
    try:
        x = float(s)
    except Exception:
        return "0.0"
    # NaN -> 0.0
    if x != x:
        return "0.0"
    return f"{x:.1f}"


def format_humanml_line(caption: str, tokens: str, f_tag: str, to_tag: str) -> str:
    caption = _safe_caption(caption)
    tokens = (tokens or "").strip() or "unk/OTHER"
    f_tag = _safe_tag(f_tag)
    to_tag = _safe_tag(to_tag)
    return f"{caption}#{tokens}#{f_tag}#{to_tag}"


def _derive_caption_from_id(mid: str) -> str:
    # best-effort: use last "__" segment, remove common suffix tokens
    s = _base_id(mid)
    tail = s.split("__")[-1]
    tail = re.sub(r"Re$", "", tail)
    tail = re.sub(r"_+", " ", tail).strip()
    tail = re.sub(r"\bclip\b", "", tail, flags=re.IGNORECASE).strip()
    tail = re.sub(r"\s+", " ", tail).strip()
    if not tail:
        return "The person performs a motion."
    # Make a simple sentence (avoid exceeding length; we don't hard-trim here).
    return f"The person performs {tail}."


def _recover_motionx_caption_from_raw(mid: str, motionx_text_root: Path) -> str:
    """
    Recover raw MotionX caption using the same relpath encoding as build script:
      motionx_clean_30hz__<relpath_flattened>
    where <relpath_flattened> uses '__' for '/'.
    """
    base = _base_id(mid)
    if not base.startswith("motionx_clean_30hz__"):
        return ""
    rel_flat = base[len("motionx_clean_30hz__") :]
    rel = rel_flat.replace("__", "/") + ".txt"
    p = motionx_text_root / rel
    if not p.is_file():
        return ""
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if ln:
            return ln
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts_dir", type=str, required=True)
    ap.add_argument("--backup_dir", type=str, default="")
    ap.add_argument("--prefix", type=str, nargs="*", default=[], help="Only process files whose name starts with any prefix.")
    ap.add_argument(
        "--allow_fallback",
        action="store_true",
        help="Allow fallback tokenization when spaCy/en_core_web_sm is unavailable (will produce OTHER POS).",
    )
    ap.add_argument(
        "--only_if_tokens_are",
        type=str,
        default="unk/OTHER",
        help="Only rewrite lines whose tokens exactly equal this value (default: unk/OTHER). Set empty to rewrite all.",
    )
    ap.add_argument(
        "--motionx_text_root",
        type=str,
        default="",
        help="Optional MotionX raw text root for recovering captions.",
    )
    ap.add_argument(
        "--motionx_force_raw_caption",
        action="store_true",
        help="If set, always overwrite MotionX captions from motionx_text_root when possible.",
    )
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    texts_dir = Path(args.texts_dir)
    if not texts_dir.is_dir():
        raise SystemExit(f"[err] texts_dir not found: {texts_dir}")
    backup_dir = Path(args.backup_dir) if args.backup_dir else None
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)

    prefixes = [p for p in (args.prefix or []) if p]
    only_if = str(args.only_if_tokens_are or "").strip()

    motionx_text_root = Path(args.motionx_text_root) if str(args.motionx_text_root).strip() else None

    nlp, has_pos = _load_spacy()
    if nlp is None:
        if not bool(args.allow_fallback):
            raise SystemExit(
                "[err] spaCy not available. Run in the correct environment (e.g. conda -n mcmldm) "
                "or pass --allow_fallback (not recommended)."
            )
        print("[warn] spaCy not available; using regex tokenizer + OTHER POS.")
    elif not has_pos:
        if not bool(args.allow_fallback):
            raise SystemExit(
                "[err] en_core_web_sm not found. Please install it in your env, or pass --allow_fallback "
                "(not recommended)."
            )
        print("[warn] en_core_web_sm not found; using tokenizer-only + OTHER POS.")

    files = sorted(texts_dir.glob("*.txt"))
    if prefixes:
        files = [p for p in files if any(p.name.startswith(pre) for pre in prefixes)]
    if int(args.limit) > 0:
        files = files[: int(args.limit)]

    n_files = 0
    n_lines = 0
    n_rewritten = 0
    n_skipped = 0
    n_empty_caption_fixed = 0
    n_bad = 0

    for p in files:
        n_files += 1
        raw = p.read_text(encoding="utf-8", errors="replace")
        lines_in = [ln.rstrip("\n") for ln in raw.splitlines() if ln.strip()]
        if not lines_in:
            n_skipped += 1
            continue

        changed = False
        lines_out: List[str] = []
        mid = p.stem
        for ln in lines_in:
            parsed = parse_humanml_line(ln)
            if parsed is None:
                lines_out.append(ln)
                n_bad += 1
                continue
            cap, toks, f_tag, to_tag = parsed

            # Only rewrite specific placeholder tokens unless user requests rewrite all.
            if only_if and toks != only_if:
                lines_out.append(format_humanml_line(cap, toks, f_tag, to_tag))
                continue

            cap2 = cap
            # MotionX: optionally recover caption from raw text even if current caption is non-empty.
            if bool(args.motionx_force_raw_caption) and motionx_text_root is not None:
                recovered = _recover_motionx_caption_from_raw(mid, motionx_text_root)
                if recovered:
                    cap2 = recovered
                    changed = True
            if not _safe_caption(cap2):
                recovered = ""
                if motionx_text_root is not None:
                    recovered = _recover_motionx_caption_from_raw(mid, motionx_text_root)
                cap2 = recovered or _derive_caption_from_id(mid)
                n_empty_caption_fixed += 1
                changed = True

            toks2 = _tokenize(cap2, nlp, has_pos)
            if toks2 != toks:
                changed = True
                n_rewritten += 1
            lines_out.append(format_humanml_line(cap2, toks2, f_tag, to_tag))
            n_lines += 1

        if not changed:
            n_skipped += 1
            continue

        if backup_dir is not None and p.is_file():
            b = backup_dir / p.name
            if not b.is_file():
                b.write_text(raw, encoding="utf-8")

        p.write_text("\n".join(lines_out) + "\n", encoding="utf-8")

    print(
        f"[OK] files={n_files} rewritten_lines={n_rewritten} fixed_empty_caption_lines={n_empty_caption_fixed} "
        f"skipped_files={n_skipped} bad_lines={n_bad} texts_dir={texts_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

