#!/usr/bin/env python3
"""Map historical caption_main/caption_details JSONL to the public formatter.

No model calls, caption cleanup, POS generation, or source-file writes occur.
MAIN and DETAILS become separate alternative descriptions of the same clip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

# Reuse the public formatter's filename/caption/token validation, without
# loading spaCy. This also keeps punctuation-bearing source IDs unchanged.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_training_texts import (  # noqa: E402
    TextFormatError, check_id, validate_caption, validate_tokens,
)


def convert(input_path: Path, include_details: bool, duplicates: str):
    latest = {}
    duplicates_seen = []
    source_bytes = input_path.read_bytes()
    for number, line in enumerate(source_bytes.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        origin = f"{input_path.name}:{number}"
        obj = json.loads(line)
        if not isinstance(obj, dict) or not isinstance(obj.get("id"), str):
            raise ValueError(f"{origin}: require an object with a string id")
        check_id(obj["id"])
        if "error" in obj:
            raise ValueError(f"{origin}: model error record; resolve it before export")
        if obj["id"] in latest:
            duplicates_seen.append({"id": obj["id"], "replaced_input_line": latest[obj["id"]][0],
                                    "selected_input_line": number})
            if duplicates == "reject":
                raise ValueError(f"{origin}: duplicate id; select records explicitly or use --duplicates latest")
        latest[obj["id"]] = (number, obj)
    if not latest:
        raise ValueError("No caption objects found")

    records = []
    lineage = []
    for clip_id in sorted(latest):
        number, obj = latest[clip_id]
        fields = ["caption_main"] + (["caption_details"] if include_details else [])
        for field in fields:
            caption = obj.get(field)
            if field == "caption_details" and (caption is None or caption == ""):
                continue
            if not isinstance(caption, str):
                raise ValueError(f"{clip_id}: {field} must be a string")
            validate_caption(caption, f"{clip_id}.{field}")
            # The historical exporter stripped these markers and repaired some
            # malformed responses. This adapter refuses to alter the wording.
            if re.search(r"\b(?:MAIN|DETAILS)\s*:", caption, flags=re.IGNORECASE):
                raise ValueError(f"{clip_id}.{field}: role marker inside caption; review explicitly")
            start, end = str(obj.get("from", "0.0")), str(obj.get("to", "0.0"))
            times = [float(start), float(end)]
            if (any(not math.isfinite(t) or t < 0 for t in times)
                    or (times != [0.0, 0.0] and times[1] <= times[0])):
                raise ValueError(f"{clip_id}: invalid from/to interval")
            record = {"id": clip_id, "caption": caption, "from": start, "to": end}
            token_key = "tokens_main" if field == "caption_main" else "tokens_details"
            if token_key in obj:
                if not isinstance(obj[token_key], str):
                    raise ValueError(f"{clip_id}.{token_key}: tokens must be a string")
                validate_tokens(obj[token_key], f"{clip_id}.{token_key}")
                record["tokens"] = obj[token_key]
            records.append(record)
            lineage.append({"id": clip_id, "output_line": len(records),
                            "input_line": number, "input_field": field,
                            "caption_sha256": hashlib.sha256(caption.encode("utf-8")).hexdigest()})
    report = {"schema_version": 1, "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
              "selected_unique_ids": len(latest), "output_caption_rows": len(records),
              "include_details": include_details, "duplicate_policy": duplicates,
              "duplicate_selections": duplicates_seen, "caption_rewrites": 0,
              "tokenization_performed": False, "model_calls": 0, "lineage": lineage}
    return records, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path,
                        help="New file; also creates <output>.manifest.json")
    parser.add_argument("--main-only", action="store_true", help="Omit the DETAILS alternative")
    parser.add_argument("--duplicates", choices=("reject", "latest"), default="reject",
                        help="latest selects the last input row per ID and records each selection")
    args = parser.parse_args()
    try:
        output = args.output_jsonl.resolve()
        manifest = output.with_name(output.name + ".manifest.json")
        if output.exists() or manifest.exists():
            raise FileExistsError("Output or manifest already exists; choose a new output path")
        if output == args.input_jsonl.resolve():
            raise ValueError("Output must differ from input")
        records, report = convert(args.input_jsonl, not args.main_only, args.duplicates)
        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        report["output_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        with manifest.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(json.dumps({"status": "prepared", "ids": len({r["id"] for r in records}),
                          "caption_rows": len(records), "caption_rewrites": 0,
                          "output": str(output)}, ensure_ascii=True))
        return 0
    except (ValueError, TextFormatError, OSError, UnicodeError) as exc:
        parser.exit(2, f"[error] {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
