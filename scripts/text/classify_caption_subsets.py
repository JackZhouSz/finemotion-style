#!/usr/bin/env python3
"""Run archived style/content rules on a caption copy into a NEW directory.

This creates a new caption-derived version. It never replaces frozen paper lists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
from export_frozen_benchmark import load_partitions, split_map


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path,
                        help="Contains texts/ and splits/{all,train,val,test}.txt")
    parser.add_argument("--out-dir", required=True, type=Path, help="Must not exist or overlap the input")
    parser.add_argument("--view", required=True, choices=("style", "content"))
    parser.add_argument("--no-style-exclude", action="store_true",
                        help="Content only: disable historical strong-style exclusion")
    args = parser.parse_args()
    if args.no_style_exclude and args.view != "content":
        parser.error("--no-style-exclude applies only to content")
    try:
        dataset = args.dataset_root.resolve(strict=True)
        out = args.out_dir.resolve()
        code = Path(__file__).resolve().parent
        if out.exists() or args.out_dir.is_symlink():
            raise FileExistsError("Output already exists; choose a new version directory")
        for source in (dataset, code):
            if out == source or source in out.parents or out in source.parents:
                raise ValueError("Output must not overlap dataset or code")
        if not (dataset / "texts").is_dir():
            raise ValueError("Missing texts directory")
        parts = load_partitions(dataset / "splits")
        partition_map = split_map(parts)
        if set(partition_map) != set(parts["all"]):
            raise ValueError("Base all.txt must equal the disjoint train/val/test union")
        name = ("build_splits_sem_v3_1_from_263_texts.py" if args.view == "style"
                else "build_finemotion_content_lists_from_texts.py")
        script = code / "historical" / name
        manifest = json.loads((code / "historical" / "manifest.json").read_text(encoding="utf-8"))
        hashes = {row["file"]: row["sha256"] for row in manifest["files"]}
        for dependency in (name, "semantic_subset_config_v3.py"):
            actual = hashlib.sha256((code / "historical" / dependency).read_bytes()).hexdigest()
            if actual != hashes[dependency]:
                raise ValueError(f"Historical implementation hash differs: {dependency}")
        command = [sys.executable, str(script), "--dataset_root", str(dataset),
                   "--out_split_root" if args.view == "style" else "--out_root", str(out)]
        if args.no_style_exclude:
            command.append("--no_style_exclude")
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
        result = subprocess.run(command, env=environment, check=False)
        if result.returncode:
            return result.returncode
        receipt = {"view": args.view, "historical_script": name,
                   "historical_script_sha256": hashes[name],
                   "base_split_sha256": {p: hashlib.sha256((dataset / "splits" / f"{p}.txt").read_bytes()).hexdigest()
                                         for p in ("all", "train", "val", "test")},
                   "historical_paper_lists_replaced": False,
                   "description": "New caption-derived view; source-specific snapshot determines membership."}
        with (out / "_classification_manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return 0
    except (OSError, ValueError) as exc:
        parser.exit(2, f"[error] {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
