#!/usr/bin/env python3
"""Create a zero-rotation BVH calibration frame using only the standard library.

Default: one frame for creating a user-local MotionBuilder source Character/FBX.
With --prepend: calibration followed by the original motion rows, byte for byte.
Only rotation-channel numeric tokens change in the calibration frame. Hierarchy,
OFFSETs, channel order, frame time and all position tokens are preserved.

Zero rotations expose the skeleton's stored bind/rest pose. The author reports a
T-pose for the Motorica skeleton used in this workflow; this is not a generic
BVH-to-T-pose algorithm. Visually verify the result before Characterize. This
helper does not run MotionBuilder, characterize, retarget or resample anything.

Provenance: project-authored parser and transformation; no third-party code or
source dataset is bundled. Source data and MotionBuilder retain their own terms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys


NUMBER = re.compile(rb"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\Z")
CHANNEL_NAMES = {axis + kind for axis in "XYZ" for kind in ("position", "rotation")}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def finite_number(token: bytes, context: str) -> float:
    if not NUMBER.fullmatch(token):
        raise ValueError(f"Invalid numeric token in {context}: {token!r}")
    number = float(token)
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite number in {context}")
    return number


def parse_hierarchy(data: bytes) -> list[dict]:
    """Validate standard BVH hierarchy and return its flat channel order."""
    text = data.decode("utf-8-sig")
    if "\x00" in text:
        raise ValueError("NUL bytes are not supported")
    tokens = re.findall(r"[{}]|[^\s{}]+", text)
    cursor = 0
    channels: list[dict] = []

    def take(expected: str | None = None) -> str:
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError("Unexpected end of BVH hierarchy")
        token = tokens[cursor]
        cursor += 1
        if expected is not None and token != expected:
            raise ValueError(f"Expected {expected!r} in hierarchy, got {token!r}")
        return token

    def offset() -> None:
        take("OFFSET")
        for _ in range(3):
            finite_number(take().encode("ascii"), "OFFSET")

    def joint(kind: str, parent: str = "", depth: int = 0) -> None:
        if depth > 128:
            raise ValueError("Hierarchy depth exceeds 128")
        take(kind)
        name = take()
        if name in ("{", "}"):
            raise ValueError("Missing joint name")
        joint_path = parent + "/" + name if parent else name
        take("{")
        offset()
        take("CHANNELS")
        count_token = take()
        if not re.fullmatch(r"[0-9]+", count_token):
            raise ValueError("CHANNELS count must be a positive integer")
        count = int(count_token)
        if not 1 <= count <= 6:
            raise ValueError("Each joint must declare 1 to 6 standard BVH channels")
        names = [take() for _ in range(count)]
        if len(set(names)) != count or any(name not in CHANNEL_NAMES for name in names):
            raise ValueError("Invalid or duplicate BVH channel name")
        for channel_name in names:
            channels.append({"index": len(channels), "joint_path": joint_path,
                             "name": channel_name, "zero_in_calibration": channel_name.endswith("rotation")})
        while cursor < len(tokens) and tokens[cursor] != "}":
            if tokens[cursor] == "JOINT":
                joint("JOINT", joint_path, depth + 1)
            elif tokens[cursor] == "End":
                take("End")
                take("Site")
                take("{")
                offset()
                take("}")
            else:
                raise ValueError(f"Unexpected hierarchy token {tokens[cursor]!r}")
        take("}")

    take("HIERARCHY")
    joint("ROOT")
    if cursor != len(tokens):
        raise ValueError("Unexpected content after ROOT hierarchy")
    if not any(channel["zero_in_calibration"] for channel in channels):
        raise ValueError("No rotation channels found")
    return channels


def parse_bvh(data: bytes) -> dict:
    lines = data.splitlines(keepends=True)
    motion_indices = [index for index, line in enumerate(lines) if line.strip() == b"MOTION"]
    if len(motion_indices) != 1:
        raise ValueError("Expected exactly one MOTION header")
    motion_index = motion_indices[0]
    hierarchy = b"".join(lines[:motion_index])
    channels = parse_hierarchy(hierarchy)
    header_indices = [index for index in range(motion_index + 1, len(lines)) if lines[index].strip()]
    if len(header_indices) < 3:
        raise ValueError("Missing frame headers or motion rows")
    frames_index, time_index = header_indices[:2]
    frames_match = re.fullmatch(rb"(\s*Frames\s*:\s*)([0-9]+)(\s*)", lines[frames_index])
    time_match = re.fullmatch(rb"\s*Frame\s+Time\s*:\s*(\S+)\s*", lines[time_index])
    if not frames_match or not time_match:
        raise ValueError("Malformed Frames or Frame Time header")
    frame_count = int(frames_match.group(2))
    if frame_count < 1:
        raise ValueError("Frames must be positive")
    frame_time = finite_number(time_match.group(1), "Frame Time")
    if frame_time <= 0:
        raise ValueError("Frame Time must be positive")
    row_indices = [index for index in range(time_index + 1, len(lines)) if lines[index].strip()]
    if len(row_indices) != frame_count:
        raise ValueError(f"Declared {frame_count} frames but found {len(row_indices)} motion rows")
    for frame, index in enumerate(row_indices):
        tokens = lines[index].split()
        if len(tokens) != len(channels):
            raise ValueError(f"Frame {frame} has {len(tokens)} values; expected {len(channels)}")
        for token in tokens:
            finite_number(token, f"frame {frame}")
    return {"lines": lines, "channels": channels, "hierarchy": hierarchy,
            "frames_index": frames_index, "frames_match": frames_match,
            "time_index": time_index, "frame_count": frame_count,
            "frame_time": frame_time, "frame_time_token": time_match.group(1).decode("ascii"),
            "row_indices": row_indices, "motion_data": b"".join(lines[time_index + 1:])}


def transform(data: bytes, source_frame: int, prepend: bool) -> tuple[bytes, dict]:
    parsed = parse_bvh(data)
    count = parsed["frame_count"]
    if not 0 <= source_frame < count:
        raise ValueError(f"--source-frame must be between 0 and {count - 1}")
    lines = parsed["lines"]
    original_row = lines[parsed["row_indices"][source_frame]]
    tokens = list(re.finditer(rb"\S+", original_row))
    calibration = original_row
    rotation_indices = [channel["index"] for channel in parsed["channels"] if channel["zero_in_calibration"]]
    for index in reversed(rotation_indices):
        match = tokens[index]
        calibration = calibration[:match.start()] + b"0" + calibration[match.end():]
    if calibration.endswith(b"\r\n"):
        newline = b"\r\n"
    elif calibration.endswith(b"\n"):
        newline = b"\n"
    elif lines[parsed["time_index"]].endswith(b"\r\n"):
        newline = b"\r\n"
    else:
        newline = b"\n"
    calibration = calibration.rstrip(b"\r\n") + newline
    new_count = count + 1 if prepend else 1
    prefix = list(lines[:parsed["time_index"] + 1])
    frames_match = parsed["frames_match"]
    index = parsed["frames_index"]
    prefix[index] = frames_match.group(1) + str(new_count).encode("ascii") + frames_match.group(3)
    # A source without a newline after Frame Time cannot contain valid row lines.
    result = b"".join(prefix) + calibration + (parsed["motion_data"] if prepend else b"")
    output_parsed = parse_bvh(result)
    assert output_parsed["hierarchy"] == parsed["hierarchy"]
    assert output_parsed["channels"] == parsed["channels"]
    assert output_parsed["frame_count"] == new_count
    calibration_tokens = calibration.split()
    original_tokens = original_row.split()
    for channel in parsed["channels"]:
        index = channel["index"]
        assert calibration_tokens[index] == (b"0" if channel["zero_in_calibration"] else original_tokens[index])
    history = [{"output_start_inclusive": 0, "output_end_exclusive": 1,
                "kind": "zero_rotation_calibration", "position_tokens_from_source_frame": source_frame,
                "rotation_tokens_set_to_zero": True}]
    if prepend:
        history.append({"output_start_inclusive": 1, "output_end_exclusive": count + 1,
                        "kind": "unchanged_original_motion_rows", "source_start_inclusive": 0,
                        "source_end_exclusive": count, "output_minus_source_index": 1,
                        "original_data_section_bytes_preserved": True})
    metadata = {
        "format_version": 1, "operation": "prepend_calibration" if prepend else "one_frame_calibration",
        "source_frames": count, "output_frames": new_count, "source_frame_zero_based": source_frame,
        "frame_time_seconds": parsed["frame_time"], "frame_time_numeric_token": parsed["frame_time_token"],
        "frame_rate_changed": False, "motion_retargeted": False, "motionbuilder_invoked": False,
        "hierarchy_bytes_preserved": True, "hierarchy_sha256": sha(parsed["hierarchy"]),
        "source_motion_data_sha256": sha(parsed["motion_data"]),
        "selected_source_row_sha256": sha(original_row), "calibration_row_sha256": sha(calibration),
        "calibration_nonrotation_numeric_tokens_preserved": True,
        "channels": parsed["channels"], "frame_history": history,
        "pose_scope": "Zero rotations reveal the stored bind/rest pose. The author reports a T-pose for the Motorica skeleton used in this workflow; visually verify before Characterize. This is not a generic T-pose guarantee.",
        "prepend_scope": "When prepended, original source frame i is output frame i+1; remove only the added calibration frame after retargeting if retaining the original motion timeline.",
        "provenance": "Project-authored standard-library helper; no source dataset or MotionBuilder implementation bundled.",
    }
    return result, metadata


def checked_paths(input_path: Path, output_path: Path) -> tuple[Path, Path, Path, Path]:
    source = input_path.expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("Input must be a regular BVH file")
    output = output_path.expanduser().absolute()
    manifest = output.with_name(output.name + ".manifest.json")
    sums = output.with_name(output.name + ".SHA256SUMS")
    resolved = [source]
    for path in (output, manifest, sums):
        if path.exists() or path.is_symlink():
            raise ValueError(f"Refusing to overwrite existing output: {path}")
        if not path.parent.is_dir():
            raise ValueError(f"Output parent directory must already exist: {path.parent}")
        candidate = path.resolve()
        if any(os.path.normcase(str(candidate)) == os.path.normcase(str(other)) for other in resolved):
            raise ValueError("Input/output paths must be separate")
        resolved.append(candidate)
    return source, output, manifest, sums


def run(args: argparse.Namespace) -> dict:
    source, output, manifest_path, sums_path = checked_paths(args.input_bvh, args.output_bvh)
    original = source.read_bytes()
    output_data, metadata = transform(original, args.source_frame, args.prepend)
    source_hash = sha(original)
    output_hash = sha(output_data)
    metadata.update({"input_filename": source.name, "input_bytes": len(original), "input_sha256": source_hash,
                     "output_filename": output.name, "output_bytes": len(output_data), "output_sha256": output_hash,
                     "script_filename": Path(__file__).name, "script_sha256": sha(Path(__file__).read_bytes())})
    manifest_data = (json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    manifest_hash = sha(manifest_data)
    sums_data = (f"{output_hash}  {output.name}\n{manifest_hash}  {manifest_path.name}\n").encode("utf-8")
    if sha(source.read_bytes()) != source_hash:
        raise ValueError("Source changed during preparation; no outputs written")
    # Validate every input and all three destination names before creating files.
    # Exclusive creation also prevents overwriting a destination created meanwhile.
    created: list[Path] = []
    try:
        for path, content in ((output, output_data), (manifest_path, manifest_data), (sums_path, sums_data)):
            with path.open("xb") as handle:
                created.append(path)
                handle.write(content)
        if source.read_bytes() != original:
            raise ValueError("Source changed during output creation; new outputs removed")
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return {"status": "created", "output_bvh": str(output), "output_sha256": output_hash,
            "manifest": str(manifest_path), "manifest_sha256": manifest_hash, "checksums": str(sums_path),
            "source_sha256_unchanged": source_hash, "source_frames": metadata["source_frames"],
            "output_frames": metadata["output_frames"], "source_frame_zero_based": args.source_frame,
            "mode": metadata["operation"], "visual_pose_verification_required": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-bvh", required=True, type=Path, help="User-acquired original BVH; never modified")
    parser.add_argument("--output-bvh", required=True, type=Path, help="Fresh BVH path in an existing output directory")
    parser.add_argument("--source-frame", type=int, default=0, help="Zero-based source frame supplying position tokens (default: 0)")
    parser.add_argument("--prepend", action="store_true", help="Prepend calibration to original rows instead of writing only one frame")
    args = parser.parse_args()
    try:
        result = run(args)
    except (OSError, ValueError, UnicodeError, RecursionError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
