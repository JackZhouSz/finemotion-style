# Bandai BVH companion: original input and SMPL-derived view

The separate BVH companion adds two complete Bandai views to the existing
motion/text package:

- `native_bvh/dataset-1/` and `native_bvh/dataset-2/`: 3,077 original upstream
  animation BVH files, preserving their source skeletons and file bytes.
- `smpl_bvh_30hz/`: 3,077 existing SMPL-skeleton animation BVH files corresponding
  to the archived 30Hz SMPL24 sequences. These are a later BVH export view of the
  SMPL cache, **not the original MotionBuilder export stage**.

The two views cover the same sequence IDs as the main Bandai NPZ package.
Source-sequence counts differ from the 2,676 downstream feature/text clip IDs.
The companion contains no duplicate NPZ/features/captions, other-source motion,
AppleDouble `._*` sidecars, standalone target templates, FBX or body-model weights.

## Mapping and verification

`metadata/sequence_mapping.jsonl` records the correspondence between each native
source, SMPL sequence ID and SMPL-derived BVH, with file hashes and frame/layout
information. `metadata/clip_to_sequence_mapping.jsonl` maps the 2,676 paired
feature/text clips to their source sequences. A source sequence can contribute
multiple windows or no downstream feature/text clip in this build.

All 3,077 derived BVH frame counts and sampling rates were checked against the
archived NPZ ledger. Three sampled motions match the recovered writer's numeric
formatting exactly. These checks establish file/mapping consistency, not a new
MotionBuilder run or a full visual review. The checksum index covers the payload
files; all regular archive members, including that index, were read back and
verified against their expected SHA256 values.

The derived files use an SMPL skeleton with root `Pelvis`, 24 joints and a 30Hz
timeline. The native wave example has root `joint_Root`, 22 joints and 187 frames;
its derived view has 186 frames. Different skeletons, body proportions and
recorded frame handling are expected. The native frame time must be read per
file rather than inferred from the derived directory name.

The reconstruction stage is:

```text
native Bandai BVH
    -> calibration and MotionBuilder retargeting
    -> archived SMPL24 NPZ
    -> later SMPL-skeleton BVH export/view
```

The [worked wave example](BVH_TO_SMPL.md) documents its actual preparation,
retargeting and frame history. Those single-example frame counts are not a
universal rule for all Bandai motions. The companion preserves the existing
files and does not claim that every historical MotionBuilder run was recovered
or rerun.

## Terms and attribution

Original and adapted Bandai motion retain **CC BY-NC 4.0**, source attribution
to Bandai Namco Research Inc., the original notices and the transformation
history. Both source licence texts accompany the archive.

SMPL-Body skeleton-rig material retains separate **CC BY 4.0** attribution to the
Max Planck Institute for Intelligent Systems; see the [official SMPL-Body terms](https://smpl.is.tue.mpg.de/bodylicense.html).
This does not grant rights to SMPL-Model software or shape-generation assets.
The animation's Bandai terms and the rig's attribution remain distinct.

Read the companion's `NOTICE.md`, `README.md` and checksum files. Preserve the
native and derived labels in any repackaging. Archive filenames, bytes, SHA256
and hosting status are recorded in the [release archive manifest](../release_metadata/release_archives.json).
