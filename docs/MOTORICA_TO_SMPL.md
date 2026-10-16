# Motorica BVH to SMPL with MotionBuilder

The author used MotionBuilder's automatic source-character definition for the
original Motorica skeleton, then the same retarget/bake/SMPL-export method used
in the [Bandai example](BVH_TO_SMPL.md). The calibration pose was made by taking
one source BVH and setting its rotation channels to zero. A user-local FBX can
save that source character for later clips; the old author's FBX is not needed
to follow this procedure.

This documents the author-confirmed historical operation and a tested BVH
preparation helper. A fresh original-Motorica-to-archived-cache comparison has
not been run. Existing local Motorica previews are already retargeted SMPL
skeletons, so they are not evidence of original-source characterization.

## 1. Obtain an original Motorica BVH

Acquire [Motorica Dance Dataset](https://github.com/simonalexanderson/MotoricaDanceDataset)
from its official provider under its research-use terms. Keep the native source
hierarchy and filenames. Obtain MotionBuilder and a characterized SMPL target
under their own terms, as in the Bandai tutorial.

## 2. Generate a calibration pose locally

From the repository root:

```bash
mkdir -p work/motorica
python scripts/motorica/prepare_motorica_calibration.py \
  --input-bvh /path/to/acquired/original_motorica.bvh \
  --output-bvh work/motorica/calibration.bvh
```

The Python helper uses only the standard library. It writes a one-frame BVH,
`calibration.bvh.manifest.json` and `calibration.bvh.SHA256SUMS`. Choose a fresh
output filename. `--source-frame` selects the zero-based input frame whose
translation values are retained; its default is zero.

The helper preserves the source hierarchy, bone names, OFFSET values, channel
order and every position channel, including any child translations. It sets
only rotation channels to zero. The original BVH is unchanged. The manifest
records source/output hashes, channel counts, frame time and index history.

Zero rotations give the skeleton's bind/rest pose. The author used that pose
as the Motorica T-pose; this is not a guarantee that every other source skeleton
becomes a T-pose. Confirm arm alignment, facing direction and the resulting
stance in MotionBuilder before characterizing it.

## 3. Define the source Character and save it if useful

Import the calibration BVH. Use MotionBuilder's automatic characterization or
its matching HIK naming template to populate the Character definition. Confirm
the required bones and valid characterization. The author reports that the
Motorica source was recognized automatically.

Autodesk documents automatic mapping for recognized bone names and template
mapping for other conventions: [automatic characterization](https://help.autodesk.com/cloudhelp/2023/ENU/MotionBuilder/files/GUID-0008F9D7-A2A4-4A30-A149-BA6C03F9910A.htm)
and [HIK skeleton definition](https://help.autodesk.com/cloudhelp/2025/ENU/MotionBuilder-Character-Animation/files/Character-Animation/HIK-character-in-MotionBuilder/Set-up-an-HIK-character/GUID-25126449-FAA8-4F6C-B033-B93816140182.html).
Automatic mapping depends on the imported names, namespace/prefix and pose;
check the definition rather than assuming that an imported skeleton is already
a valid Character.

Save a user-local FBX scene with the characterized source, or save the naming
definition, to reuse it with the same source skeleton. Reuse the mapping/stance
while applying each clip's own animation. This repository does not bundle a
Motorica-derived FBX or an SMPL target template.

## 4. Retarget, bake and export the SMPL skeleton

Connect the source Character to the separately characterized SMPL target.
Retarget the animation, bake the target motion, and export the target skeleton
to BVH. Convert that SMPL-skeleton BVH to 24-joint axis-angle parameters with the
[2025 conversion implementation](https://github.com/dongran/tempo-changing-music2motion)
and the recorded export settings. The resulting training motion convention is
`poses(T,24,3)` in radians and `trans(T,3)` in metres.

The shared method does not make the Bandai source mapping or every Bandai
setting a Motorica setting. Record the actual source/target names, units,
characterization result, export rate, frame range and helper revision. The
recovered local generic finalizer and the pinned Bandai helper are different
versions; their translation-scale defaults also differ. Select the scale from
the actual BVH units and verify the resulting translation, rather than copying
a default from another source.

### Optional: prepend calibration before the motion

For a batch route that characterizes the first frame of each input:

```bash
python scripts/motorica/prepare_motorica_calibration.py \
  --input-bvh /path/to/acquired/original_motorica.bvh \
  --output-bvh work/motorica/with_calibration.bvh --prepend
```

This adds one calibration frame and preserves all original motion-row bytes.
At the input rate, original frame `i` becomes `i+1`. Remove the calibration
interval according to the actual MotionBuilder bake/export timeline. Changing
the sampling rate changes that interval's sample representation; the Bandai
example's final frame-removal count is not a universal rule.

## 5. Match the historical clip IDs and text format

The recovered [cache standardizer](../scripts/reconstruction/historical/add_motorica_finedance_to_finemotion.py)
records the historical 120-to-30Hz Motorica intermediate route. Common feature
processing uses the paired 20Hz timeline. Keep the source-recording/crop ledger
and use the [frozen protocol](BENCHMARK_PROTOCOL.md#frozen-ids-with-regenerated-text)
for ID membership and partitions. Generate MAIN/DETAILS and lemma/POS using the
[text pipeline](TEXT_FORMAT.md), with a separately named regenerated-text version.

The calibration helper was checked on synthetic mixed-order channels and a real
Bandai BVH for channel preservation. These checks did not run MotionBuilder or
verify original Motorica retargeting quality. A saved source Character is a
convenience; the original source skeleton, valid calibration and recorded
retarget/export settings are the reproducible inputs.
