# FineDance native NPY to SMPL parameters

These are the project's recovered native converter and batch driver. They start
from FineDance motion NPY files, before the cached SMPL stage used by the unified
FineMotion importer. No FineDance motion, label, music, cached parameters,
body-model files or upstream implementation is included here. Acquire source
data under its own terms and keep generated data and run metadata outside Git.

The original files are recorded in project commit
`7a28f561c35d04964c352a205aca08f7b46b6be8`, author `dongran`, dated
2026-02-09. `recovery_manifest.json` records original and public SHA-256 values.
Only CRLF-to-LF normalization was applied to these two Python files.

## Usage

Use Python with NumPy. No SMPL model, GPU, MotionBuilder, PyTorch3D installation
or rendering dependency is required for this parameter conversion.
On Windows, use `python -X utf8` when local paths contain non-ASCII characters.

```bash
python scripts/finedance/finedance_to_smpl_npz.py \
  --in-npy /path/to/FineDance/motion/078.npy \
  --out-npz work/finedance_native/078.npz \
  --hand-mode index1 --add-height-y 0

python scripts/finedance/batch_finedance_motion_to_smpl_npz.py \
  --in-dir /path/to/FineDance/motion \
  --out-dir work/finedance_native_batch \
  --hand-mode index1 --add-height-y 0 --workers 1 \
  --manifest work/finedance_native_batch_manifest.json
```

Choose and record the hand policy explicitly: the historical **single-file
default is `zeros`**, while the **batch default is `index1`**. The examples above
choose `index1`; use `--hand-mode zeros` when identity terminal-hand rotations
are intended. The historical single-file converter overwrites an existing
output NPZ. Use a fresh output path. The batch driver skips existing outputs
unless `--overwrite` is explicitly supplied. Its manifest may contain local
input/output paths and is not intended for publication without review.

## Layout and approximation

Accepted native layouts are:

- `(T, 315)`: translation 3 plus 52 joint rotations in 6D representation.
- `(T, 319)`: discard the first four derived-prefix dimensions, then use the
  315-dimensional layout.
- `(T, 159)`: translation 3 plus 52 axis-angle rotations.
- A leading singleton in `(1, T, C)` is squeezed.
- `C == 69` is explicitly rejected. Its appearance in the original generic
  shape-check message does not mean that this native conversion is supported.
  Other dimensions and non-2D arrays after the optional squeeze are rejected.

The 6D Gram-Schmidt basis is stacked as **rows**, matching the
[PyTorch3D rotation convention](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py).
The converter makes no further axis swap or global coordinate rotation.
It forwards translation values as float32, without unit scaling; the batch
documentation assumes metres and Y-up. `--add-height-y` defaults to zero and
adds only an explicit constant Y offset. The upstream renderer's `+1.3` display
offset is not automatically applied to reconstructed motions.

The axis-angle layout is root 3 + body 63 + left hand 45 + right hand 45.
The output `poses` has 24 joints: root, 21 body joints, then two terminal SMPL
hand joints. Wrists already occur in the body block.

`zeros` gives the two terminal joints identity rotations. `index1` uses the
first rotation from each hand block (`66:69` and `111:114` in the 156-dimensional
axis-angle array), following the SMPL visualization approximation in the
[FineDance renderer](https://github.com/li-ronghui/FineDance/blob/main/render.py).
This is a joint-selection approximation, not a fitted SMPLH-to-SMPL surface
transfer. It does not preserve or reconstruct articulated finger motion.

## Output and frame rate

The native NPZ contains only:

- `poses`: `(T, 24, 3)` float32 axis-angle in radians.
- `trans`: `(T, 3)` float32 translation.

**No FPS is stored and no resampling is performed.** If matching WAV and label
files are available, an estimated rate is returned or printed; it is not written
to the NPZ. The later historical FineMotion importer assumed FineDance was
30 Hz, used factor 1, and normally wrote `mocap_framerate=30`. That import-stage
assumption must be recorded explicitly; the native NPY/NPZ layout alone cannot
establish its timebase. Continue with the [source reconstruction guide](../../docs/SOURCE_RECONSTRUCTION.md)
for unified caches and the separate 20 Hz training representation.

## Historical cache limits

Read-only inspection of three existing cached sequences found nonzero terminal
hand rotations, inconsistent with the `zeros` policy and compatible with
`index1`. No native input/run manifest was located to prove the actual hand flag
used for those files. The default value of a script is not evidence of an old
run's options.

Sequence `001` is an unresolved historical exception: all pose axis-angle
components in its current unified cache are exact negatives of the raw SMPL
cache, while translation is unchanged. This affects local joints as well as the
root, so it is not a root-only coordinate correction. The current `001` cache
also lacks an FPS key. The original cause and repair command were not recovered.
These scripts provide no automatic `001` or whole-corpus sign patch.

CPU synthetic checks cover identity, a positive 90-degree X rotation encoded
as 6D rows, hand policies, height zero, accepted/rejected shapes, overwrite
behavior and CLI help. They do not establish byte-exact reconstruction of the
historical 203 cached sequences. The original custom matrix-to-axis-angle
helper, including its near-pi branch, is retained without a numerical rewrite;
the smoke checks are not an exhaustive near-pi accuracy guarantee.

## Attribution

The project implementation uses NumPy. Its layout and hand approximation refer
to [FineDance](https://github.com/li-ronghui/FineDance), while its rotation
convention follows the [PyTorch3D reference](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py)
for [Zhou et al., CVPR 2019](https://arxiv.org/abs/1812.07035). PyTorch3D's own
implementation retains its [upstream BSD-style licence](https://github.com/facebookresearch/pytorch3d/blob/main/LICENSE).
No upstream source file is bundled or relicensed here. Retain the corresponding
upstream notices if using a separately copied implementation. Data and body-model
terms remain separate from project code terms.
