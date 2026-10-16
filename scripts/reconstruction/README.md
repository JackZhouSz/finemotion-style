# Historical motion reconstruction code

This directory publishes the recoverable project conversion scripts. It contains code and a hash manifest, **no source motion files, SMPL weights, target FBX/template or third-party implementation tree**. Obtain each source and dependency under its own terms. The verified portable [Bandai example](../../docs/BVH_TO_SMPL.md) remains the worked conversion example.

[`historical/manifest.json`](historical/manifest.json) records original filenames, source hashes, public filenames and any adaptation. Seven files preserve their original bytes. The LaFAN1 retargeter has English comment localization. The common 263 builder has a documented metadata adaptation: the embedded LaFAN1 theme prose is removed and loaded from a user-local map instead. Original private snapshots remain unchanged.

## Recovered entrypoints

- [`BVHtoSmpl_LaFAN1.py`](historical/BVHtoSmpl_LaFAN1.py): LaFAN1-specific source HIK mapping, characterization and retargeting inside Autodesk MotionBuilder. The public filename and English comment localization differ from the historical entrypoint; the operational AST is preserved. It requires prepared BVH with a first calibration pose and a separately obtained characterized SMPL FBX.
- [`step3_finalize_npz_LaFAN1.py`](historical/step3_finalize_npz_LaFAN1.py): standardize the retargeted SMPL-skeleton BVH and export local axis-angle NPZ using the shared 2025 conversion implementation. Translation scale is `0.01`.
- [`convert_humanml3d_amass_npz_to_smpl_npz.py`](historical/convert_humanml3d_amass_npz_to_smpl_npz.py): AMASS SMPL-H parameters to body22, padded SMPL24 or full52 output; explicitly select `--export_mode smpl24` for this route.
- [`export_humanml3d_clips_30hz_to_finemotion.py`](historical/export_humanml3d_clips_30hz_to_finemotion.py): use HumanML3D `index.csv` to resample/crop converted AMASS sequences and retain clip IDs, with a skip-list option for missing mappings.
- [`convert_motionx_smplx322_to_smpl24_npz.py`](historical/convert_motionx_smplx322_to_smpl24_npz.py): extract body rotations, translation and shape from packed Motion-X parameters; optional SMPL-based axis detection. Parameter layout is a factual source-format mapping, not bundled Motion-X motion/annotation data.
- [`select_motionx_clean_subset.py`](historical/select_motionx_clean_subset.py): finite-value, axis, translation and group filtering; writes selected relative paths and a report, with copying enabled only by `--copy`.
- [`add_motorica_finedance_to_finemotion.py`](historical/add_motorica_finedance_to_finemotion.py): standardize existing SMPL24 caches and reproduce their historical intermediate-rate handling. The earlier stages are documented separately in the [native FineDance converter](../finedance/README.md) and [Motorica source-character walkthrough](../../docs/MOTORICA_TO_SMPL.md).
- [`build_finemotion_humanml3d263_dataset.py`](historical/build_finemotion_humanml3d263_dataset.py): SMPL forward, 20Hz joints, feature windows, preliminary texts, original ID-based splits and normalization statistics. LaFAN1 inclusion now requires `--lafan-theme-map`.
- [`convert_finemotion_smpl_npz_to_humanml3d_vecs.py`](historical/convert_finemotion_smpl_npz_to_humanml3d_vecs.py): shared SMPL/joint/263 utilities and an individual-file converter, with optional joint previews.

## Restore the expected layout before using historical scripts

The files were originally located in a SOTA processing workspace. Several compute their workspace root as `Path(__file__).parents[1]`, import from `tools`, or search for a neighboring conversion repository. **Their defaults do not become portable simply because they are published here.** Absolute Linux/Windows paths embedded in the historical scripts are local-path placeholders, not download locations. Override relevant path arguments, or copy the selected scripts into a separate working tree with this layout:

```text
my_processing_workspace/
  tools/
    build_finemotion_humanml3d263_dataset.py
    convert_finemotion_smpl_npz_to_humanml3d_vecs.py
    convert_motionx_smplx322_to_smpl24_npz.py
    select_motionx_clean_subset.py
    ... other selected tools ...
  datasets/
    FineMotion/<source>/*.npz         # user-acquired / locally converted
    raw/<source>/...                 # user-acquired / locally converted
  third_party/
    mcm-ldm-official/                # separately acquired implementation
      mld/data/humanml/...
      datasets/humanml3d/new_joint_vecs/...  # acquired reference data if used
```

`select_motionx_clean_subset.py` imports `tools.convert_motionx_smplx322_to_smpl24_npz` during axis checking, so keep that converter in the same restored `tools` package. The 263 builder imports its shared converter from the same directory. The shared converter imports HumanML motion processing from the separately acquired MCM-LDM tree. Reuse the matching reference skeleton/offset configuration: its fallback offsets are not a guarantee of historical feature equality.

The two LaFAN1 MotionBuilder/finalizer files instead expect a dedicated `LaFAN1_to_smpl/` working directory, prepared `bvhForC/`, and the original relative layout around `tempo-changing-music2motion`. Inspect their constants and shared-helper paths before launching MotionBuilder. Do not run either in the published source directory. Preparation of the LaFAN1-specific calibration input and licensed target assets remains a separate step; the Bandai skeleton mapping cannot be substituted unchanged.

For argument discovery in an environment with NumPy, these commands do not process motions:

```bash
python scripts/reconstruction/historical/convert_humanml3d_amass_npz_to_smpl_npz.py convert --help
python scripts/reconstruction/historical/export_humanml3d_clips_30hz_to_finemotion.py --help
python scripts/reconstruction/historical/convert_motionx_smplx322_to_smpl24_npz.py convert --help
python scripts/reconstruction/historical/select_motionx_clean_subset.py --help
python scripts/reconstruction/historical/build_finemotion_humanml3d263_dataset.py --help
```

A parameter-only conversion can use explicit input/output roots. For example, this shows the Motion-X converter interface without invoking automatic axis fitting:

```bash
python scripts/reconstruction/historical/convert_motionx_smplx322_to_smpl24_npz.py convert \
  --in_root /path/to/acquired/smplx_322 \
  --out_root /path/to/new_processing/Motion-X_smpl24 \
  --trans_scale 1.0 --global_rx_deg 0 --global_ry_deg 0 --write_meta
```

This is a conversion invocation, not a claim that rerunning it selects the exact archived clean subset or matches every coordinate convention. Preserve the saved selection and coordinate manifests described in [source reconstruction](../../docs/SOURCE_RECONSTRUCTION.md).

## Dependencies and upstream attribution

Parameter conversion and cache standardization use Python and NumPy. SMPL forward, optional axis fitting and 263 extraction additionally need PyTorch, `smplx`, a separately obtained SMPL neutral model, and the external MCM-LDM/HumanML motion-processing dependencies. Optional previews use Matplotlib and system FFmpeg. The LaFAN1 retargeter requires MotionBuilder/`pyfbsdk`; its finalizer imports the `dataset.python.smpl_bvh_to_smpl_npz` helper from [tempo-changing-music2motion](https://github.com/dongran/tempo-changing-music2motion).

These are recovered FineMotion project scripts from the author's processing workspace. They are not copies of the complete upstream dataset implementations. Retain the project [MIT notice](../../LICENSE), the [2025 conversion-code notice](../../licenses/TEMPO_CODE_LICENSE.txt), the [HumanML3D notice](../../licenses/HUMANML3D_REPOSITORY_LICENSE.txt) where its reference processing is used, and the [source/asset notices](../../NOTICE.md). External packages, their source code and model assets retain their own terms; imports do not relicense them. A fresh end-to-end conversion was not performed for these historical files during release packaging, and the exact dependency environment for every original run has not been recovered.

## LaFAN1 metadata adaptation

The common builder no longer embeds the upstream README's 15 source-theme descriptions. If `--include` contains `lafan1_30hz`, supply `--lafan-theme-map /path/to/my_local_theme_map.json`: a JSON object mapping normalized lowercase filename theme keys to descriptions acquired from the user's upstream metadata. Every selected LaFAN1 NPZ filename must have a map entry. Missing maps, missing themes and malformed metadata stop before feature output creation or SMPL loading; no generic fallback is invented. The local metadata SHA-256 is saved in the output `meta.json`.

The user-local metadata and preliminary captions derived from it retain source provenance and terms. Publishing the independent code does not make source theme prose MIT project content. The later label-aware VLM builder has the same [explicit local-metadata requirement](../../docs/SOURCE_RECONSTRUCTION.md#historical-text-implementation).

## Original write behavior and remaining gaps

Use a new, separate processing workspace and explicit output roots. These historical implementations are preserved for inspection and adaptation, not supplied with the new-directory safeguards of the public text wrappers:

- The LaFAN1 retargeter defaults to `CLEAR_OUTPUT_DIR = True` and deletes existing output BVH files before processing. Inspect/change that constant for your working copy.
- Parameter converters usually skip existing outputs unless `--overwrite`; finalizers, split/report files, features and normalization arrays can overwrite existing files. Some writers replace temporary NPZ files atomically.
- The common builder regenerates segment-ID splits and computes Mean/Std over its built feature collection. These are the original producer behaviors. Do not use a new run to overwrite frozen published splits or reinterpret it as a new recording-grouped protocol.
- Preliminary builder captions can use `unk/OTHER` or filename-derived fallbacks for sources other than the explicitly required LaFAN metadata. They are not the final archived MAIN/DETAILS descriptions. Apply the separate [text pipeline](../../docs/TEXT_FORMAT.md), preserving inherited text fields and provenance.
- The native FineDance route and Motorica source-character procedure are documented in the current [source guide](../../docs/SOURCE_RECONSTRUCTION.md). Fresh original-Motorica retarget verification, complete FineDance cache reconciliation, the exact historical AMASS coordinate command and complete availability manifests remain outstanding. Publishing recovered code does not authorize redistribution of source motions.

Only AST parsing, source/hash inspection, CLI argument discovery and the LaFAN metadata preflight were verified for this historical code package. No MotionBuilder application, SMPL forward pass, source conversion, model inference or full dataset rebuild was run as part of packaging it.
