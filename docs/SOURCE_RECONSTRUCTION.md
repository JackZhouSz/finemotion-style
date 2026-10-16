# Reconstructing motions and training texts

The Bandai [BVH-to-SMPL example](BVH_TO_SMPL.md) demonstrates the conversion with source files, settings and output checks. The text tools below work with descriptions from any source. Motion acquisition and preprocessing remain source specific; obtain each upstream dataset and any SMPL model assets under its own terms.

```text
User-acquired source motion
    → source-specific conversion / retargeting
    → SMPL24 poses + translation
    → 20 Hz joint trajectories and HumanML3D 263D feature clips
                                           │
Source labels + rendered clip → MAIN / DETAILS descriptions
                                           │
                              lemma + POS tokenization
                                           │
                    <clip_id>.txt paired with <clip_id>.npy
```

Sampling rate is a processing choice. The archived conversion intermediates commonly use 30 Hz; this is not a requirement of the method. The training features in this release use 20 Hz. Changing a rate requires consistent feature extraction, time tags and clip mappings, rather than renaming files.

## Descriptions, lemmas and POS tags

The four newly captioned sources are Bandai (2,676 clips), LaFAN1 (3,980), Motorica (5,423) and FineDance (6,644), totaling 18,723. Historical VideoLLaMA3 processing combined an upstream-label-derived MAIN anchor with a visually generated DETAILS sentence. MAIN and DETAILS were written as two alternative descriptions of the same clip, not two consecutive time intervals. HumanML3D and Motion-X descriptions follow their own inherited/adapted annotation routes; the four-source video job did not generate every caption in the integrated collection.

The historical label-context builder used Bandai action/style filenames, LaFAN1 sequence themes, Motorica dance-style codes and FineDance label JSON. A sequence theme can be broader than a cropped action. These anchors preserve source context; they are not proof that a sentence is visually accurate for every segment.

The training format is:

```text
caption#lemma/POS lemma/POS ...#start_time#end_time
```

`VERB`, `NOUN`, `ADJ` and other POS values describe the English words. The script annotates existing words and lemmatizes them; it does not add invented action words. For example, an inflected verb can become `wave/VERB`. Style benchmark families such as Balance or Shake are a separate caption-based classification step; they are not POS tags. See [TEXT_FORMAT.md](TEXT_FORMAT.md) for the exact two tokenization profiles and loader limits, and [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) for classification.

From the repository root, convert a historical VLM JSONL file containing `id`, `caption_main` and optional `caption_details`:

```bash
python scripts/text/caption_fields_to_jsonl.py \
  --input-jsonl /path/to/captions.labelaware.merged.jsonl \
  --output-jsonl /path/to/new_release/captions_for_formatter.jsonl

python -m pip install -r scripts/requirements-text.txt
python scripts/prepare_training_texts.py \
  --input-jsonl /path/to/new_release/captions_for_formatter.jsonl \
  --output-dir /path/to/new_release/texts \
  --profile legacy_finemotion

python scripts/prepare_training_texts.py \
  --input-dir /path/to/new_release/texts --validate-only
```

The adapter preserves caption strings, clip IDs and any supplied `tokens_main` / `tokens_details` fields. It records an input hash and a line-to-field manifest. Missing time tags become `0.0#0.0`, the historical whole-clip convention; supplied `from` / `to` values must be valid. Missing MAIN, malformed role markers, model error rows, duplicate input IDs and existing outputs cause an error. Historical jobs sometimes appended retries: `--duplicates latest` explicitly selects the last row per ID and records that selection. `--main-only` deliberately omits DETAILS. No model is called and no source file is edited.

For existing HumanML3D-format TXT files, use the formatter directly:

```bash
python scripts/prepare_training_texts.py \
  --input-dir /path/to/source/texts \
  --output-dir /path/to/new_release/texts \
  --profile legacy_finemotion
```

Its default preserves existing tokens and only fills missing/placeholder tokens. `--retokenize` explicitly regenerates all token fields. Use `humanml_reference` only when intentionally choosing the reference HumanML3D rule; the profiles differ. The supplied dependency versions describe a verified release environment, not a recovered environment for every historical export. Regenerated tokens can therefore differ from the archived training tokens. Keep the archived text and its ID mappings for exact historical use.

The author-confirmed eight paper captions belong in a versioned paper-aligned copy, with the literal strings and changed-line hashes recorded. A formatting pass does not apply paper overrides, resolve semantic mistakes or change frozen benchmark memberships automatically.

## Historical text implementation

[`scripts/text/historical/`](../scripts/text/historical/) contains six unchanged snapshots and one explicitly adapted label-context builder, with original/public SHA-256 hashes in `manifest.json`:

- `vlm_build_label_context_v4.py`: source labels → MAIN anchors and prompt context. The public adaptation removes the embedded LaFAN1 README descriptions and requires user-local metadata for LaFAN1.
- `vlm_caption_videos_videollama3.py`: historical VideoLLaMA3 caption generation; `--main_mode anchor_only` selects the source anchor as MAIN.
- `vlm_jsonl_to_humanml_texts.py`: historical MAIN/DETAILS cleanup, tokenization and TXT write-back.
- `tokenize_humanml_texts_spacy.py`: historical placeholder-token upgrade.
- `build_splits_sem_v3_1_from_263_texts.py` and `semantic_subset_config_v3.py`: style and attribute views.
- `build_finemotion_content_lists_from_texts.py`: later all-source content views.

The original video pipeline configuration recorded `VideoLLaMA3-7B`, 512×384 input, 12 sampled frames/s, at most 360 frames and 128 generated tokens; the rendered videos were 20 Hz. Those are historical settings, not a requirement for using the released texts. The 2026 Qwen audit is a separate review of descriptions and is not the original caption producer.

For LaFAN1, acquire the upstream dataset metadata separately and prepare a local JSON object mapping normalized filename theme keys to their source descriptions. No theme descriptions are bundled here. Then run the adapted context builder:

```bash
python scripts/text/historical/vlm_build_label_context_v4.py \
  --videos_dir /path/to/my_rendered_clips \
  --lafan-theme-map /path/to/my_local_lafan_theme_descriptions.json \
  --out_jsonl /path/to/new_release/label_context.jsonl
```

The selected LaFAN1 inputs must all have an entry in the supplied map. Missing maps or themes stop the command before output creation; the script does not invent a generic fallback. The map's SHA-256 is recorded in each LaFAN1 context record. The source metadata, and anchors derived from it, retain upstream provenance and terms. Project software's MIT notice does not relicense LaFAN1 dataset metadata. Bandai filename labels and Motorica style-code names provide factual source context, while FineDance labels are read from the user's local files; this does not make every MAIN sentence newly authored project prose.

Historical scripts retain old path/dependency assumptions and some write in place or overwrite outputs. Use the public adapter and formatter for new exports. The six unchanged snapshots preserve the original implementation; the one metadata-loading adaptation is documented separately.

## Motion routes by source

The recovered motion conversion and shared feature scripts are now published in [`scripts/reconstruction/historical/`](../scripts/reconstruction/historical/), with original/public names and hashes in its manifest. Read the [historical-code setup and dependencies](../scripts/reconstruction/README.md) before running them: the original workspace layout, path constants and write behavior are retained. The common 263 builder has the same user-local LaFAN metadata adaptation as the public context builder; it does not bundle the source theme prose.

**Bandai.** Obtain the [original BVH](https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset). The [worked example](BVH_TO_SMPL.md) covers calibration-pose insertion, MotionBuilder retargeting, SMPL-skeleton BVH export, local axis-angle NPZ conversion and optional resampling. The implementation builds on [tempo-changing-music2motion](https://github.com/dongran/tempo-changing-music2motion); retain its attribution. The distributed Bandai archive records the transformations and Bandai notices separately from SMPL model assets.

**LaFAN1.** Obtain [LaFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) BVH. [`BVHtoSmpl_LaFAN1.py`](../scripts/reconstruction/historical/BVHtoSmpl_LaFAN1.py) (public filename for the unchanged historical retargeter) and [`step3_finalize_npz_LaFAN1.py`](../scripts/reconstruction/historical/step3_finalize_npz_LaFAN1.py) use the same general MotionBuilder route with a different source skeleton mapping; the finalizer uses translation scale `0.01`. Bandai's mapping must not be applied unchanged to LaFAN1. A portable, independently rerun LaFAN1 example is still pending.

**HumanML3D / AMASS.** Obtain [AMASS](https://amass.is.tue.mpg.de/) motion parameters and [HumanML3D](https://github.com/EricGuo5513/HumanML3D) `index.csv` and text mappings. `convert_humanml3d_amass_npz_to_smpl_npz.py` extracts root + 21 body joints from SMPL-H and pads the two SMPL terminal joints. `export_humanml3d_clips_30hz_to_finemotion.py` maps IDs, resamples to the 20 Hz index timeline, crops `[start,end)` and exports the archived intermediate rate. HumanML3D joint-position `pose_data` is not an axis-angle input. Preserve missing-source / skipped-ID manifests. The historical root convention is encoded by the cache name (`Rx -90`, `Ry 180`), but the exact original command is not fully recorded; a clean-checkout motion reproduction claim requires resolving this gap.

**Motion-X.** Obtain [Motion-X](https://github.com/IDEA-Research/Motion-X) motion and annotations. `convert_motionx_smplx322_to_smpl24_npz.py` reads root `:3`, body `3:66`, translation `309:312` and shape `312:` from packed 322D arrays, then pads the terminal joints. This is a body-parameter extraction, not a fitted full SMPL-X-to-SMPL mesh conversion. `select_motionx_clean_subset.py` applies the historical clean-selection rules. Use the saved selection manifest rather than promising the same IDs from a new environment-dependent axis scan.

**FineDance.** Obtain [FineDance](https://github.com/li-ronghui/FineDance). The recovered `add_motorica_finedance_to_finemotion.py` starts from an existing SMPL24 cache and standardizes it. The earlier native-motion-to-SMPL24 adapter has not been recovered or rerun. This is a documented reconstruction gap; the cache-standardization script alone is not an end-to-end converter.

**Motorica.** Obtain [Motorica Dance Dataset](https://github.com/simonalexanderson/MotoricaDanceDataset) BVH. The same cache-standardization helper starts from retargeted SMPL24 parameters; its historical 120→30 Hz step is recorded. The source-specific MotionBuilder mapping and original BVH-to-SMPL entrypoint have not been recovered. The generic Bandai example explains the method but does not establish Motorica-specific settings.

## Shared features and mappings

The common feature scripts are [`build_finemotion_humanml3d263_dataset.py`](../scripts/reconstruction/historical/build_finemotion_humanml3d263_dataset.py) and [`convert_finemotion_smpl_npz_to_humanml3d_vecs.py`](../scripts/reconstruction/historical/convert_finemotion_smpl_npz_to_humanml3d_vecs.py). SMPL forward generates 22 joint trajectories, which are resampled and canonicalized using HumanML3D/MCM-LDM motion processing. The historical defaults were 197 joint frames (196 feature frames), stride 80 and a minimum of 41 joint frames. SMPL model weights and the motion-processing implementation must be acquired separately under their own terms. When including LaFAN1, the adapted public builder requires `--lafan-theme-map` and records its hash; the motion feature and split algorithms remain the archived implementation.

Keep source filename, parent recording, crop start/end, sampling rate, coordinate convention, clip ID and hashes in a reconstruction ledger. A converted sequence can yield multiple training windows; a source-sequence count and a training-clip count are different quantities. Preserve exact filenames, including punctuation and actual mirrored `M` IDs. Do not synthesize mirrored text by dropping the prefix.

These materials provide one complete Bandai example, portable text preparation and frozen benchmark export. The outstanding source-specific motion gaps above remain explicit. Code availability and conversion do not grant new redistribution rights over upstream data or annotations; retain each source's notices and terms.
