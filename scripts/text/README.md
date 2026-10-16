# Text and benchmark tools

Run commands from the repository root. The portable entrypoints write to new destinations and never call a VLM:

- `caption_fields_to_jsonl.py`: historical `caption_main` / `caption_details` → formatter JSONL, with literal captions and a provenance manifest.
- `../prepare_training_texts.py`: English lemma/POS annotations and HumanML3D TXT format. Existing tokens are preserved unless explicitly regenerated.
- `classify_caption_subsets.py`: run archived style/content rules into a new version directory without changing source texts or frozen lists.
- `export_frozen_benchmark.py`: copy frozen lists byte for byte, derive assigned content views, and report availability/current caption matches without changing historical memberships.
- `semantic_subset_config_v3.py`: exact archived four-family taxonomy, attributes and scoring configuration. Its hash is checked by the frozen exporter.
- `prepare_compatibility_text_view.py`: hash-bound quarantines and optional historical NaN-time normalization into a new directory, preserving original affected files and exact change ledgers. See the [release guide](../../docs/RELEASE_V1.md) for the explicit flags and known limits.

See [source reconstruction](../../docs/SOURCE_RECONSTRUCTION.md), [text format](../../docs/TEXT_FORMAT.md) and [benchmark protocol](../../docs/BENCHMARK_PROTOCOL.md) for commands and scope. The exporter and taxonomy use the Python standard library. The formatter needs spaCy only when generating tokens.

`historical/` stores six unchanged project snapshots and one explicitly adapted label-context builder. The checksum manifest distinguishes them and records original/public hashes. The public `vlm_build_label_context_v4.py` removes the embedded LaFAN1 README prose: LaFAN1 inputs require `--lafan-theme-map /path/to/user_local_map.json`, an object mapping normalized filename theme keys to source descriptions acquired by the user. Missing maps/themes cause an error before output creation. Upstream metadata is not bundled or relicensed as MIT project prose; generated MAIN anchors retain its provenance.

These files are implementation evidence with old path and dependency assumptions. In particular, the original TXT exporter/tokenizer writes in place and the original subset builders can overwrite outputs. Use a disposable copy with explicit paths when inspecting those behaviors; use the portable entrypoints for release preparation. VideoLLaMA3, spaCy, HumanML3D processing, source dataset metadata and SMPL assets retain their own terms and are not bundled or relicensed here.
