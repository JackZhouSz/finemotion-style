# Caption-derived benchmarks and frozen splits

The training corpus, strict style benchmark and qualitative paper examples have different roles. The saved pose69-valid corpus contains **59,038** segments: **47,236 train / 5,866 validation / 5,936 test**. General model training uses the complete training split across sources, including clips outside the four style families. Validation and test clips are separate. The style benchmark does not limit training to its labeled subset.

The paper reports 59,039 normalized clips. The archived available-ID and partition files contain 59,038; this historical one-clip difference remains unresolved. The release preserves the saved IDs and does not add a replacement clip to match the reported number.

## How the four style families are selected

The exact rules are in [`semantic_subset_config_v3.py`](../scripts/text/semantic_subset_config_v3.py), with the unchanged historical builder in [`scripts/text/historical/`](../scripts/text/historical/).

1. Read all nonempty text lines for a clip. Take the caption before the first `#`; both MAIN and DETAILS participate when present.
2. Join captions using ` | `, lowercase and collapse whitespace. POS token fields are not classification inputs.
3. Apply the four case-insensitive regex groups: `impact`, `strike`, `balance`, `tremor`.
4. `single_strict` retains a clip only when exactly one family matches. No-hit and multiple-family clips are excluded. `multi` retains every matched family; `single_argmax` is a separate permissive view using the archived scoring and fixed-order tie break.
5. Keep the existing clip-ID train/val/test assignment. Style classification does not randomly repartition clips.

Storage class `tremor` is displayed as **Shake** in the paper. Class order is **Impact / Strike / Balance / Shake**, stored as `impact / strike / balance / tremor`, labels `0 / 1 / 2 / 3`.

Impact includes contact, landing, falling, getting up and the historical jump/hop expansion. Strike includes punching, kicking and fighting. Balance includes recovery, stumbling, wobbling and swaying. Shake includes trembling, shivering, jitter and related terms. The literal regex configuration defines membership; these summaries are not replacement rules.

These labels are caption-derived heuristics. A word hit is not a manual verification of the movement's kinematic mechanism. Broad phrases and hallucinated DETAILS can affect class membership. For this reason the release retains the frozen list version even if a later caption is corrected.

## Paper and availability views

`paper_style_20260301_v1` identifies the March 1 lists corresponding to the supplementary style table, derived from the **67,907-ID** 263-feature/caption pool:

- Impact: **4,196** — train 3,339 / val 418 / test 439.
- Strike: **3,809** — train 3,039 / val 383 / test 387.
- Balance: **4,726** — train 3,743 / val 493 / test 490.
- Shake: **626** — train 506 / val 67 / test 53.

Total: **13,357**, partitioned **10,627 / 1,361 / 1,369**. This caption pool is larger than the pose69-valid training-availability corpus. Intersecting the saved March lists with the saved pose69 keep IDs gives **11,591** strict style clips: Impact 3,608 / Strike 2,892 / Balance 4,533 / Shake 558. The corresponding partitions total **9,222 / 1,185 / 1,184**. Preserve this as a separately identified availability view.

The 67,907-ID base feature pool itself is partitioned **54,360 / 6,749 / 6,798**. Its counts, the 59,038 available IDs, the 13,357 paper labels and the 11,591 available style IDs should not be used interchangeably.

## Attributes and content classes

`periodic` and `secondary_rich` are independent attributes, not additional style families. The March paper lists contain **19,071 periodic** and **35,332 secondary-rich** IDs; after pose69 filtering, **17,913** and **33,323**. The rules are broad: walking/running words can trigger periodic, while `with` or `then` can trigger secondary-rich. Keep this heuristic interpretation when describing the benchmark.

The supplementary content table corresponds to an older **January 7** HumanML3D subset snapshot, `paper_content_20260107_v1`:

- Walk: **4,121 assigned** — train 3,308 / val 188 / test 625.
- Run: **616** — 508 / 26 / 82.
- Turn: **2,750** — 2,171 / 138 / 441.
- Circle: **2,021** — 1,644 / 104 / 273.

The assigned union has **9,508** IDs. The original Walk `all.txt` additionally contains `009707` and `M009707` without a saved partition, making the original all-list union **9,510**. Preserve the originals, provide the saved assigned union separately, and leave those two IDs unassigned.

The later all-source content builder uses priority **circle > turn > run > walk**, with optional strong-style exclusion enabled in the inspected March report. Its exclusion regex differs from the full style taxonomy, so some content/style overlap is possible. Rerunning this newer builder is useful for a new version; it does not reproduce the older January table by definition.

## Generate a new classification view

For user-created captions, arrange `texts/<clip_id>.txt` and frozen `splits/{all,train,val,test}.txt` under a dataset root, then run:

```bash
python scripts/text/classify_caption_subsets.py \
  --dataset-root /path/to/my_dataset \
  --view style --out-dir /path/to/new_versions/style_from_my_captions

python scripts/text/classify_caption_subsets.py \
  --dataset-root /path/to/my_dataset \
  --view content --out-dir /path/to/new_versions/content_from_my_captions
```

The wrapper runs the unchanged historical rules, checks the implementation hashes, requires a new output directory outside the input/code trees, checks disjoint base partitions and disables bytecode writes. Use `--no-style-exclude` only when intentionally defining content without that exclusion. Outputs describe a new caption snapshot; they do not replace frozen paper lists. Retain the exact text files and their checksums alongside the generated reports.

## Export frozen memberships and caption evidence

When preparing the historical release from acquired text and list snapshots, use the conservative exporter:

```bash
python scripts/text/export_frozen_benchmark.py \
  --semantic-root /path/to/march_splits_sem_v3-1 \
  --content-root /path/to/january_hfcontent_v0_4class \
  --base-splits /path/to/base263/splits \
  --training-splits /path/to/splits_pose69_aligned_HF_ok \
  --texts-dir /path/to/base263/texts \
  --taxonomy-config scripts/text/semantic_subset_config_v3.py \
  --keep-ids /path/to/keep_ids_pose69_aligned_HF.txt \
  --release-id paper_benchmark_v1 \
  --out-dir /path/to/new_release/benchmark_paper_v1
```

The exporter copies frozen class and partition lists byte for byte, derives the January assigned-all views, records input/output hashes and audits the current captions against the taxonomy. `caption_membership_audit.jsonl` includes clip ID, source, base and pose69 split, availability flag, caption strings and full HumanML lines, text hash, frozen memberships, matched pattern IDs/counts, current strict class or exclusion, and historical/current disagreement. It does not copy motion files, change source texts or change memberships from the current regex matches. It reports pose69 availability per row; supply the saved filtered semantic root to export the separate availability-filtered lists.

An optional `--paper-examples /path/to/paper_examples.json` copies the eight-example manifest separately. It does not apply caption overrides or add the examples to test lists. Some of the eight paper examples are train/validation clips or outside the pose69-valid set; they are qualitative demonstrations rather than eight held-out evaluation inputs.

## Split and evaluation limits

The original feature builder assigns segment IDs using an MD5-derived value of `1234:<clip_id>` and thresholds 0.8 / 0.1 / 0.1. It partitions segments rather than parent recordings. Overlapping windows of one recording can therefore occur in different partitions. A future recording-grouped split must have a new protocol version; keep the historical partitions for the existing released model.

Classifier training exports, style-reference selection and transferred-motion evaluation apply additional availability and sampling steps. Their counts are distinct from the caption taxonomy counts. Example released weights and demonstration settings are not a promise that every saved checkpoint reproduces the final paper metric table. Exact quantitative reproduction requires the relevant selected-ID/pair manifests, evaluator checkpoint and metric settings in addition to these dataset lists.

Caption revisions should ship as a separate text version with accepted old/new fields and hashes. Keep the original text and frozen benchmark lists available. Rerun classification only for an explicitly named new benchmark version.
