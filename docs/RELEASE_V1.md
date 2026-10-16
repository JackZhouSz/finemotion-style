# Bandai release and six-source reconstruction guide

The motion conversion example, text preparation code and benchmark tools are
available in this repository. The frozen-ID protocol archive is included
[directly on GitHub](../protocols/README.md). The two larger Bandai motion/text
and BVH companion archives are hosted in the
[Hugging Face Dataset](https://huggingface.co/datasets/randong/FineMotion-Style).
Other sources are acquired from their official providers and processed locally.
A package's SHA256 and repository revision identify its exact version. See the
[dataset card](../huggingface/README.md).

## Selected first-release components

- **Bandai original v1**: 3,077 converted SMPL24 motion sequences, 2,676 paired
  20Hz HumanML263 features and historical text files, frozen splits, source
  mapping and the BVH-to-SMPL example. Source sequence counts and paired training
  window counts differ. Motion and annotation terms are CC BY-NC 4.0. Full SMPL
  model weights and standalone target templates are obtained separately.
  [Download the motion/features/text archive from Hugging Face](https://huggingface.co/datasets/randong/FineMotion-Style/resolve/main/finemotion_bandai_original_v1_20261016_en.tar.gz?download=true).
- **Frozen protocol IDs**: full historical split and benchmark ID lists, with
  original bytes/order and a hash manifest. These contain no source captions or
  motions. They describe membership in the six-source protocol; they do not
  supply the missing source files or a complete source-to-crop conversion ledger.
  The 292-list archive is available in this repository at
  `protocols/finemotion_frozen_protocol_v1_20261016.tar.gz`
  ([direct download](https://github.com/dongran/finemotion-style/raw/refs/heads/main/protocols/finemotion_frozen_protocol_v1_20261016.tar.gz)).
  Its size is 7,944,168 bytes and SHA256 is
  `3209df832ac2729e30c1544bf1eb56e954179f4033efeb71a7588b27b4bd1a49`.
  See the [protocol package guide](../protocols/README.md) for checksum verification.
- **Bandai BVH companion**: 3,077 original source BVH files and 3,077 matching
  SMPL-derived BVH animation files, with source/sequence/clip mapping and hashes.
  The derived view is a later SMPL-cache export, not the historical MotionBuilder
  output stage. See the [BVH companion guide](BANDAI_BVH_COMPANION.md).
  [Download the BVH companion from Hugging Face](https://huggingface.co/datasets/randong/FineMotion-Style/resolve/main/finemotion_bandai_bvh_companion_v1_20261016_en.tar.gz?download=true).

## Retained staging alternatives

The following text archive was prepared and checked before selecting the
Bandai-first hosting plan. It remains preserved locally and is **not selected
for the first Hugging Face upload**:

- **Three-source original text v1**: HumanML3D 13,186 files, Motion-X 35,998 and
  Bandai 2,676, totalling 51,860. The default text view uses the selected
  paper-exact wording; inherited annotations and source-specific licences are
  identified separately. It contains no motion, video or body-model assets.
- **Derived compatibility view**: an explicitly generated experimental copy
  that quarantines confirmed non-action outputs/empty annotation intervals and
  optionally normalizes NaN times as the historical loader did. This is a new
  text/availability view, not a replacement for the saved paper protocol.

The complete six-source technical snapshot has 67,907 texts. The prepared
three-source archive is a subset, not a claim that the full training pool had
only 51,860 IDs. The complete paper-exact view has eight selected paper
sentences; the three-source download contains seven corresponding full clip
files. The Motorica clip is outside that download scope. The author-locked
eight-example reference identifies that distinction. The selected Bandai-only
motion/text package contains one of the eight corresponding clip files. The
eight-example verification JSON documents the previously prepared three-source
artifact and is not an inventory of the Bandai-only downloads.

## Historical annotations and the Qwen screen

Original annotations contain source-label-derived MAIN descriptions and
VideoLLaMA3 DETAILS for the four newly annotated sources; HumanML3D and Motion-X
sentences are inherited. The new Qwen screen covered 18,723 old rendered videos
from Bandai, LAFAN1, Motorica and FineDance, using timestamped 16-frame sheets.
It did not visually audit all 59,038 available IDs or the inherited Motion-X
and HumanML3D corpora.

A K result means no clear major error was flagged; U is unresolved; R is an
unaccepted correction proposal. Old views lack a fixed world reference and
fully articulated fingers/props. Missing visible detail is not a contradiction.
Do not apply the Qwen proposals as a bulk replacement. In a directed independent
check, three proposed Bandai left/right reversals were refuted by SMPL joint
motion; the original side labels were retained. This is not a measured corpus
accuracy or training-quality improvement.

## Known issues in inherited/reconstructed text

The historical text snapshot retains 44 confirmed pure conversational/refusal
or grammar-explanation clip captions, two nonzero empty annotation intervals
and 49 NaN-time lines. A broader 75-item Motion-X keyword scan includes actual
action descriptions and uncertain cases; do not delete all 75 automatically.
The four-source video screen did not cover these inherited Motion-X outputs.

An optional compatible copy can quarantine the 44 clip files pending visual
recaptioning and remove only the two empty-interval lines, retaining the valid
other lines in those two files. Their original text is kept as evidence. No
replacement action is invented. The NaN-to-zero option follows the inspected
loader field by field: 48 lines become full-clip time tags, but one becomes
0–2 seconds. This compatibility operation does not verify intended timing.

One separately inspected HumanML3D example (`001034`, line 3) describes a
backflip/fall within 0–3 seconds, while those events occur later in the current
paired 20Hz representation. Its timing remains unchanged pending provenance
review. See the [bounded timing issue](../release_metadata/temporal_issue_001034.json).
Neither text view is a fully corrected semantic or temporal annotation set.

The 44 clips include 36 saved pose69-available IDs: 29 train, four validation
and three test. None is in the four strict style classes or the locked paper
examples; some occur in secondary-attribute lists. Preserve original IDs and
lists and give any filtered usable view an explicit version and counts.

The repository includes the issue manifests and an explicit converter:

```bash
python scripts/text/prepare_compatibility_text_view.py \
  --source-root /path/to/paper_exact/texts \
  --output-root compatibility_v1 \
  --quarantine-manifest release_metadata/junk_safe_exclusion_manifest.json \
  --format-ledger release_metadata/format_issue_ledger.json \
  --apply-confirmed-quarantine --normalize-nan-times
```

Use a new output root. The converter preflights file/line hashes, retains
quarantined originals and writes exact file/line changes. `--dry-run` verifies
without writing. An intentionally smaller source subset additionally requires
`--allow-source-subset`, which records every absent issue ID. The source-scoped
three-source package contains all affected IDs, so that option is unnecessary.
The converter intentionally refuses symlink/junction paths; resolve an acquired
dataset path to its physical location before passing it as `--source-root`.

## Publication scope and source terms

The selected first release redistributes Bandai motion/text. Users obtain the
other five sources, including original annotations where available, from their
providers. This is a distribution choice, not a claim that HumanML3D or Motion-X
annotations cannot be redistributed under their stated terms.

HumanML3D inherited text retains the repository MIT notice. Motion-X expressly
licenses annotations under CC BY-NC-SA 4.0; adapted token/POS exports retain
that annotation scope and attribution. Bandai motion and source-label-derived
annotations retain CC BY-NC 4.0. See [NOTICE](../NOTICE.md).

The official LAFAN1 motion/database terms are CC BY-NC-ND 4.0. Motorica prohibits
unauthorised redistribution of database content. FineDance restricts sharing
its Data & Software without prior permission. None of these statements alone
establishes that every independently generated factual description is
prohibited. The present MAIN/DETAILS components, however, use source labels
and sometimes source theme prose, so their complete annotation redistribution
scope is not confirmed by the inspected terms alone. Those components are
kept outside the hosted motion/text component. Publishing local processing code
does not expand any upstream permission to use, modify or share the source.

The independent processing code, prompt structure, formatter and benchmark
logic are provided for local use with legitimately acquired source data. The
public LaFAN context adapter reads a user's local theme map rather than
bundling the upstream theme-description table. See [source reconstruction](SOURCE_RECONSTRUCTION.md).

Official terms:

- [HumanML3D](https://github.com/EricGuo5513/HumanML3D/blob/main/LICENSE)
- [Motion-X annotations](https://motion-x-dataset.github.io/static/license/Motion-X%20License.pdf)
- [Bandai](https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset#license)
- [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/license.txt)
- [Motorica](https://github.com/simonalexanderson/MotoricaDanceDataset/blob/main/LICENSE.txt)
- [FineDance](https://github.com/li-ronghui/FineDance/blob/main/LICENSE)

## Reconstructing texts and benchmark lists

1. Obtain source motions and source metadata under each provider's terms.
2. Follow the [source adapters](SOURCE_RECONSTRUCTION.md) to obtain the paired
   motion representation. The recovered native FineDance converter has CPU
   synthetic checks. Motorica uses the author-confirmed zero-rotation calibration
   and MotionBuilder source-character route. Only the Bandai example has a fresh
   end-to-end verification here; method recovery does not imply every old array
   has been reproduced byte for byte.
3. Reproduce the source-label prompt context and video descriptions, or supply
   your own reviewed captions. Use the [MAIN/DETAILS adapter](../scripts/text/caption_fields_to_jsonl.py)
   to export formatter JSONL without changing the sentence.
4. Generate real lemma/POS using the pinned [text formatter](TEXT_FORMAT.md).
   Preserve inherited token fields when reproducing an archived version.
5. Use the [benchmark tools](BENCHMARK_PROTOCOL.md) for either exact frozen
   list export or an explicitly named new caption-derived view.

Regeneration with another VLM/model version need not reproduce the archived
sentences byte for byte. Formatting alone does not supply motion features,
normalization statistics, a motion model or evaluation checkpoints.

For HumanML3D and Motion-X, start with their official original annotations and
the saved ID/time mapping. For Bandai, LAFAN1, Motorica and FineDance, the
historical route uses source-label MAIN context and video-generated DETAILS.
Use the original frozen IDs and partitions with an explicitly named new text
version; do not silently rerun caption regexes to replace paper memberships.
See [frozen IDs and new text versions](BENCHMARK_PROTOCOL.md#frozen-ids-with-regenerated-text).
