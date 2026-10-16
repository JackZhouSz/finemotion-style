# Attribution and license scope

## Bandai source and adapted motion

The bundled example is `dataset-2_wave-right-hand_normal_001.bvh` from **Bandai-Namco-Research-Motiondataset-2**, provided by Bandai Namco Research Inc. Original notice: © [2022] Bandai Namco Research Inc. All Rights Reserved.

Source: https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset

The original source, Bandai-skeleton calibration pose, prepared BVH, converted motion parameters, motion features, and motion previews are provided under **CC BY-NC 4.0**, with the original dataset-2 license retained at `licenses/BANDAI_DATASET2_LICENSE.txt`. License: https://creativecommons.org/licenses/by-nc/4.0/

Modifications: a calibration pose was inserted; the motion was retargeted onto a SMPL skeleton using MotionBuilder; rotations were converted to SMPL24 axis-angle; root translations were scaled from centimeters to meters; recorded frame-removal/sampling rules were applied; HumanML263 features and visual previews were produced. Display-only floor offsets in the previews are recorded separately and do not change distributed arrays. The short example caption is a visually reviewed annotation distributed under CC BY-NC 4.0.

Retain attribution, the license, and these modification notices. Noncommercial use only. The source creators do not endorse the conversion, this benchmark, or the method. Original license disclaimers continue to apply.

## Code and documentation

Project scripts and documentation are MIT licensed; see `LICENSE`. The foundational conversion implementation is [tempo-changing-music2motion / JoruriPuppet](https://github.com/dongran/tempo-changing-music2motion), Copyright (c) 2025 Ran Dong, MIT. Its original notice is retained at `licenses/TEMPO_CODE_LICENSE.txt`. The recorded helper revision is `500615821e04b248b0e0f531ec1b8096bb948ef2`.

The HumanML tokenizer reference rule follows [HumanML3D/text_process.py](https://github.com/EricGuo5513/HumanML3D/blob/main/text_process.py), Copyright (c) 2022 Chuan Guo, MIT. Its original notice is retained at `licenses/HUMANML3D_REPOSITORY_LICENSE.txt`.

MotionBuilder/pyfbsdk is Autodesk software and must be obtained separately. Historical video descriptions used [VideoLLaMA3](https://github.com/DAMO-NLP-SG/VideoLLaMA3); language annotations use [spaCy](https://spacy.io/). These dependencies retain their own licenses. They are not bundled or relicensed by this repository.

The recovered native FineDance converter is author project code, with original
commit and file hashes in `scripts/finedance/recovery_manifest.json`. Its layout
and hand approximation refer to FineDance's renderer, and its 6D convention
refers to PyTorch3D/Zhou et al.; upstream implementations are not bundled.
The new Motorica calibration helper zeroes rotation channels in user-acquired
BVH files. It does not include a Motorica-derived calibration BVH/FBX or an
SMPL target asset. Original private scripts remain preserved; English comment
localizations are recorded separately from metadata-loading adaptations.

## SMPL and model assets

Mesh previews use a separately obtained SMPL neutral model with zero shape coefficients. Please credit [SMPL](https://smpl.is.tue.mpg.de/) and its authors; the citation is included in `CITATIONS.bib`. SMPL model weights, shape blend shapes, the characterized target FBX, and target skeleton templates are not distributed here. Obtain or generate suitable target assets under their applicable terms.

Motion parameters, model assets, and software have distinct provenance and license scope. The MIT software license does not override motion-data or model-asset terms.

The Bandai BVH companion preserves native Bandai animation and a separately
labelled SMPL-cache BVH export. SMPL-Body rig material retains CC BY 4.0 credit
to the Max Planck Institute for Intelligent Systems under the
[official body licence](https://smpl.is.tue.mpg.de/bodylicense.html); Bandai
animation retains CC BY-NC 4.0. No SMPL-Model software, shape blend shapes or
target FBX/template is included in that companion.

## Source-scoped text annotations

The two large archives hosted in the
[Hugging Face Dataset](https://huggingface.co/datasets/randong/FineMotion-Style)
contain Bandai motion/text and the Bandai BVH companion. The separate 292-list frozen
protocol-ID archive is included directly in this GitHub repository under
`protocols/`, with its project MIT notice; see the [protocol package guide](protocols/README.md).
It contains memberships and package metadata, not upstream motions, source
captions or model assets. These ID lists do not grant rights to the source data.
Users acquire the other five sources from their official providers and process
them locally. The previously prepared three-source text archive described below
is retained as a staging alternative, not selected for the initial Hugging Face
download. Source-licensed issue/reference excerpts in this code repository are
distinct from a bulk annotation corpus.

The prepared original text download covers HumanML3D, Motion-X and Bandai, with each source identified in its manifest. HumanML3D captions and tokens are inherited from the upstream repository, Copyright (c) 2022 Chuan Guo, under its retained MIT notice in `licenses/HUMANML3D_REPOSITORY_LICENSE.txt`. FineMotion maps the selected source captions to clip IDs; the paper-exact changes only sentence case or final punctuation in selected fields.

Motion-X annotations, Copyright (c) 2023 The International Digital Economy Academy and their respective authors, are licensed under **CC BY-NC-SA 4.0**. FineMotion inherits the sentences and adds/maps HumanML3D-format lemma/POS and time fields. The annotations and adaptations retain attribution, noncommercial and ShareAlike terms, the modification history and the original disclaimer. See `licenses/MOTIONX_ANNOTATIONS_LICENSE.pdf`, [Motion-X](https://github.com/IDEA-Research/Motion-X) and [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). This does not grant rights to redistribute the underlying motions or separate subset assets.

Bandai's MAIN label anchors and generated DETAILS are offered by Ran Dong, © 2026, under CC BY-NC 4.0, with the original Bandai attribution and modification notices retained. Both dataset-1 and dataset-2 source terms are CC BY-NC 4.0. Their licences accompany the separate full Bandai archive; this repository's bundled motion example remains dataset-2 as identified above.

The captions embedded in `release_metadata/junk_safe_exclusion_manifest.json` are Motion-X annotations and retain CC BY-NC-SA 4.0; placing them next to software does not make them MIT. The inherited HumanML3D captions in `release_metadata/format_issue_ledger.json` and `release_metadata/temporal_issue_001034.json` retain their upstream notice. These ledgers identify bounded issues and do not invent replacement descriptions. The eight-example verification record is an author-locked paper reference, not a declaration that the full eight source clip files are in every download.

Other source motions are acquired by users from their providers. The current LAFAN1, Motorica and FineDance mixed label-derived annotation components are outside the prepared public text download while their scope is resolved. The inspected terms do not expressly prohibit every independent new factual description, but do not settle this complete mixed component's redistribution scope either. The public LaFAN context builder consumes user-obtained metadata; it does not bundle the official theme-description table. See `docs/RELEASE_V1.md` and `docs/SOURCE_RECONSTRUCTION.md`.
