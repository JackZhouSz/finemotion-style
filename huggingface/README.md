---
pretty_name: FineMotion-Style — Bandai motion and reconstruction protocols
language:
- en
license: cc-by-nc-4.0
tags:
- 3d
- motion-capture
- motion-style-transfer
- humanml3d
---

# FineMotion-Style

This is the dataset card draft for the planned Hugging Face Dataset release.
Archives are prepared; no live Hugging Face download is claimed by this draft.

FineMotion-Style accompanies *Learning Kinematic Frequency-Aware Disentanglement
for Motion Style Transfer and Editing*, SIGGRAPH Asia 2026. The integrated
research corpus uses six sources. This hosted component distributes **Bandai
motion and annotations**, plus a separate archive of frozen protocol IDs.
Other sources are acquired from their official providers and processed locally.

[Project](https://www.dr-lab.org/projects/kinematic-frequency-motion/) ·
[Method code](https://github.com/dongran/kinematic-frequency-motion) ·
[Dataset code and reconstruction guide](https://github.com/dongran/finemotion-style)

## Download components

The selected components and their SHA256 values are recorded in the
[archive manifest](https://github.com/dongran/finemotion-style/blob/main/release_metadata/release_archives.json).

- Bandai original v1: 3,077 archived SMPL24 source sequences, 2,676 paired
  20Hz HumanML263 feature/text clips, source mappings, split projections and
  a complete BVH-to-MotionBuilder-to-SMPL example. The archived source sequences
  use 30Hz; another project can choose a different conversion rate.
- Frozen protocol v1: 292 original split/benchmark ID lists, with byte hashes
  and a description of the different full-corpus, availability and benchmark
  counts. These contain no source captions, motions or model weights.

The two components have different scopes. The protocol IDs refer to all six
sources; the Bandai download alone does not reconstruct the full research
corpus. The previously prepared three-source text archive is not selected for
this initial hosted release.

## Acquiring the other sources

- [HumanML3D](https://github.com/EricGuo5513/HumanML3D), with
  [AMASS](https://amass.is.tue.mpg.de/) where original motion parameters are needed.
- [Motion-X](https://github.com/IDEA-Research/Motion-X).
- [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).
- [Motorica Dance Dataset](https://github.com/simonalexanderson/MotoricaDanceDataset).
- [FineDance](https://github.com/li-ronghui/FineDance).

Obtain each dataset, labels and required model assets under its provider's terms.
The repository documents source conversion, ID/crop mappings, video descriptions,
HumanML-format text, lemma/POS and benchmark rules. Historical code and recovered
settings are distinguished from fresh end-to-end verification in the
[source guide](https://github.com/dongran/finemotion-style/blob/main/docs/SOURCE_RECONSTRUCTION.md).

## Text, time tags and frozen IDs

HumanML3D and Motion-X use their inherited annotation routes. For the four newly
annotated sources, historical MAIN descriptions commonly use source labels and
DETAILS were generated with VideoLLaMA3. The text format is:

```text
caption#lemma/POS lemma/POS ...#start_seconds#end_seconds
```

The two caption lines are alternatives for the same motion; `0#0` denotes the
whole paired clip. POS tags describe words, not the four benchmark families.

Regenerating descriptions with a VLM need not reproduce the old wording or
POS tokens. Keep a separately named text version while retaining the historical
clip IDs, train/validation/test partitions and benchmark lists. Missing source
files must be reported rather than filled with new IDs. Reclassifying revised
captions defines a new benchmark version, not the frozen paper protocol.

The saved motion-availability pool has 59,038 IDs; the paper's 59,039 count has
an unresolved one-ID discrepancy. The 67,907-ID base text/feature pool and its
13,357 strict style labels have different scopes. See the
[benchmark protocol](https://github.com/dongran/finemotion-style/blob/main/docs/BENCHMARK_PROTOCOL.md).

## Provenance, limitations and licences

Bandai data and annotations are distributed under **CC BY-NC 4.0**, with source
attribution, original notices and the transformation history in the archive.
The source creator is Bandai Namco Research Inc. The conversion uses
[our 2025 processing method](https://github.com/dongran/tempo-changing-music2motion).
Project code, documentation and the separate protocol-ID material retain their
MIT notices; the card's data-licence tag does not replace component-specific terms.

SMPL model weights, blend shapes and characterized target templates are obtained
separately. Other providers' motions and full annotation corpora are not included.
No source creator endorses this adaptation. See the
[source notices](https://github.com/dongran/finemotion-style/blob/main/NOTICE.md).

The original annotations are preserved; Qwen revision proposals are not applied
automatically. The four-source first-pass review completed on 2026-10-16, but
model flags are not ground-truth accuracy measurements. The selected October 16
Bandai packaging revision translates comments and removes absolute preparation
paths while preserving all motion, feature, caption and split bytes from the
October 14 archive. Current findings and bounded known issues are in the
[release guide](https://github.com/dongran/finemotion-style/blob/main/docs/RELEASE_V1.md).

Preserve an immutable original-v1 revision. Publish later accepted caption or
conversion changes as separately identified versions with old/new hashes and
change ledgers. A different text version does not silently replace frozen IDs.

## Citation

```bibtex
@inproceedings{dong2026kinematic,
  title={Learning Kinematic Frequency-Aware Disentanglement for Motion Style Transfer and Editing},
  author={Dong, Ran and Xie, Haoran and Yang, Xi},
  booktitle={SIGGRAPH Asia 2026 Conference Papers},
  year={2026},
  note={to appear}
}
```

Retain the Bandai source credit and cite each additional source used in a local
reconstruction; the FineMotion citation does not replace upstream attribution.
