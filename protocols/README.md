# Frozen FineMotion protocol IDs

The frozen protocol archive is included directly in this GitHub repository:

- [Download `finemotion_frozen_protocol_v1_20261016.tar.gz`](https://github.com/dongran/finemotion-style/raw/refs/heads/main/protocols/finemotion_frozen_protocol_v1_20261016.tar.gz)
- Size: **7,944,168 bytes**.
- SHA256: `3209df832ac2729e30c1544bf1eb56e954179f4033efeb71a7588b27b4bd1a49`.
- [SHA256SUMS.txt](SHA256SUMS.txt).

The package contains **292 frozen split/benchmark ID lists**, preserving their
original bytes and order, plus package metadata and integrity records. These
memberships refer to the full six-source research protocol. They do not contain
source captions, motion arrays or model assets, and do not reconstruct the
source recording/crop mapping by themselves.

## Verify the archive

From the repository root, on a system with `sha256sum`:

```bash
cd protocols
sha256sum -c SHA256SUMS.txt
```

Use the expected SHA256 above to identify this exact archive version. The
archive also provides its own package metadata and checksum information.

## Use the frozen memberships

Read the [benchmark protocol](../docs/BENCHMARK_PROTOCOL.md) for the historical
base pool, pose69 availability view, strict style families, independent
attributes and January content lists. These populations have different counts
and roles; keep the saved partitions and report missing source IDs explicitly.

Obtain the other sources from their official providers and follow the
[source reconstruction guide](../docs/SOURCE_RECONSTRUCTION.md). Locally
regenerated captions can have a separate text version while retaining the
frozen memberships. Rerunning caption classification creates a new benchmark
version and must not silently replace these lists.

The two larger Bandai archives—motion/features/text and the BVH companion—are
hosted in the [Hugging Face Dataset](https://huggingface.co/datasets/randong/FineMotion-Style).
This protocol archive is supplied here on GitHub independently of those data
downloads. See the [release guide](../docs/RELEASE_V1.md) for both direct data links.

## Terms

Project protocol material retains the [MIT notice](../LICENSE). Upstream source
data, annotations, software and model assets retain their own terms. Publishing
ID membership does not expand the right to acquire, use or redistribute those
assets. See [NOTICE.md](../NOTICE.md) and the [release guide](../docs/RELEASE_V1.md).
