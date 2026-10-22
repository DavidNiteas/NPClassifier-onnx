# NPClassifier-onnx documentation (English)

**Language:** English · [简体中文](../zh-CN/index.md)

NPClassifier-onnx runs [NPClassifier](https://github.com/mwang87/NP-Classifier) inference with ONNX Runtime. It preserves the official Morgan-count fingerprints and class/pathway/superclass decoding, and provides a local HDF5-to-ONNX converter. It is an independent wrapper, not the official NPClassifier release.

## Pick a starting point

| If you want to... | Read |
|---|---|
| Understand why models are not shipped | [Why code-only](./08-why-code-only.md) |
| Install and run a small example | [Installation](./01-installation.md) → [Quickstart](./02-quickstart.md) |
| Convert official HDF5 models to ONNX | [Model conversion](./03-conversion.md) |
| Reproduce the official decision logic | [Official mode](./04-official-mode.md) |
| Use the Python API | [Python API](./05-api.md) |
| Look up a command | [CLI reference](./06-cli.md) |
| Check attribution, citation and licenses | [References and notices](./07-references.md) |

## How to read these docs

- **First time here?** [Install](./01-installation.md), then follow the [Quickstart](./02-quickstart.md).
- **Need model files?** Convert the official HDF5 models locally with the [conversion guide](./03-conversion.md).
- **Need official thresholds and voting?** Read [Official mode](./04-official-mode.md).
- **Need a flag or signature?** See [CLI](./06-cli.md) and [Python API](./05-api.md).

## What you need to provide

- The official NPClassifier HDF5 models and `index_v1.json`; convert them locally.
- An input table or molecule list (SMILES or InChI).
- A model directory containing the ONNX files produced by the converter.

## Authors and license

- David Niteas `<david.ql.d@foxmail.com>`
- License: MIT for the wrapper; upstream NPClassifier is MIT, and its models/data/ontology are CC0.

**Next:** [Installation](./01-installation.md)
