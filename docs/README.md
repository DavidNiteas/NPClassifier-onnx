# NPClassifier-onnx documentation

[中文](README.zh.md) · [Project README](../README.md)

`NPClassifier-onnx` runs [NPClassifier](https://github.com/mwang87/NP-Classifier)
inference with ONNX Runtime. It preserves the official Morgan-count fingerprint
definition and class/pathway/superclass decoding while removing the TensorFlow
serving dependency.

This repository contains **inference code and a reproducible HDF5-to-ONNX
converter**, not model weights. ONNX files are generated locally from the
official Keras models and are not committed, packaged or distributed here.

## Documentation

| Page | What you will find |
| --- | --- |
| [Quick start](quickstart.md) | Install, model directory, Python API, CLI |
| [Official mode](en/official-mode.md) | Sigmoid outputs, official thresholds, voting |
| [Conversion](conversion.md) | HDF5-to-ONNX workflow, manifest, parity check |
| [Python API](en/api.md) | `NPClassifier` methods and score semantics |
| [CLI reference](en/cli.md) | `predict`, `convert`, `verify`, `doctor` |
| [References and notices](en/references.md) | Upstream repo, citation, licenses, non-official notice |

中文文档：[总览](README.zh.md) · [快速开始](quickstart.zh.md) ·
[官方模式](zh/official-mode.md) · [模型转换](conversion.zh.md) ·
[Python API](zh/api.md) · [命令行接口](zh/cli.md) · [引用与声明](zh/references.md)

## Scope and license

The wrapper code is MIT licensed. NPClassifier is MIT licensed; the official
models, data and ontology are CC0. This project is not the official NPClassifier
release. See [References and notices](en/references.md) and the upstream
[repository](https://github.com/mwang87/NP-Classifier).
