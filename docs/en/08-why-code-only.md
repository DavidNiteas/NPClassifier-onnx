# Why code-only

[中文](../zh-CN/08-why-code-only.md) | **English** · [Documentation home](../../README.md#english)

**Level:** 0 — Scope

**Prerequisites:** None.

This repository contains the ONNX inference wrapper, the HDF5-to-ONNX converter and documentation. It does not include NPClassifier model weights, converted ONNX files or `index_v1.json`.

## Models

The official V1 Keras HDF5 models and label mapping come from the [NPClassifier project](https://github.com/mwang87/NP-Classifier). They are CC0, but they are not redistributed here. Convert them locally with the [conversion guide](./03-conversion.md).

## Data

No training data is needed. Inference input is your own SMILES/InChI table or molecule list. The repository ships no example datasets.

## Why not bundle models

Each ONNX model is large (roughly 250 MiB), and model releases change. Local conversion keeps the source HDF5, checksums and ONNX graph auditable, and avoids tying the package to one model snapshot. The converter writes a `manifest.json` with source and output checksums.

## Official project

This is not the official NPClassifier release. For the official implementation, web service or support, use <https://github.com/mwang87/NP-Classifier>.

**Next:** [Documentation index](./index.md)
