# Installation

[中文](../zh-CN/01-installation.md) | **English** · [Documentation home](../../README.md#english)

**Level:** 1 — Start

**Prerequisites:** Python `>=3.9` and `pip`. When installing alongside `smiles2class` and `unimol_lite`, use Python 3.10 or 3.11.

## Install from Git

```shell
python -m pip install "git+https://github.com/DavidNiteas/NPClassifier-onnx.git@v0.3.1"
```

## Install from a checkout

```shell
git clone https://github.com/DavidNiteas/NPClassifier-onnx.git
cd NPClassifier-onnx
python -m pip install .
```

## Optional extras

| Extra | Purpose |
|---|---|
| `excel` | `.xlsx` input/output (`openpyxl`) |
| `hdf` | HDF5 table input (`tables`) |
| `convert` | HDF5-to-ONNX conversion (`h5py`, `onnx`) |
| `dev` | Tests and conversion tooling |

```shell
python -m pip install ".[excel]"
python -m pip install ".[hdf]"
python -m pip install ".[convert]"
python -m pip install ".[dev]"
```

## Verify the install

```shell
npc --help
python -c "import NPClassifier_onnx; print(NPClassifier_onnx.__version__)"
```

## Model files

This project does not distribute ONNX models, HDF5 weights or `index_v1.json`. Inference requires a model directory produced from the official NPClassifier models; see the [conversion guide](./03-conversion.md).

```text
model_dir/
├── class_model.onnx
├── pathway_model.onnx
├── superclass_model.onnx
├── index_v1.json
└── manifest.json          # optional
```

## Build a wheel

```shell
python -m pip install build
python -m build
python -m pip install dist/npclassifier_onnx-*.whl
```

## Common installation problems

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: openpyxl` | Install `.[excel]` |
| `ModuleNotFoundError: tables` | Install `.[hdf]` |
| `ModuleNotFoundError: h5py` / `onnx` | Install `.[convert]` |
| `npc: command not found` | Run `python -m pip install .` again, or use `python -m NPClassifier_onnx.cli` |
| ONNX Runtime import error | Reinstall `onnxruntime>=1.17` |

**Next:** [Quickstart](./02-quickstart.md)
