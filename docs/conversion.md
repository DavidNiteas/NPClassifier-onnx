# Model conversion

This project does **not** distribute NPClassifier model weights. It provides a
local, reproducible path from the official Keras HDF5 models to the ONNX files
used by this package.

The official V1 models are fully-connected networks. `tools/convert/convert_npc.py`
reads the Keras `model_config` and weights directly from HDF5 and builds an
equivalent ONNX graph with NumPy and ONNX. It does not need TensorFlow, which
avoids the old TensorFlow 2.3 conversion environment.

## Inputs

You need the three official HDF5 files and the label mapping from the official
NPClassifier source tree:

```text
NP_classifier_class_V1.hdf5
NP_classifier_pathway_V1.hdf5
NP_classifier_superclass_V1.hdf5
index_v1.json
```

These files are not included in this repository. The official models, data and
ontology are CC0; NPClassifier itself is MIT licensed.

## Install conversion dependencies

```bash
pip install ".[convert]"
```

or, in an isolated environment:

```bash
pip install -r tools/convert/requirements.txt
```

## Convert

```bash
python tools/convert/convert_npc.py \
  --hdf5-dir /path/to/hdf5 \
  --index /path/to/index_v1.json \
  --output-dir /path/to/model_dir \
  --verify
```

The output directory will contain:

```text
class_model.onnx
pathway_model.onnx
superclass_model.onnx
index_v1.json
manifest.json
```

`manifest.json` records the source file names, SHA-256 checksums, output
checksums, ONNX opset and class counts. It helps reproduce and audit a local
conversion. Nothing is uploaded.

Useful options:

| Option | Meaning |
|---|---|
| `--models class,pathway` | Convert only selected models |
| `--opset 13` | ONNX opset version (default: 13) |
| `--force` | Overwrite existing ONNX files |
| `--verify` | Run each converted model once with ONNX Runtime |

The converter accepts these Keras layers: `InputLayer`, `Concatenate`, `Dense`,
`BatchNormalization`, `Dropout` and `Activation`. The official V1 models use
only this set. Unsupported layers fail with a clear error rather than producing
an incorrect graph.

## Verify parity

If you have a trusted reference ONNX export, compare it with the new conversion:

```bash
python tools/convert/verify_parity.py \
  --reference-dir /path/to/reference \
  --model-dir /path/to/model_dir \
  --models class,pathway,superclass \
  --samples 16
```

The tool prints one JSON object per model with `max_abs_diff`, top-k agreement
and a `status` field. It exits with code 1 if a model exceeds `--atol`
(default `1e-5`). Random fingerprints can put a model far outside its training
distribution, so the top-k check only considers rows with a clear, non-zero
winner; numerical parity is still checked on every row.

Converting the same HDF5 files with this tool and comparing to a reference
export of the official V1 models gives sub-`1e-5` maximum absolute differences
on random fingerprint batches, with matching top-1 labels.

## Docker

Build from the repository root:

```bash
docker build -f tools/convert/Dockerfile -t npclassifier-convert .
```

Run with the weights and output directory mounted:

```bash
docker run --rm \
  -v /path/to/hdf5:/data:ro \
  -v /path/to/model_dir:/out \
  npclassifier-convert \
  --hdf5-dir /data \
  --index /data/index_v1.json \
  --output-dir /out \
  --verify
```

The image contains conversion code only; model files are supplied at runtime.

## How it works

For each Dense layer the converter emits `MatMul` plus an optional `Add`, then
the configured activation. Batch normalization is emitted as the standard ONNX
inference op using the stored moving mean/variance, gamma, beta and epsilon.
Dropout is an identity at inference time. The two fingerprint inputs are
concatenated along the feature axis, and the batch dimension remains dynamic.

The runtime maps inputs by their feature width, so converted models using
`input_2048`/`input_4096` work alongside older exports using
`inputs`/`inputs_1`.

## Storage and licensing

ONNX files are large (roughly 250 MiB per model). They are ignored by
`.gitignore` and are not included in wheels or source distributions. Generated
ONNX files remain subject to the CC0 terms of the official model weights; keep
the upstream attribution when redistributing a conversion.
