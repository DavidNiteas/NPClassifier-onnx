# NPClassifier-onnx

[中文文档](docs/README.zh.md) · [Quick start](docs/quickstart.md) · [Official mode](docs/en/official-mode.md) · [Conversion](docs/conversion.md)

`NPClassifier-onnx` runs [NPClassifier](https://github.com/mwang87/NP-Classifier)
inference with ONNX Runtime. It keeps the original Morgan-count fingerprint
definition and the class/pathway/superclass label decoding, while removing the
TensorFlow serving dependency.

The repository contains **inference code and a reproducible HDF5-to-ONNX
converter**, not model weights. ONNX files are generated locally from the
official NPClassifier Keras models; they are not committed, packaged, or
distributed here.

## Features

- SMILES and InChI input, one molecule or batches
- Same formula/count fingerprint definition as the official implementation
- Class, pathway and superclass predictions from ONNX Runtime
- Raw per-class scores plus an `official` mode with the upstream 0.1/0.3/0.5
  thresholds and hierarchical voting
- Row-aligned output with an explicit `ok`/`invalid` status
- CSV, TSV/TXT, Excel, pickle, parquet and HDF5 table input
- Python API and the `npc` command line tool
- Conversion tool that does **not** require TensorFlow

## Install

```bash
git clone https://github.com/DavidNiteas/NPClassifier-onnx.git
cd NPClassifier-onnx
pip install .

# Optional extras
pip install ".[excel]"     # xlsx input/output
pip install ".[hdf]"       # HDF5 table input
pip install ".[convert]"   # HDF5 -> ONNX conversion
```

## Model directory

Inference expects one ONNX file per model and one label mapping:

```text
model_dir/
├── class_model.onnx
├── pathway_model.onnx
├── superclass_model.onnx
├── index_v1.json
└── manifest.json          # optional, written by the converter
```

Only the models requested in `model_set` need to be present. The loader maps
model inputs by their feature width (2048 and 4096), so exports using either
`input_2048`/`input_4096` or `inputs`/`inputs_1` are accepted.

Model weights are not included in this repository. See
[the conversion guide](docs/conversion.md) for the supported local workflow.

## Python API

```python
from NPClassifier_onnx import NPClassifier

classifier = NPClassifier(
    "/path/to/model_dir",
    model_set=["class", "pathway", "superclass"],
)

frame = classifier.batch_predict_and_decode(
    ["CC(=O)OC1=CC=CC=C1C(=O)O", "not-a-smiles"],
    top_k=3,
)
print(frame[["status", "class_model_label_1", "class_model_prob_1"]])
```

`status` is `ok` for a parsed molecule and `invalid` otherwise. Invalid rows
stay in the output, with empty prediction columns. To get raw probability
matrices instead:

```python
probabilities, status = classifier.batch_predict(
    ["CCO", "c1ccccc1"], return_status=True
)
```

To reproduce the official NPClassifier decision logic (class 0.1, superclass
0.3, pathway 0.5, hierarchical voting and glycoside detection):

```python
official = classifier.predict_official(["CCO", "c1ccccc1"])
print(official[["isglycoside", "pathway_results", "class_results"]])
```

`official` mode requires all three ONNX models and an `index_v1.json` containing
`Class_hierarchy` and `Super_hierarchy`. See
[the official-mode guide](docs/en/official-mode.md).

## Command line

```bash
npc \
  --model-dir /path/to/model_dir \
  --input molecules.csv \
  --column smiles \
  --top-k 3 \
  --output results.csv
```

`--model_dir`, `--input_file`, `--tag_column`, `--batch_size` and `--top_k` are
accepted as aliases for compatibility. Use `npc --help` for all options.

To reproduce the official NPClassifier decision logic (class 0.1, superclass
0.3, pathway 0.5, hierarchical voting, glycoside check):

```bash
npc --model-dir /path/to/model_dir --input molecules.csv --mode official
```

The `scores` mode remains the default. See
[the official-mode guide](docs/en/official-mode.md).

Output columns follow the model file stems:

- `class_model_prob_1`, `class_model_label_1`, ...
- `pathway_model_prob_1`, `pathway_model_label_1`, ...
- `superclass_model_prob_1`, `superclass_model_label_1`, ...
- `status`

## Documentation

- [Quick start](docs/quickstart.md) · [快速开始](docs/quickstart.zh.md)
- [Conversion guide](docs/conversion.md) · [转换指南](docs/conversion.zh.md)
- [中文总览](docs/README.zh.md)

## License and attribution

The wrapper code is MIT licensed. NPClassifier is MIT licensed; its official
models, data and ontology are CC0. See [LICENSE](LICENSE) and the
[NPClassifier repository](https://github.com/mwang87/NP-Classifier).

If you use NPClassifier in research, please cite:

> Kim, H. W.; Wang, M.; Leber, C. A.; et al. NPClassifier: A Deep Neural
> Network-Based Structural Classification Tool for Natural Products.
> *J. Nat. Prod.* **2021**, 84, 2795–2807.
> https://doi.org/10.1021/acs.jnatprod.1c00399
