# Quick start

## 1. Install

From the repository root:

```bash
pip install .
```

Optional extras:

```bash
pip install ".[excel]"     # xlsx input/output
pip install ".[hdf]"       # HDF5 table input
pip install ".[convert]"   # HDF5 -> ONNX conversion
```

## 2. Prepare a model directory

Inference reads ONNX models and a label mapping from one directory:

```text
model_dir/
├── class_model.onnx
├── pathway_model.onnx
├── superclass_model.onnx
└── index_v1.json
```

Only the models you request need to exist. The label mapping is the
`index_v1.json` file shipped with the NPClassifier source tree. The model
weights are not part of this package; generate the ONNX files with the
[conversion guide](conversion.md).

## 3. Python API

```python
from NPClassifier_onnx import NPClassifier

classifier = NPClassifier(
    "/path/to/model_dir",
    model_set=["class", "pathway", "superclass"],
)

frame = classifier.batch_predict_and_decode(
    [
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    ],
    top_k=3,
)
print(frame)
```

Each input produces exactly one output row. The `status` column is `ok` or
`invalid`; invalid rows keep their position and have empty prediction columns.

To work with raw probability matrices:

```python
probabilities, status = classifier.batch_predict(
    ["CCO", "c1ccccc1"],
    return_status=True,
)
# probabilities["class_model"].shape == (2, 687)
```

To select a subset of models:

```python
classifier = NPClassifier("/path/to/model_dir", ["class"])
frame = classifier.batch_predict_and_decode(["CCO"], top_k=5)
```

`top_k` must not exceed the number of classes of the selected model.

## 4. Predict from a table

```python
frame = classifier.predict_from_table_file(
    "molecules.csv",
    tag_column="smiles",
    batch_size=256,
    top_k=3,
)
frame.to_csv("results.csv")
```

The returned frame is indexed by the original molecule strings. Supported
inputs: `.csv`, `.tsv`/`.tab`/`.txt`, `.xlsx`, `.pkl`, `.parquet`, and
`.h5`/`.hdf5`/`.hdf` (the last requires the `[hdf]` extra).

## 5. Command line

```bash
npc \
  --model-dir /path/to/model_dir \
  --input molecules.csv \
  --column smiles \
  --batch-size 256 \
  --top-k 3 \
  --output results.csv
```

The output format is selected by the output extension: `.csv`, `.tsv`, `.xlsx`,
`.pkl` or `.parquet`. Excel output requires the `[excel]` extra. Use `--quiet`
to hide the progress bar and summary line.

To use the official NPClassifier decision logic instead of top-k scores:

```bash
npc --model-dir /path/to/model_dir --input molecules.csv --mode official
```

`--mode official` applies class 0.1, superclass 0.3, pathway 0.5 and the
official hierarchical voting; see [official mode](en/official-mode.md).

Compatibility aliases are accepted: `--model_dir`, `--input_file`,
`--tag_column`, `--batch_size`, `--top_k`.

## 6. Output columns

Column names use the ONNX file stem as the prefix:

```text
status
class_model_prob_1, class_model_label_1, class_model_prob_2, class_model_label_2, ...
pathway_model_prob_1, pathway_model_label_1, ...
superclass_model_prob_1, superclass_model_label_1, ...
```

If a model is named `class.onnx` instead of `class_model.onnx`, its columns use
the `class_` prefix.

## 7. Input notes

- Strings starting with `InChI` are parsed with RDKit's InChI parser; all other
  strings are treated as SMILES.
- Hydrogens are added explicitly before fingerprinting, matching the official
  NPClassifier implementation.
- Invalid molecules are reported as `invalid` rather than silently dropped.
  Use `on_error="raise"` with `molecules2fingerprints()` if you prefer an
  exception.
- The loader validates that each ONNX output width matches the number of labels
  in `index_v1.json`; a mismatch raises a clear error.

## 8. Performance

ONNX Runtime uses the CPU execution provider by default. For large tables,
increase `--batch-size` until memory or latency becomes the limiting factor.
Fingerprint generation is usually the dominant cost, so preprocessing
molecules once and reusing the resulting fingerprints can help repeated runs.
