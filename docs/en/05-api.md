# Python API reference

[中文](../zh-CN/05-api.md) | **English** · [Documentation home](../../README.md#english)

**Level:** 3 — Reference

**Prerequisites:** [Quickstart](./02-quickstart.md).

The main entry point is `NPClassifier`. Model weights are never bundled; pass a
directory containing ONNX files and an `index*.json` label mapping.

## `NPClassifier(model_dir_path, model_set=None)`

Loads the requested ONNX sessions and the label mapping.

```python
from NPClassifier_onnx import NPClassifier

classifier = NPClassifier(
    "/path/to/model_dir",
    model_set=["class", "pathway", "superclass"],
)
```

- `model_dir_path`: directory with `<model>_model.onnx` (or `<model>.onnx`) and
  `index_v1.json`.
- `model_set`: any subset of `class`, `pathway`, `superclass`; default is all
  three. Only requested models need to be present.
- Raises `FileNotFoundError`, `NotADirectoryError` or `ValueError` when the
  directory, model file, label mapping or model/label widths are inconsistent.

Attributes:

| Attribute | Meaning |
| --- | --- |
| `sessions` | `{session_key: onnxruntime.InferenceSession}` |
| `mapping` | `{model_name: pandas.Series}` label-index mapping |
| `ontology` | parsed `index*.json`, including hierarchy sections |

## Molecule parsing and fingerprints

### `NPClassifier.get_mol(value) -> rdkit.Chem.Mol`

Parses SMILES or InChI and adds explicit hydrogens, matching the official
NPClassifier input preparation. Raises `ValueError` for empty or unparsable
strings.

### `NPClassifier.calculate_fingerprint(mol, n_bits=2048, radius=2)`

Returns `(formula, binary)` float32 arrays shaped `(1, 2048)` and `(1, 4096)`.

- `formula` is the radius-0 Morgan count vector.
- `binary` contains the radius-1 and radius-2 count increments.
- The result matches the official formula/count fingerprint definition,
  including duplicated atom environments.

### `NPClassifier.molecules2fingerprints(input_list, return_status=False, on_error="zero")`

Featurizes a sequence of SMILES/InChI strings or RDKit molecules.

- Invalid rows produce zero fingerprints and `status="invalid"` by default, so
  output rows stay aligned.
- `on_error="raise"` fails on the first invalid input instead.
- With `return_status=True`, returns `(formula, binary, status)`.

## Inference

### `batch_predict(input_list, used_model="auto", return_status=False)`

Returns `{session_key: probabilities}` with shape `(n_molecules, n_classes)`.
Rows are aligned with the input; invalid rows are all zeros.

### `batch_predict_and_decode(input_list, used_model="auto", top_k=1)`

Returns a `pandas.DataFrame` with one row per input and columns such as
`class_model_prob_1`, `class_model_label_1`, plus `status`.

### `predict_from_table_file(data_path, tag_column="smiles", batch_size=256, top_k=1, progress=True)`

Reads a table column and returns a decoded DataFrame indexed by the original
molecule strings. Supported suffixes: `.csv`, `.tsv`/`.tab`/`.txt`, `.xlsx`,
`.pkl`, `.parquet`, `.h5`/`.hdf5`/`.hdf` (the last requires `.[hdf]`).

### `predict_official(input_list, thresholds=None)`

Reproduces the official decision logic and returns a DataFrame with
`status`, `isglycoside`, `pathway_results`, `superclass_results` and
`class_results`.

`thresholds` accepts keys `class`, `superclass`, `pathway` and overrides the
official defaults `0.1` / `0.3` / `0.5`. Official mode requires all three models
and hierarchy sections in the index file.

### `predict_official_from_table_file(data_path, tag_column="smiles", batch_size=256, progress=True, thresholds=None)`

Table-input variant of `predict_official`.

## Score semantics

The three heads are sigmoid outputs trained with binary cross-entropy:

- each value is a per-class probability estimate;
- not softmax;
- not normalized across classes;
- not guaranteed to be calibrated.

The official thresholds are decision rules, not properties of the model output.
They are not guaranteed to transfer across model releases; calibrate overrides
on a local validation cohort.

## Decoding helpers

| Helper | Purpose |
| --- | --- |
| `index2label(index, model_name)` | Map one index or a sequence to labels |
| `indexs2labels(indexs, model_name)` | Map an array of indices, preserving shape |
| `get_top_k_index(array, top_k)` | Return top-k indices in descending score order |
| `get_top_k_prob(array, top_k)` | Return top-k scores in descending order |
| `decode_result(results, model_name, top_k=1, valid=None)` | Decode one model's matrix |
| `decode_results(results_dict, top_k=1, status=None)` | Decode multiple models and add `status` |

`top_k` must be positive and no larger than the model's class count.

**Next:** [Command-line interface](./06-cli.md)
