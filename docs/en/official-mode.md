# Official NPClassifier mode

[中文](../zh/official-mode.md) | **English**

The default `scores` mode returns raw per-class probabilities. The `official`
mode reproduces the decision logic used by the official NPClassifier web
application.

## What the model outputs

The class, superclass and pathway heads are **sigmoid** outputs trained with
binary cross-entropy. Each value is a per-class Bernoulli-style probability
estimate from the model:

- not softmax;
- not normalized across classes;
- not guaranteed to be calibrated;
- the official application does **not** use 0.5 for every head.

The official application uses:

| Level | Official threshold |
|---|---:|
| class | 0.1 |
| superclass | 0.3 |
| pathway | 0.5 |

It then performs hierarchical voting with `Class_hierarchy` and
`Super_hierarchy`, and reports the glycoside SMARTS check. The implementation in
`NPClassifier_onnx/official.py` is a direct port of
`Classifier/prediction_voting.py` and `app.py` from the official repository.

## Python API

```python
from NPClassifier_onnx import NPClassifier

classifier = NPClassifier("/path/to/model_dir")  # all three models
frame = classifier.predict_official(["CCO", "c1ccccc1"])

print(frame.columns)
# Index(['status', 'isglycoside', 'pathway_results',
#        'superclass_results', 'class_results'], dtype='object')
```

Each result column contains Python lists. Invalid molecules remain row-aligned
with `status="invalid"` and empty result lists.

Raw probabilities remain available:

```python
probabilities, status = classifier.batch_predict(
    ["CCO"], used_model=["class", "pathway", "superclass"], return_status=True
)
```

Use `scores` for raw probabilities and custom thresholds; use `official` when
you need to match the official tool's output contract.

## CLI

```bash
npc predict \
  --model-dir /path/to/model_dir \
  --input molecules.csv \
  --column smiles \
  --mode official \
  --output official_results.csv
```

`scores` is the default mode. `official` requires all three models and an index
file containing `Class_hierarchy` and `Super_hierarchy`.

CSV/XLSX output joins list cells with `;`; use pickle or parquet if you need
Python lists preserved exactly.

## Version note

The decision logic is independent of model weights, but the thresholds are
**not necessarily transferable across model releases**. The official web
application uses V25 weights; a publicly converted V1 export can have a very
different score scale. On a V1 export, class scores may rarely reach 0.1, so the
official 0.1 threshold can produce almost no class results even though the
scores still rank classes usefully.

For that reason the official-mode thresholds are overridable:

```bash
npc predict --model-dir /path/to/model_dir --input molecules.csv --mode official \
  --class-threshold 1e-7
```

Calibrate an override on a local validation cohort and report the selected
value; do not present a V1-specific threshold as the official V25 default.

Thresholds and voting will match the official code; exact probabilities and
result lists can differ when the underlying weights differ.
