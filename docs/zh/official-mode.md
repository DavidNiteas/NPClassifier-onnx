# Official NPClassifier mode

**中文** | [English](../en/official-mode.md)

The default `scores` mode returns raw per-class probabilities. The
`official` mode reproduces the decision logic used by the official
NPClassifier web application.

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

Use `scores` for AP/AUROC and custom thresholds; use `official` when you need to
match the official tool's output contract.

## CLI

```bash
npc \
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

The decision logic is independent of model weights, but the ONNX files you
convert may come from a different official model release than the web
application. Thresholds and voting will match; exact probabilities and result
lists can differ when the underlying weights differ.
