# AGENTS.md

Instructions for AI assistants working in `NPClassifier-onnx`.

## What this repo is

`NPClassifier-onnx` runs NPClassifier inference with ONNX Runtime. It keeps the
official Morgan-count fingerprint definition and class/pathway/superclass
decoding, and provides a local HDF5-to-ONNX converter.

It is an independent wrapper, **not** the official NPClassifier release. It does
not ship NPClassifier weights, ONNX files or `index_v1.json`.

## First, figure out what the user wants

| User asks for | What to do |
|---|---|
| "Predict these molecules" | Check for a model directory; if none, convert the official HDF5 models first |
| "Convert the official models" | Confirm the HDF5 and index source, then run `npc convert` |
| "Reproduce official results" | Use `--mode official`; confirm the model release and thresholds |
| "Check my conversion" | Run `npc verify` against a trusted reference directory |
| "Just run something" | Run `npc doctor` and a small prediction; label it as a smoke test |

Do not start inference before confirming that the model directory is complete.
Do not invent predictions or parity numbers.

## Hard rules

- Do not commit ONNX files, HDF5 models, `index_v1.json`, datasets or artifacts.
- Do not fabricate fingerprints, predictions or parity results.
- Only use the official NPClassifier HDF5 models and index, or a model directory
  the user already trusts.
- Do not claim this is the official NPClassifier release.
- Do not use unofficial mirrors for model weights.
- Keep changes small and run the checks before answering.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip

python -m pip install -e ".[dev]"
python -m pip install -e ".[convert]"
```

Verify:

```bash
npc --help
python -c "import NPClassifier_onnx; print(NPClassifier_onnx.__version__)"
```

There is no CI in this repo. Run the checks locally.

## Model source: talk to the user first

The repo does not include model weights. Ask before downloading anything:

- Do you already have the official NPClassifier HDF5 models and `index_v1.json`?
- Which model release (V1 export or another version) should be used?
- Do you need `scores` mode, `official` mode, or both?
- Where should HDF5 inputs, ONNX outputs and reports be stored?
- Do you accept the CC0 terms of the official models/data/ontology?

Official sources:

- NPClassifier repository: <https://github.com/mwang87/NP-Classifier>
- NPClassifier web service: <https://npclassifier.ucsd.edu/>

The official models, data and ontology are CC0; NPClassifier software is MIT
licensed. Keep upstream attribution when redistributing a conversion.

## Conversion gate

Inputs:

```text
NP_classifier_class_V1.hdf5
NP_classifier_pathway_V1.hdf5
NP_classifier_superclass_V1.hdf5
index_v1.json
```

Convert:

```bash
npc convert \
  --hdf5-dir /path/to/hdf5 \
  --index /path/to/index_v1.json \
  --output-dir /path/to/model_dir \
  --verify
```

Expected output:

```text
model_dir/
├── class_model.onnx
├── pathway_model.onnx
├── superclass_model.onnx
├── index_v1.json
└── manifest.json
```

Check `manifest.json` for source/output checksums, opset and class counts.

## Verify parity

```bash
npc verify \
  --reference-dir /path/to/reference \
  --model-dir /path/to/model_dir \
  --models class,pathway,superclass \
  --samples 16
```

Report the actual `max_abs_diff`, top-k agreement and `status`; do not claim
global parity from one environment. Random fingerprints can be far from the
training distribution.

## Predict

```bash
npc predict \
  --model-dir /path/to/model_dir \
  --input molecules.csv \
  --column smiles \
  --top-k 3 \
  --output results.csv
```

For official decision logic:

```bash
npc predict \
  --model-dir /path/to/model_dir \
  --input molecules.csv \
  --mode official \
  --output official_results.csv
```

`official` requires all three models and `Class_hierarchy` /
`Super_hierarchy` in the index. Thresholds are model-release dependent; do not
present one release's threshold as the universal official default.

Check `status`: invalid molecules stay row-aligned with empty predictions.

## Missing dependencies

| Error | Fix |
|---|---|
| `ModuleNotFoundError: openpyxl` | `python -m pip install -e ".[excel]"` |
| `ModuleNotFoundError: tables` | `python -m pip install -e ".[hdf]"` |
| `ModuleNotFoundError: h5py` / `onnx` | `python -m pip install -e ".[convert]"` |
| `npc: command not found` | `python -m pip install -e .` |

## Before you answer

Run what is relevant:

```bash
pytest tests/ -q
npc doctor --model-dir /path/to/model_dir
git status --short
```

Remove any model, ONNX, HDF5, dataset or report artifact from the commit.

When you report results, include the command, model source/version, mode
(`scores` or `official`), thresholds, and any invalid-row counts.

## If the user says "just give me the result"

1. Look for a model directory or attached HDF5 files. If neither exists, explain
   that no weights are shipped and ask for the official models.
2. If they only want to see the pipeline work, run `npc doctor` and a small
   prediction on synthetic molecules; label it as a demo.
3. Never make up predictions, scores or parity results.

## 中文速览

先确认用户是要推理、转换模型、校验一致性，还是只跑示例。本仓库不提供 NPClassifier
权重或 ONNX 文件；推理前必须准备官方 HDF5 模型与 `index_v1.json`，用 `npc convert`
在本地转换。官方模型/数据/ontology 是 CC0，NPClassifier 软件是 MIT；不要从非官方
镜像下载，不要声称这是官方版本。`official` 模式的阈值与模型版本相关，不能把一个版本
的阈值当成所有版本的默认值。不要编造预测或一致性数字，不要提交 ONNX、HDF5、数据或报告。
