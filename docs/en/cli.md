# Command-line interface

[中文](../zh/cli.md) | **English**

`npc` provides four subcommands. The historical single-command form is still
accepted, but new scripts should use subcommands:

```shell
npc --help
npc predict --help
npc convert --help
npc verify --help
npc doctor --help
```

## Commands

| Command | Purpose | Extra dependency |
| --- | --- | --- |
| `npc predict` | Run ONNX inference on a table | None (base install) |
| `npc convert` | Convert official Keras HDF5 models to ONNX | `.[convert]` |
| `npc verify` | Compare outputs of two ONNX model directories | None (base install) |
| `npc doctor` | Check environment, fingerprints, model dir and input | None (base install) |

No command downloads or bundles model weights.

## npc predict

```shell
npc predict \
  --model-dir /path/to/model_dir \
  --input molecules.csv \
  --column smiles \
  --top-k 3 \
  --output results.csv
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--model-dir` | required | Directory with `<model>_model.onnx` and `index*.json` |
| `--input`, `-i` | required | Input table: csv, tsv/txt, xlsx, pkl, parquet, h5 |
| `--column` | `smiles` | SMILES/InChI column |
| `--batch-size` | `256` | Molecules per inference batch |
| `--mode` | `scores` | `scores` returns top-k probabilities; `official` reproduces upstream decision logic |
| `--top-k` | `1` | Labels per model in `scores` mode |
| `--model-set` | all | Comma-separated: `class,pathway,superclass` |
| `--output`, `-o` | `npc_results.csv` | Output file; extension selects the format |
| `--class-threshold` | `0.1` | `official` class threshold |
| `--superclass-threshold` | `0.3` | `official` superclass threshold |
| `--pathway-threshold` | `0.5` | `official` pathway threshold |
| `--quiet` | off | Disable the progress bar and summary line |

The output format follows the `--output` extension: `.csv`, `.tsv`, `.xlsx`,
`.pkl`, `.parquet`. Excel needs `.[excel]`. Official-mode results are lists;
CSV/Excel join them with `;`, while pickle/parquet preserve Python lists.

### Official mode

```shell
npc predict \
  --model-dir /path/to/model_dir \
  --input molecules.csv \
  --mode official \
  --output official_results.csv
```

Official mode requires all three models and an `index_v1.json` containing
`Class_hierarchy` and `Super_hierarchy`. Thresholds can be overridden per model
release:

```shell
npc predict --model-dir /path/to/model_dir --input molecules.csv \
  --mode official --class-threshold 1e-7
```

Calibrate an override on a local validation cohort; do not present one model
release's threshold as the official default for every release. See
[official mode](official-mode.md).

## npc convert

```shell
npc convert \
  --hdf5-dir /path/to/hdf5 \
  --output-dir /path/to/model_dir \
  --verify
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--hdf5-dir` | required | Directory with official `NP_classifier_*_V1.hdf5` |
| `--output-dir` | required | Output ONNX, `index_v1.json` and `manifest.json` |
| `--index` | auto-detect | Path to the official `index_v1.json` |
| `--models` | all | Comma-separated: `class,pathway,superclass` |
| `--opset` | `13` | ONNX opset |
| `--force` | off | Overwrite existing ONNX files |
| `--verify` | off | Run converted models once with ONNX Runtime |
| `--quiet` | off | Suppress the summary |

The converter does not require TensorFlow and does not download weights.
`manifest.json` records source/output SHA-256 checksums, class counts and the
opset. See the [conversion guide](conversion.md) for the complete workflow.

## npc verify

```shell
npc verify \
  --reference-dir /path/to/reference \
  --model-dir /path/to/converted \
  --models class,pathway,superclass \
  --samples 16
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--reference-dir` | required | Trusted reference ONNX directory |
| `--model-dir` | required | ONNX directory under test |
| `--models` | all | Models to compare |
| `--samples` | `8` | Number of random fingerprint rows |
| `--seed` | `0` | Random fingerprint seed |
| `--atol` | `1e-5` | Maximum allowed absolute difference |
| `--top-k` | `1` | Top-k label agreement to compare |
| `--quiet` | off | Only print failures/errors |

One JSON line is printed per model with `max_abs_diff`, `top_k_match`,
`status` and related fields. Exit code is `0` when all checks pass, `1` when a
check fails, and `2` for argument/file errors. Random fingerprints can be far
from the training distribution, so top-k agreement only considers rows where
the reference has a clear non-zero winner; numerical error is still checked on
every row.

## npc doctor

```shell
npc doctor
npc doctor --model-dir /path/to/model_dir --models class,pathway,superclass
npc doctor --model-dir /path/to/model_dir --input molecules.csv --json
```

`doctor` checks:

- Python and availability of `numpy` / `pandas` / `onnxruntime` / `rdkit`;
- whether `h5py` / `onnx` are installed (conversion only);
- the Morgan count fingerprint shapes for `CCO`;
- model directory, ONNX input/output shapes, label counts and a `CCO` smoke
  prediction;
- whether the input file exists.

`--json` emits a machine-readable report. Exit code is `0` when every check
passes, otherwise `1`.

## Legacy form

The historical form still works and is equivalent to `npc predict`:

```shell
npc --model-dir /path/to/model_dir --input molecules.csv --top-k 3
```

Legacy option names such as `--model_dir`, `--input_file`, `--tag_column`,
`--batch_size` and `--top_k` remain accepted.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success (`verify`/`doctor`: all checks passed) |
| `1` | `verify` or `doctor` check failed |
| `2` | Argument, file, dependency or runtime error |

Errors go to stderr, for example:

```text
npc: error: conversion requires the 'convert' extra; install with 'pip install "npclassifier-onnx[convert]"'
```
