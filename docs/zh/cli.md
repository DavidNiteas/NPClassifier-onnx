# 命令行接口

**中文** | [English](../en/cli.md)

`npc` 提供四个子命令。旧版的单命令写法仍然兼容，但新脚本建议使用子命令：

```shell
npc --help
npc predict --help
npc convert --help
npc verify --help
npc doctor --help
```

## 命令总览

| 命令 | 作用 | 额外依赖 |
| --- | --- | --- |
| `npc predict` | 用 ONNX 模型对表格做推理 | 无（基础依赖） |
| `npc convert` | 把官方 Keras HDF5 模型转换为 ONNX | `.[convert]` |
| `npc verify` | 对比两个 ONNX 模型目录的输出一致性 | 无（基础依赖） |
| `npc doctor` | 检查环境、指纹、模型目录和输入文件 | 无（基础依赖） |

所有命令都不下载、不包含模型权重。

## npc predict

```shell
npc predict \
  --model-dir /path/to/model_dir \
  --input molecules.csv \
  --column smiles \
  --top-k 3 \
  --output results.csv
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model-dir` | 必填 | 包含 `<model>_model.onnx` 和 `index*.json` 的目录 |
| `--input`, `-i` | 必填 | 输入表格：csv、tsv/txt、xlsx、pkl、parquet、h5 |
| `--column` | `smiles` | SMILES/InChI 列名 |
| `--batch-size` | `256` | 每批分子数 |
| `--mode` | `scores` | `scores` 输出 top-k 概率；`official` 复刻官方决策逻辑 |
| `--top-k` | `1` | `scores` 模式每个模型返回的标签数 |
| `--model-set` | 全部 | 逗号分隔：`class,pathway,superclass` |
| `--output`, `-o` | `npc_results.csv` | 输出文件；扩展名决定格式 |
| `--class-threshold` | `0.1` | `official` 模式的 class 阈值 |
| `--superclass-threshold` | `0.3` | `official` 模式的 superclass 阈值 |
| `--pathway-threshold` | `0.5` | `official` 模式的 pathway 阈值 |
| `--quiet` | 关闭 | 关闭进度条与汇总行 |

输出格式由 `--output` 扩展名选择：`.csv`、`.tsv`、`.xlsx`、`.pkl`、`.parquet`。Excel 需要 `.[excel]`。`official` 模式的结果是列表；CSV/Excel 会用 `;` 连接，pickle/parquet 保留 Python 列表。

### 官方模式示例

```shell
npc predict \
  --model-dir /path/to/model_dir \
  --input molecules.csv \
  --mode official \
  --output official_results.csv
```

官方模式需要三个模型，并要求 `index_v1.json` 包含 `Class_hierarchy` 与 `Super_hierarchy`。阈值可按模型版本覆盖：

```shell
npc predict --model-dir /path/to/model_dir --input molecules.csv \
  --mode official --class-threshold 1e-7
```

阈值选择应基于本地验证队列；不要把某个模型版本的阈值当作所有版本的官方默认值。详见[官方模式](official-mode.md)。

## npc convert

```shell
npc convert \
  --hdf5-dir /path/to/hdf5 \
  --output-dir /path/to/model_dir \
  --verify
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--hdf5-dir` | 必填 | 包含官方 `NP_classifier_*_V1.hdf5` 的目录 |
| `--output-dir` | 必填 | 输出 ONNX、`index_v1.json` 和 `manifest.json` |
| `--index` | 自动查找 | 官方 `index_v1.json` 路径 |
| `--models` | 全部 | 逗号分隔：`class,pathway,superclass` |
| `--opset` | `13` | ONNX opset |
| `--force` | 关闭 | 覆盖已有 ONNX |
| `--verify` | 关闭 | 转换后用 ONNX Runtime 试跑 |
| `--quiet` | 关闭 | 不打印汇总 |

转换器不依赖 TensorFlow，也不下载权重。输出目录中的 `manifest.json` 记录源文件、输出文件的 SHA-256、类别数和 opset。完整流程见[模型转换](conversion.zh.md)。

## npc verify

```shell
npc verify \
  --reference-dir /path/to/reference \
  --model-dir /path/to/converted \
  --models class,pathway,superclass \
  --samples 16
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--reference-dir` | 必填 | 可信参考 ONNX 目录 |
| `--model-dir` | 必填 | 待验证 ONNX 目录 |
| `--models` | 全部 | 要比较的模型 |
| `--samples` | `8` | 随机指纹行数 |
| `--seed` | `0` | 随机指纹种子 |
| `--atol` | `1e-5` | 最大允许绝对误差 |
| `--top-k` | `1` | 比较 top-k 标签一致性 |
| `--quiet` | 关闭 | 只输出失败/错误结果 |

每个模型输出一行 JSON，包含 `max_abs_diff`、`top_k_match`、`status` 等字段。全部通过退出码为 `0`，有失败为 `1`，参数或文件错误为 `2`。随机指纹可能远离训练分布，因此 top-k 比较只统计参考模型有明确非零赢家的行；数值误差仍检查所有行。

## npc doctor

```shell
npc doctor
npc doctor --model-dir /path/to/model_dir --models class,pathway,superclass
npc doctor --model-dir /path/to/model_dir --input molecules.csv --json
```

`doctor` 检查：

- Python 与 `numpy` / `pandas` / `onnxruntime` / `rdkit` 是否可用；
- `h5py` / `onnx` 是否安装（仅转换需要）；
- `CCO` 的 Morgan 计数指纹形状；
- 模型目录、ONNX 输入输出形状、标签数和一次 `CCO` 推理；
- 输入文件是否存在。

`--json` 输出机器可读报告。全部检查通过退出码为 `0`，否则为 `1`。

## 兼容旧写法

以下写法仍然可用，等价于 `npc predict`：

```shell
npc --model-dir /path/to/model_dir --input molecules.csv --top-k 3
```

`--model_dir`、`--input_file`、`--tag_column`、`--batch_size`、`--top_k` 等旧参数名也继续接受。

## 退出码

| 退出码 | 含义 |
| --- | --- |
| `0` | 成功（`verify`/`doctor` 表示检查通过） |
| `1` | `verify` 或 `doctor` 检查失败 |
| `2` | 参数、文件、依赖或运行时错误 |

错误信息写到 stderr，例如：

```text
npc: error: conversion requires the 'convert' extra; install with 'pip install "npclassifier-onnx[convert]"'
```
