# NPClassifier-onnx

**English** · [简体中文](#zh-cn)

<a id="english"></a>

## English

`NPClassifier-onnx` runs [NPClassifier](https://github.com/mwang87/NP-Classifier) inference with ONNX Runtime. It preserves the official Morgan-count fingerprint definition and class/pathway/superclass label decoding, while removing the TensorFlow serving dependency.

> This project is **not** the official NPClassifier release. For the official implementation, web service, models and support, use the [NPClassifier repository](https://github.com/mwang87/NP-Classifier).

### Docs

- **English:** [Documentation index](docs/en/index.md)
- **简体中文：** [文档索引](docs/zh-CN/index.md)

The docs are ordered by how deep you want to go:

| If you need to... | Read |
|---|---|
| Run something once | [Installation](docs/en/01-installation.md) → [Quickstart](docs/en/02-quickstart.md) |
| Convert official HDF5 models to ONNX | [Model conversion](docs/en/03-conversion.md) |
| Match the official decision logic | [Official mode](docs/en/04-official-mode.md) |
| Use the Python API | [Python API](docs/en/05-api.md) |
| Look up a command | [CLI reference](docs/en/06-cli.md) |
| Check attribution, citation and licenses | [References and notices](docs/en/07-references.md) |
| Understand why models are not shipped | [Why code-only](docs/en/08-why-code-only.md) |

### Just want results?

The repository does not distribute NPClassifier weights or ONNX files. To run inference, obtain the official HDF5 models and `index_v1.json`, convert them locally, then run `npc predict`. If an AI assistant is doing the work, point it at [`AGENTS.md`](./AGENTS.md); it will confirm the model source and license before downloading anything.

### What's here

- SMILES and InChI input, one molecule or batches
- Same formula/count fingerprint definition as the official implementation
- Class, pathway and superclass predictions from ONNX Runtime
- Raw per-class scores plus an `official` mode with upstream thresholds and hierarchical voting
- Row-aligned output with explicit `ok`/`invalid` status
- CSV, TSV/TXT, Excel, pickle, parquet and HDF5 table input
- Python API and the `npc` command line tool
- Reproducible HDF5-to-ONNX converter that does not require TensorFlow

### No model files

The repository contains inference code and the converter, not model weights. ONNX files are generated locally from the official Keras HDF5 models and are not committed, packaged or distributed here. The official models, data and ontology are CC0; NPClassifier itself is MIT licensed.

### Install

Python `>=3.9` is required. When installing alongside `smiles2class` and `unimol_lite`, use Python 3.10 or 3.11.

```bash
python -m pip install "git+https://github.com/DavidNiteas/NPClassifier-onnx.git@v0.3.1"

# Or from a checkout
git clone https://github.com/DavidNiteas/NPClassifier-onnx.git
cd NPClassifier-onnx
python -m pip install .
```

Optional extras:

```bash
python -m pip install ".[excel]"     # xlsx input/output
python -m pip install ".[hdf]"       # HDF5 table input
python -m pip install ".[convert]"   # HDF5 -> ONNX conversion
python -m pip install ".[dev]"       # tests and conversion tools
```

### Commands

```bash
# Convert official HDF5 models to ONNX
npc convert \
  --hdf5-dir /path/to/hdf5 \
  --index /path/to/index_v1.json \
  --output-dir /path/to/model_dir \
  --verify

# Predict
npc predict \
  --model-dir /path/to/model_dir \
  --input molecules.csv \
  --column smiles \
  --top-k 3 \
  --output results.csv

# Compare a conversion with a trusted reference
npc verify --reference-dir /path/to/reference --model-dir /path/to/model_dir

# Check environment and model directory
npc doctor --model-dir /path/to/model_dir
```

Output columns use the ONNX file stem as the prefix: `class_model_prob_1`, `class_model_label_1`, and the same pattern for pathway and superclass, plus `status`.

### Status

Inference, official mode, conversion, parity verification and CLI are covered by tests. Tests use synthetic data; the repository ships no model weights.

### Authors

- David Niteas `<david.ql.d@foxmail.com>`

See [`AUTHORS`](./AUTHORS). What's shipped is described in [`NOTICE`](./NOTICE).

### License

MIT for the wrapper. See [`LICENSE`](./LICENSE). Citation metadata is in [`CITATION.cff`](./CITATION.cff); the NPClassifier paper citation is in [Citation](#citation).

---

<a id="zh-cn"></a>

## 简体中文

[English](#english) · **简体中文**

`NPClassifier-onnx` 用 ONNX Runtime 运行 [NPClassifier](https://github.com/mwang87/NP-Classifier) 推理。它保留官方 Morgan 计数指纹定义，以及 class / pathway / superclass 三类标签的解码方式，同时不再依赖 TensorFlow 服务。

> 本项目**不是** NPClassifier 官方版本。官方实现、在线服务、模型与支持见 [NPClassifier 仓库](https://github.com/mwang87/NP-Classifier)。

### 文档

- **English:** [Documentation index](docs/en/index.md)
- **简体中文：** [文档索引](docs/zh-CN/index.md)

文档按使用深度排列：

| 你要做什么 | 看哪里 |
|---|---|
| 先跑通一次 | [安装](docs/zh-CN/01-installation.md) → [快速开始](docs/zh-CN/02-quickstart.md) |
| 把官方 HDF5 模型转换为 ONNX | [模型转换](docs/zh-CN/03-conversion.md) |
| 复刻官方决策逻辑 | [官方模式](docs/zh-CN/04-official-mode.md) |
| 使用 Python API | [Python API](docs/zh-CN/05-api.md) |
| 查命令参数 | [命令行接口](docs/zh-CN/06-cli.md) |
| 查归属、引用与许可 | [引用与声明](docs/zh-CN/07-references.md) |
| 理解为什么不发布模型 | [为什么只发代码](docs/zh-CN/08-why-code-only.md) |

### 只想要结果？

仓库不分发 NPClassifier 权重或 ONNX 文件。要运行推理，需要先获取官方 HDF5 模型与 `index_v1.json`，在本地转换，再运行 `npc predict`。希望 agent 代跑时，把 [`AGENTS.md`](./AGENTS.md) 交给它；它会先与你确认模型来源与许可，再下载和执行。

### 仓库里有什么

- 支持 SMILES、InChI，单条或批量输入
- 与官方实现一致的 formula/count 指纹
- 通过 ONNX Runtime 输出 class、pathway、superclass
- 原始逐类分数，以及带官方阈值与层级投票的 `official` 模式
- 输出行与输入严格对齐，并提供 `ok` / `invalid` 状态
- 支持 CSV、TSV/TXT、Excel、pickle、parquet、HDF5 表格输入
- 提供 Python API 和 `npc` 命令行工具
- 可复现的 HDF5 → ONNX 转换器，不依赖 TensorFlow

### 没有模型文件

仓库只包含推理代码与转换器，不包含模型权重。ONNX 文件由官方 Keras HDF5 模型在本地转换得到，不提交、不打包、不分发。官方模型、数据和 ontology 使用 CC0；NPClassifier 本身使用 MIT 许可证。

### 安装

需要 Python `>=3.9`。与 `smiles2class`、`unimol_lite` 一起安装时，使用 Python 3.10 或 3.11。

```bash
python -m pip install "git+https://github.com/DavidNiteas/NPClassifier-onnx.git@v0.3.1"

# 或从源码安装
git clone https://github.com/DavidNiteas/NPClassifier-onnx.git
cd NPClassifier-onnx
python -m pip install .
```

可选依赖：

```bash
python -m pip install ".[excel]"     # xlsx 读写
python -m pip install ".[hdf]"       # HDF5 表格输入
python -m pip install ".[convert]"   # HDF5 -> ONNX 转换
python -m pip install ".[dev]"       # 测试与转换工具
```

### 常用命令

```bash
# 把官方 HDF5 模型转换为 ONNX
npc convert \
  --hdf5-dir /path/to/hdf5 \
  --index /path/to/index_v1.json \
  --output-dir /path/to/model_dir \
  --verify

# 预测
npc predict \
  --model-dir /path/to/model_dir \
  --input molecules.csv \
  --column smiles \
  --top-k 3 \
  --output results.csv

# 与可信的参考导出对比
npc verify --reference-dir /path/to/reference --model-dir /path/to/model_dir

# 检查环境与模型目录
npc doctor --model-dir /path/to/model_dir
```

输出列名以 ONNX 文件名主干为前缀：`class_model_prob_1`、`class_model_label_1`，pathway 与 superclass 同理，另有 `status`。

### 状态

推理、官方模式、模型转换、一致性校验与 CLI 有测试覆盖；测试使用合成数据，仓库不附带模型权重。

### 作者

- David Niteas `<david.ql.d@foxmail.com>`

见 [`AUTHORS`](./AUTHORS)。仓库包含什么见 [`NOTICE`](./NOTICE)。

### 许可证

封装代码使用 MIT，见 [`LICENSE`](./LICENSE)。引用元数据见 [`CITATION.cff`](./CITATION.cff)；NPClassifier 论文引用见[引用](#citation)。

---

<a id="citation"></a>

## 引用 / Citation

使用 NPClassifier 或其分类结果时，应引用原始论文。If you use NPClassifier or its classifications in research, cite the original paper:

```bibtex
@article{kim2021npclassifier,
  title   = {NPClassifier: A Deep Neural Network-Based Structural Classification Tool for Natural Products},
  author  = {Kim, Hyun Woo and Wang, Mingxun and Leber, Christopher A. and others},
  journal = {Journal of Natural Products},
  volume  = {84},
  number  = {11},
  pages   = {2795--2807},
  year    = {2021},
  doi     = {10.1021/acs.jnatprod.1c00399}
}
```

- DOI: <https://doi.org/10.1021/acs.jnatprod.1c00399>
- Title: *NPClassifier: A Deep Neural Network-Based Structural Classification Tool for Natural Products*
- Journal: *Journal of Natural Products* **2021**, 84, 2795–2807
- Full notices: [English](docs/en/07-references.md) · [简体中文](docs/zh-CN/07-references.md)
