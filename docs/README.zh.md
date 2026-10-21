# NPClassifier-onnx 中文文档

[English README](../README.md) · [快速开始](quickstart.zh.md) · [模型转换](conversion.zh.md) · [命令行接口](zh/cli.md) · [引用与声明](zh/references.md)

`NPClassifier-onnx` 用 ONNX Runtime 运行
[NPClassifier](https://github.com/mwang87/NP-Classifier) 推理。它保留官方实现的
Morgan 计数指纹定义，以及 class / pathway / superclass 三类标签的解码方式，
同时不再依赖 TensorFlow 服务。

本仓库只包含**推理代码和可复现的 HDF5 → ONNX 转换工具**，不包含模型权重。
ONNX 文件由官方 Keras 模型在本地转换得到，不会提交到 Git，也不会随包分发。

## 能力

- 支持 SMILES、InChI，单条或批量输入
- 与官方实现一致的 formula/count 指纹
- 通过 ONNX Runtime 输出 class、pathway、superclass
- 输出行与输入严格对齐，并提供 `ok` / `invalid` 状态
- 支持 CSV、TSV/TXT、Excel、pickle、parquet、HDF5 表格输入
- 提供 Python API 和 `npc` 命令行工具
- 转换工具不依赖 TensorFlow

## 安装

```bash
git clone https://github.com/DavidNiteas/NPClassifier-onnx.git
cd NPClassifier-onnx
pip install .

# 可选依赖
pip install ".[excel]"     # xlsx 读写
pip install ".[hdf]"       # HDF5 表格输入
pip install ".[convert]"   # HDF5 -> ONNX 转换
```

## 模型目录

推理需要一个模型目录，其中包含 ONNX 模型和标签映射：

```text
model_dir/
├── class_model.onnx
├── pathway_model.onnx
├── superclass_model.onnx
├── index_v1.json
└── manifest.json          # 可选，由转换工具生成
```

如果只使用部分模型，只需放对应文件。加载器按输入特征宽度（2048 和 4096）
识别输入，因此 `input_2048`/`input_4096` 和 `inputs`/`inputs_1` 两种命名都支持。

模型权重不在本仓库中。转换流程见[模型转换指南](conversion.zh.md)。

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

`status` 为 `ok` 表示分子解析成功，`invalid` 表示解析失败。失败行仍保留在
输出中，预测列为空。若需要原始概率矩阵：

```python
probabilities, status = classifier.batch_predict(
    ["CCO", "c1ccccc1"], return_status=True
)
```

要复刻 NPClassifier 官方决策逻辑（class 0.1、superclass 0.3、pathway 0.5、
层级投票和糖苷检测）：

```python
official = classifier.predict_official(["CCO", "c1ccccc1"])
print(official[["isglycoside", "pathway_results", "class_results"]])
```

`official` 模式需要三个 ONNX 模型，且 `index_v1.json` 必须包含
`Class_hierarchy` 和 `Super_hierarchy`。详见[官方模式](zh/official-mode.md)。

## 命令行

```bash
npc predict \
  --model-dir /path/to/model_dir \
  --input molecules.csv \
  --column smiles \
  --top-k 3 \
  --output results.csv
```

旧版单命令写法仍兼容：

```bash
npc --model-dir /path/to/model_dir --input molecules.csv --top-k 3
```

`npc` 还提供 `convert`、`verify`、`doctor` 子命令。同时兼容 `--model_dir`、
`--input_file`、`--tag_column`、`--batch_size`、`--top_k` 等下划线写法。
完整参数见[命令行接口](zh/cli.md)。

输出列名以模型文件名主干为前缀：

- `class_model_prob_1`、`class_model_label_1`、……
- `pathway_model_prob_1`、`pathway_model_label_1`、……
- `superclass_model_prob_1`、`superclass_model_label_1`、……
- `status`

## 文档

- [快速开始](quickstart.zh.md) · [Quick start](quickstart.md)
- [官方模式](zh/official-mode.md) · [Official mode](en/official-mode.md)
- [模型转换](conversion.zh.md) · [Conversion](conversion.md)
- [Python API](zh/api.md) · [Python API](en/api.md)
- [命令行接口](zh/cli.md) · [CLI reference](en/cli.md)
- [引用与声明](zh/references.md) · [References](en/references.md)
- [English README](../README.md)

## 许可证与引用

`NPClassifier-onnx` 是独立的第三方推理封装，**不是** NPClassifier 官方版本。
需要官方实现、Web 服务或官方支持时，请前往
[NPClassifier 仓库](https://github.com/mwang87/NP-Classifier)。

包装代码使用 MIT 许可证。NPClassifier 本身使用 MIT 许可证；官方模型、数据和
ontology 使用 CC0。本仓库不分发模型权重。详见 [LICENSE](../LICENSE) 和
[引用与声明](zh/references.md)。

如果研究中使用 NPClassifier，请引用：

> Kim, H. W.; Wang, M.; Leber, C. A.; et al. NPClassifier: A Deep Neural
> Network-Based Structural Classification Tool for Natural Products.
> *J. Nat. Prod.* **2021**, 84, 2795–2807.
> https://doi.org/10.1021/acs.jnatprod.1c00399
