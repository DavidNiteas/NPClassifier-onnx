# 快速开始

## 1. 安装

在仓库根目录执行：

```bash
pip install .
```

可选依赖：

```bash
pip install ".[excel]"     # xlsx 读写
pip install ".[hdf]"       # HDF5 表格输入
pip install ".[convert]"   # HDF5 -> ONNX 转换
```

## 2. 准备模型目录

推理从一个目录读取 ONNX 模型和标签映射：

```text
model_dir/
├── class_model.onnx
├── pathway_model.onnx
├── superclass_model.onnx
└── index_v1.json
```

只需要放实际使用的模型。标签映射是 NPClassifier 源码树中的
`index_v1.json`。模型权重不属于本包；请按照[模型转换指南](conversion.zh.md)
在本地生成 ONNX 文件。

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

每条输入严格对应一行输出。`status` 列为 `ok` 或 `invalid`；解析失败的行
保留原位置，预测列为空。

需要原始概率矩阵时：

```python
probabilities, status = classifier.batch_predict(
    ["CCO", "c1ccccc1"],
    return_status=True,
)
# probabilities["class_model"].shape == (2, 687)
```

只使用部分模型：

```python
classifier = NPClassifier("/path/to/model_dir", ["class"])
frame = classifier.batch_predict_and_decode(["CCO"], top_k=5)
```

`top_k` 不能超过所选模型的类别数。

## 4. 从表格预测

```python
frame = classifier.predict_from_table_file(
    "molecules.csv",
    tag_column="smiles",
    batch_size=256,
    top_k=3,
)
frame.to_csv("results.csv")
```

返回表的索引是原始分子字符串。支持 `.csv`、`.tsv`/`.tab`/`.txt`、`.xlsx`、
`.pkl`、`.parquet`，以及 `.h5`/`.hdf5`/`.hdf`（需要 `[hdf]` 额外依赖）。

## 5. 命令行

```bash
npc \
  --model-dir /path/to/model_dir \
  --input molecules.csv \
  --column smiles \
  --batch-size 256 \
  --top-k 3 \
  --output results.csv
```

输出格式由输出文件扩展名决定：`.csv`、`.tsv`、`.xlsx`、`.pkl` 或 `.parquet`。
Excel 输出需要 `[excel]` 额外依赖。加 `--quiet` 可关闭进度条和摘要。

同时兼容 `--model_dir`、`--input_file`、`--tag_column`、`--batch_size`、
`--top_k` 等旧写法。

## 6. 输出列

列名以 ONNX 文件名主干为前缀：

```text
status
class_model_prob_1, class_model_label_1, class_model_prob_2, class_model_label_2, ...
pathway_model_prob_1, pathway_model_label_1, ...
superclass_model_prob_1, superclass_model_label_1, ...
```

如果模型文件叫 `class.onnx` 而不是 `class_model.onnx`，列前缀就是 `class_`。

## 7. 输入说明

- 以 `InChI` 开头的字符串使用 RDKit 的 InChI 解析器，其余按 SMILES 解析。
- 指纹计算前会显式加氢，与官方 NPClassifier 实现一致。
- 非法分子会标记为 `invalid`，不会被静默丢弃。若希望直接抛异常，可在
  `molecules2fingerprints()` 中使用 `on_error="raise"`。
- 加载器会校验 ONNX 输出宽度与 `index_v1.json` 的标签数；不一致时会给出
  明确错误。

## 8. 性能

ONNX Runtime 默认使用 CPU。处理大表时可以增大 `--batch-size`，直到内存或
延迟成为瓶颈。指纹生成通常是主要开销；如果需要重复推理，可以预先计算并
复用指纹。
