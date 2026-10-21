# NPClassifier 官方模式

[English](../en/official-mode.md) | **中文**

默认的 `scores` 模式返回原始逐类概率。`official` 模式复刻 NPClassifier 官方
网页应用的决策逻辑。

## 模型输出是什么

class、superclass、pathway 三个输出头都是 **sigmoid**，训练损失是 binary
cross-entropy。每个值是模型给出的逐类 Bernoulli 概率估计：

- 不是 softmax；
- 不要求各类概率和为一；
- 不保证校准；
- 官方应用**不是**每个头都用 0.5。

官方应用使用：

| 层级 | 官方阈值 |
|---|---:|
| class | 0.1 |
| superclass | 0.3 |
| pathway | 0.5 |

然后使用 `Class_hierarchy` 和 `Super_hierarchy` 做层级投票，并输出官方糖苷
SMARTS 检测结果。`NPClassifier_onnx/official.py` 是对官方
`Classifier/prediction_voting.py` 与 `app.py` 的直接移植。

## Python API

```python
from NPClassifier_onnx import NPClassifier

classifier = NPClassifier("/path/to/model_dir")  # 需要三个模型
frame = classifier.predict_official(["CCO", "c1ccccc1"])

print(frame.columns)
# Index(['status', 'isglycoside', 'pathway_results',
#        'superclass_results', 'class_results'], dtype='object')
```

结果列是 Python list。非法分子的行仍然保留，`status="invalid"`，结果列为空。

原始概率接口保持不变：

```python
probabilities, status = classifier.batch_predict(
    ["CCO"], used_model=["class", "pathway", "superclass"], return_status=True
)
```

需要原始概率或自定义阈值时用 `scores`；需要对齐官方工具输出时用 `official`。

## CLI

```bash
npc predict \
  --model-dir /path/to/model_dir \
  --input molecules.csv \
  --column smiles \
  --mode official \
  --output official_results.csv
```

`scores` 是默认模式。`official` 需要三个模型，且 index 文件必须包含
`Class_hierarchy` 和 `Super_hierarchy`。

CSV/XLSX 输出用 `;` 连接 list 单元格；需要精确保留 Python list 时用 pickle 或
parquet。

## 版本说明

决策逻辑与模型权重无关，但**阈值不能跨模型版本直接搬**。官方网页当前使用
V25 权重；公开转换的 V1 导出可能具有完全不同的分数尺度。在 V1 上，class 分数
可能很少达到 0.1，因此官方 0.1 阈值可能几乎不输出 class，即使分数仍有排序价值。

官方模式的阈值可以覆盖：

```bash
npc predict --model-dir /path/to/model_dir --input molecules.csv --mode official \
  --class-threshold 1e-7
```

覆盖值必须在本地 validation cohort 上校准并记录；不要把 V1 专用阈值说成官方
V25 默认值。

阈值和投票逻辑会与官方代码一致；底层权重不同会导致概率和最终结果列表不同。
