# Python API 参考

**中文** | [English](../en/05-api.md) · [返回主页](../../README.md#zh-cn)

**层级：** 3 — 参考

**前置条件：** [快速开始](./02-quickstart.md)。

主入口是 `NPClassifier`。本仓库不分发模型权重；请传入包含 ONNX 文件和
`index*.json` 标签映射的目录。

## `NPClassifier(model_dir_path, model_set=None)`

加载指定的 ONNX 会话和标签映射。

```python
from NPClassifier_onnx import NPClassifier

classifier = NPClassifier(
    "/path/to/model_dir",
    model_set=["class", "pathway", "superclass"],
)
```

- `model_dir_path`：包含 `<model>_model.onnx`（或 `<model>.onnx`）和
  `index_v1.json` 的目录。
- `model_set`：`class`、`pathway`、`superclass` 的任意子集，默认全部；只需
  放请求的模型文件。
- 目录、模型文件、标签映射或输出宽度不一致时抛出 `FileNotFoundError`、
  `NotADirectoryError` 或 `ValueError`。

属性：

| 属性 | 含义 |
| --- | --- |
| `sessions` | `{session_key: onnxruntime.InferenceSession}` |
| `mapping` | `{model_name: pandas.Series}` 标签下标映射 |
| `ontology` | 解析后的 `index*.json`，含层级表 |

## 分子解析与指纹

### `NPClassifier.get_mol(value) -> rdkit.Chem.Mol`

解析 SMILES 或 InChI，并显式加氢，与官方 NPClassifier 输入准备一致。空字符串
或无法解析时抛出 `ValueError`。

### `NPClassifier.calculate_fingerprint(mol, n_bits=2048, radius=2)`

返回 `(formula, binary)`，均为 float32，形状分别为 `(1, 2048)` 和
`(1, 4096)`。

- `formula` 是 radius-0 Morgan 计数向量；
- `binary` 是 radius-1 与 radius-2 的计数增量；
- 结果与官方 formula/count 指纹定义一致，包括重复原子环境。

### `NPClassifier.molecules2fingerprints(input_list, return_status=False, on_error="zero")`

对一组 SMILES/InChI 字符串或 RDKit 分子做特征化。

- 默认非法行返回零指纹和 `status="invalid"`，保证输出行与输入对齐；
- `on_error="raise"` 时在第一个非法输入处抛异常；
- `return_status=True` 返回 `(formula, binary, status)`。

## 推理

### `batch_predict(input_list, used_model="auto", return_status=False)`

返回 `{session_key: probabilities}`，形状 `(n_molecules, n_classes)`。行与输入
对齐；非法行全为 0。

### `batch_predict_and_decode(input_list, used_model="auto", top_k=1)`

返回 `pandas.DataFrame`，每行对应一个输入，列如 `class_model_prob_1`、
`class_model_label_1`，并带 `status`。

### `predict_from_table_file(data_path, tag_column="smiles", batch_size=256, top_k=1, progress=True)`

读取表格列并返回解码后的 DataFrame，索引是原始分子字符串。支持 `.csv`、
`.tsv`/`.tab`/`.txt`、`.xlsx`、`.pkl`、`.parquet`、`.h5`/`.hdf5`/`.hdf`
（最后一种需要 `.[hdf]`）。

### `predict_official(input_list, thresholds=None)`

复刻官方决策逻辑，返回包含 `status`、`isglycoside`、`pathway_results`、
`superclass_results`、`class_results` 的 DataFrame。

`thresholds` 可传 `class`、`superclass`、`pathway`，覆盖官方默认值
`0.1` / `0.3` / `0.5`。官方模式需要三个模型，且 index 文件包含层级表。

### `predict_official_from_table_file(data_path, tag_column="smiles", batch_size=256, progress=True, thresholds=None)`

`predict_official` 的表格输入版本。

## 分数语义

三个输出头都是 sigmoid，训练损失为 binary cross-entropy：

- 每个值是逐类概率估计；
- 不是 softmax；
- 不要求各类概率和为一；
- 不保证校准。

官方阈值是决策规则，不是模型输出的固有属性，也不保证跨模型版本可迁移；覆盖
阈值应在本地验证队列上校准。

## 解码辅助函数

| 函数 | 作用 |
| --- | --- |
| `index2label(index, model_name)` | 单个或一组下标转标签 |
| `indexs2labels(indexs, model_name)` | 按下标数组形状转标签 |
| `get_top_k_index(array, top_k)` | 按分数降序返回 top-k 下标 |
| `get_top_k_prob(array, top_k)` | 按分数降序返回 top-k 分数 |
| `decode_result(results, model_name, top_k=1, valid=None)` | 解码单个模型矩阵 |
| `decode_results(results_dict, top_k=1, status=None)` | 解码多个模型并加 `status` |

`top_k` 必须为正，且不能超过模型的类别数。

**下一页：** [命令行接口](./06-cli.md)
