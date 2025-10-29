# NPClassifier-onnx

NPClassifier 是一个用于分类天然产物（Natural Products, NP）的工具，主要基于机器学习模型和特定的分类学方法。它能够根据化学结构（如 SMILES 表示法）对天然产物进行分类，并提供详细的分类层次信息。该工具支持本地部署，当官方途径是通过 Docker 和 TensorFlow 提供服务。本项目基于NPClassifier官方公开的TensorFlow模型权重，基于通用的ONNX格式重写了整个模型的推理过程，打包为一个可pip按照的Python代码库，并提供一个CLI应用程序入口。

## 功能特点

- 支持SMILES和InChI格式的分子输入
- 基于Morgan指纹的分子特征提取
- 批量处理分子数据
- 支持多种文件格式（CSV、Excel、TSV等）
- 提供命令行接口和Python API

## 安装

```bash
git clone https://github.com/npclassifier/NPClassifier.git
cd NPClassifier
pip install .
```

## 使用方法

### 命令行接口

```bash
npc --model_dir /path/to/models --input_file /path/to/input.xlsx --tag_column inchi --batch_size 256 --top_k 3 --output results.xlsx
```

参数说明：
- `--model_dir`: 模型文件夹路径（必需）
- `--input_file`: 输入文件路径（必需）
- `--tag_column`: 输入文件中包含分子结构的列名（默认：inchi）
- `--batch_size`: 批处理大小（默认：256）
- `--top_k`: 返回的前k个预测结果（默认：1）
- `--output`: 输出文件路径（默认：./npc_results.xlsx）

### Python API

```python
from NPClassifier import NPClassifier

# 初始化分类器
classifier = NPClassifier("/path/to/models", ["class", "pathway", "superclass"])

# 预测单个分子
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
results = classifier.batch_predict_and_decode([smiles], top_k=3)
print(results)

# 从文件批量预测
results_df = classifier.predict_from_table_file(
    data_path="/path/to/input.xlsx",
    tag_column="inchi",
    batch_size=256,
    top_k=3
)
results_df.to_excel("/path/to/output.xlsx")
```

## 输入格式

NPClassifier支持以下分子格式：
- SMILES字符串
- InChI字符串

支持以下输入文件格式：
- CSV (.csv)
- Excel (.xlsx)
- TSV/TSV (.tsv, .tab, .txt)
- Pickle (.pkl)
- Parquet (.parquet)
- HDF5 (.h5, .hdf5, .hdf)

## 输出格式

输出为Excel文件，包含以下列：
- 对于每个模型类别（class、pathway、superclass）：
  - `{model}_prob_1`, `{model}_label_1`: 最高概率的预测结果
  - `{model}_prob_2`, `{model}_label_2`: 第二高概率的预测结果（如果top_k > 1）
  - 以此类推...

## 模型

NPClassifier使用预训练的ONNX模型进行预测。模型文件应包含以下结构：

```
model/np_classifier/
├── class_model.onnx      # 类别分类模型
├── pathway_model.onnx    # 通路分类模型
├── superclass_model.onnx # 超类分类模型
└── index_v1.json         # 标签映射文件
```

### 模型文件说明

- **class_model.onnx**: 用于预测天然产物的具体类别（如生物碱、萜类等）
- **pathway_model.onnx**: 用于预测天然产物的生物合成通路
- **superclass_model.onnx**: 用于预测天然产物的大类（如聚酮化合物、苯丙素类等）
- **index_v1.json**: 包含所有模型的标签映射关系，将模型输出的数字索引转换为可读的类别名称

### 模型使用

在初始化NPClassifier时，需要指定模型目录路径：

```python
from NPClassifier import NPClassifier

# 指定模型目录
model_dir = "/path/to/model/np_classifier"

# 初始化分类器，指定要使用的模型类型
classifier = NPClassifier(model_dir, ["class", "pathway", "superclass"])
```

### 模型输出

每个模型会输出预测概率和对应的标签：

- **概率值**: 0到1之间的浮点数，表示预测的置信度
- **标签**: 对应的类别名称，从index_v1.json文件中映射而来

通过设置`top_k`参数，可以获取每个模型的前k个最可能预测结果。

## 依赖项

- numpy
- pandas
- onnxruntime
- rdkit
- tqdm
- openpyxl
- tables

## 引用

如果您在研究中使用了NPClassifier-ONNX，请引用相关论文。
- [NPClassifier 论文](https://pubs.acs.org/doi/10.1021/acs.jnatprod.1c00399)
