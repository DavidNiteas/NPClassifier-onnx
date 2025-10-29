# NPClassifier

NPClassifier是一个用于分类天然产物的机器学习工具，它可以根据分子结构预测天然产物的类别、超类和生物合成途径。

## 功能特点

- 支持SMILES和InChI格式的分子输入
- 基于Morgan指纹的分子特征提取
- 批量处理分子数据
- 支持多种文件格式（CSV、Excel、TSV等）
- 提供命令行接口和Python API

## 安装

### 从源码安装

```bash
git clone https://github.com/npclassifier/NPClassifier.git
cd NPClassifier
pip install -e .
```

### 使用pip安装

```bash
pip install NPClassifier
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

NPClassifier使用预训练的ONNX模型进行预测。模型文件应包含：
- `.onnx`文件：模型权重
- `index*.json`文件：标签映射

## 依赖项

- numpy
- pandas
- onnxruntime
- rdkit
- tqdm
- openpyxl
- tables

## 开发

### 安装开发依赖

```bash
pip install -e ".[dev]"
```

### 运行测试

```bash
pytest
```

### 代码格式化

```bash
black NPClassifier/
```

### 代码检查

```bash
flake8 NPClassifier/
```

## 引用

如果您在研究中使用了NPClassifier，请引用相关论文。
