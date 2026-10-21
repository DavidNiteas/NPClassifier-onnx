# 模型转换

本项目**不分发** NPClassifier 模型权重。它提供一条本地、可复现的路径，把官方
Keras HDF5 模型转换成推理所需的 ONNX 文件。

官方 V1 模型是全连接网络。`npc convert` 直接读取 HDF5 中的 Keras
`model_config` 和权重，用 NumPy 与 ONNX 构建等价图，不需要 TensorFlow，从而
避开旧的 TensorFlow 2.3 转换环境。

## 输入文件

需要官方 NPClassifier 源码树中的三个 HDF5 文件和标签映射：

```text
NP_classifier_class_V1.hdf5
NP_classifier_pathway_V1.hdf5
NP_classifier_superclass_V1.hdf5
index_v1.json
```

这些文件不在本仓库中。官方模型、数据和 ontology 使用 CC0；NPClassifier 本身
使用 MIT 许可证。

## 安装转换依赖

```bash
pip install ".[convert]"
```

或在独立环境中：

```bash
pip install -r tools/convert/requirements.txt
```

## 执行转换

```bash
npc convert \
  --hdf5-dir /path/to/hdf5 \
  --index /path/to/index_v1.json \
  --output-dir /path/to/model_dir \
  --verify
```

从源码树直接运行、未安装包时，兼容脚本 `tools/convert/convert_npc.py`
接受相同参数。

输出目录包含：

```text
class_model.onnx
pathway_model.onnx
superclass_model.onnx
index_v1.json
manifest.json
```

`manifest.json` 记录源文件名、SHA-256 校验和、输出校验和、ONNX opset 和类别数，
便于复现和审计。整个过程不会上传任何文件。

常用参数：

| 参数 | 含义 |
|---|---|
| `--models class,pathway` | 只转换指定模型 |
| `--opset 13` | ONNX opset 版本（默认 13） |
| `--force` | 覆盖已有 ONNX 文件 |
| `--verify` | 用 ONNX Runtime 试跑每个转换结果 |

转换器支持 `InputLayer`、`Concatenate`、`Dense`、`BatchNormalization`、
`Dropout`、`Activation`。官方 V1 模型只使用这些层；遇到不支持的层会明确报错，
不会静默生成错误图。

## 校验一致性

如果已有可信的参考 ONNX 导出，可以与新转换结果对比：

```bash
npc verify \
  --reference-dir /path/to/reference \
  --model-dir /path/to/model_dir \
  --models class,pathway,superclass \
  --samples 16
```

从源码树直接运行、未安装包时，`tools/convert/verify_parity.py` 接受相同参数。

工具会为每个模型输出一行 JSON，包含 `max_abs_diff`、top-k 一致性和 `status`。
超过 `--atol`（默认 `1e-5`）时退出码为 1。随机指纹可能让模型处于训练分布之外，
因此 top-k 检查只统计有明确非零赢家的行；数值一致性仍检查所有行。

请记录你实际使用的转换结果的 `max_abs_diff`、top-k 一致性和 `status`；
本仓库不把某次环境的对比结果当作所有环境的通用结论。

## Docker

在仓库根目录构建：

```bash
docker build -f tools/convert/Dockerfile -t npclassifier-convert .
```

挂载权重和输出目录运行：

```bash
docker run --rm \
  -v /path/to/hdf5:/data:ro \
  -v /path/to/model_dir:/out \
  npclassifier-convert \
  --hdf5-dir /data \
  --index /data/index_v1.json \
  --output-dir /out \
  --verify
```

镜像只包含转换代码；模型文件在运行时挂载。

## 实现说明

每个 Dense 层转换为 `MatMul`，按需加 `Add`，再接配置的激活函数。Batch
Normalization 使用存储的 moving mean/variance、gamma、beta 和 epsilon，转换为
标准 ONNX 推理算子。Dropout 在推理时是恒等变换。两个指纹输入沿特征轴拼接，
batch 维保持动态。

运行时按特征宽度识别输入，因此使用 `input_2048`/`input_4096` 的转换结果与
使用 `inputs`/`inputs_1` 的旧导出都能加载。

## 存储与许可

ONNX 文件较大（每个约 250 MiB），已被 `.gitignore` 忽略，不会进入 wheel 或
源码分发包。生成的 ONNX 文件仍受官方模型权重的 CC0 条款约束；再分发转换结果
时请保留上游署名。
