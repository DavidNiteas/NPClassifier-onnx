# 安装

**中文** | [English](../en/01-installation.md) · [返回主页](../../README.md#zh-cn)

**层级：** 1 — 开始

**前置条件：** Python `>=3.9` 与 `pip`。与 `smiles2class`、`unimol_lite` 一起安装时，使用 Python 3.10 或 3.11。

## 从 Git 安装

```shell
python -m pip install "git+https://github.com/DavidNiteas/NPClassifier-onnx.git@v0.3.1"
```

## 从源码安装

```shell
git clone https://github.com/DavidNiteas/NPClassifier-onnx.git
cd NPClassifier-onnx
python -m pip install .
```

## 可选依赖

| extra | 用途 |
|---|---|
| `excel` | `.xlsx` 读写（`openpyxl`） |
| `hdf` | HDF5 表格输入（`tables`） |
| `convert` | HDF5 → ONNX 转换（`h5py`、`onnx`） |
| `dev` | 测试与转换工具 |

```shell
python -m pip install ".[excel]"
python -m pip install ".[hdf]"
python -m pip install ".[convert]"
python -m pip install ".[dev]"
```

## 验证安装

```shell
npc --help
python -c "import NPClassifier_onnx; print(NPClassifier_onnx.__version__)"
```

## 模型文件

本项目不分发 ONNX 模型、HDF5 权重或 `index_v1.json`。推理需要一个由官方 NPClassifier 模型转换得到的模型目录，见[模型转换](./03-conversion.md)。

```text
model_dir/
├── class_model.onnx
├── pathway_model.onnx
├── superclass_model.onnx
├── index_v1.json
└── manifest.json          # 可选
```

## 构建 wheel

```shell
python -m pip install build
python -m build
python -m pip install dist/npclassifier_onnx-*.whl
```

## 常见安装问题

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: openpyxl` | 安装 `.[excel]` |
| `ModuleNotFoundError: tables` | 安装 `.[hdf]` |
| `ModuleNotFoundError: h5py` / `onnx` | 安装 `.[convert]` |
| `npc: command not found` | 重新执行 `python -m pip install .`，或使用 `python -m NPClassifier_onnx.cli` |
| ONNX Runtime 导入失败 | 重新安装 `onnxruntime>=1.17` |

**下一页：** [快速开始](./02-quickstart.md)
