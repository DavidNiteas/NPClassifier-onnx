# 为什么只发代码

**中文** | [English](../en/08-why-code-only.md) · [返回主页](../../README.md#zh-cn)

**层级：** 0 — 范围

**前置条件：** 无。

本仓库包含 ONNX 推理封装、HDF5 → ONNX 转换器和文档，不包含 NPClassifier 模型权重、转换后的 ONNX 文件或 `index_v1.json`。

## 模型

官方 V1 Keras HDF5 模型与标签映射来自 [NPClassifier 项目](https://github.com/mwang87/NP-Classifier)。它们是 CC0，但本仓库不转存。请按[模型转换](./03-conversion.md)在本地生成 ONNX 文件。

## 数据

不需要训练数据。推理输入是你自己的 SMILES/InChI 表或分子列表；仓库不附带示例数据集。

## 为什么不打包模型

每个 ONNX 模型约 250 MiB，且模型版本会变化。本地转换可以保留源 HDF5、校验和与 ONNX 图，便于审计，也避免把包绑定到某个模型快照。转换器会写出包含源文件与输出校验和的 `manifest.json`。

## 官方项目

本项目不是 NPClassifier 官方版本。官方实现、在线服务与支持见 <https://github.com/mwang87/NP-Classifier>。

**下一页：** [文档索引](./index.md)
