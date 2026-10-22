# NPClassifier-onnx 文档（简体中文）

**语言：** [English](../en/index.md) · 简体中文

NPClassifier-onnx 用 ONNX Runtime 运行 [NPClassifier](https://github.com/mwang87/NP-Classifier) 推理。它保留官方 Morgan 计数指纹与 class / pathway / superclass 解码，并提供本地 HDF5 → ONNX 转换器。它是独立封装，不是 NPClassifier 官方版本。

## 从哪开始

| 你想做什么 | 阅读顺序 |
|---|---|
| 理解为什么不发布模型 | [为什么只发代码](./08-why-code-only.md) |
| 安装并跑通一个小例子 | [安装](./01-installation.md) → [快速开始](./02-quickstart.md) |
| 把官方 HDF5 模型转换为 ONNX | [模型转换](./03-conversion.md) |
| 复刻官方决策逻辑 | [官方模式](./04-official-mode.md) |
| 使用 Python API | [Python API](./05-api.md) |
| 查命令参数 | [命令行接口](./06-cli.md) |
| 查归属、引用与许可 | [引用与声明](./07-references.md) |

## 怎么读这些文档

- **第一次用：** [安装](./01-installation.md)，然后跑 [快速开始](./02-quickstart.md)。
- **需要模型文件？** 用[模型转换](./03-conversion.md)在本地转换官方 HDF5 模型。
- **需要官方阈值与投票？** 读[官方模式](./04-official-mode.md)。
- **需要参数或函数签名？** 见[命令行接口](./06-cli.md)与 [Python API](./05-api.md)。

## 你需要自备

- 官方 NPClassifier HDF5 模型与 `index_v1.json`；在本地转换。
- 输入表或分子列表（SMILES 或 InChI）。
- 由转换器生成的 ONNX 模型目录。

## 作者与许可

- David Niteas `<david.ql.d@foxmail.com>`
- 封装代码使用 MIT；上游 NPClassifier 使用 MIT，官方模型/数据/ontology 使用 CC0。

**下一页：** [安装](./01-installation.md)
