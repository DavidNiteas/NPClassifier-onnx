# 引用与声明

**中文** | [English](../en/references.md)

## 本项目与官方 NPClassifier 的关系

`NPClassifier-onnx` 是独立的第三方推理封装，**不是** NPClassifier 的官方版本，也不由 NPClassifier 作者维护或背书。它保留官方的 Morgan 计数指纹、标签解码、阈值与层级投票逻辑，但把推理后端换成了 ONNX Runtime。

如果你需要官方实现、官方 Web 服务、官方模型发布或官方支持，请前往：

- 官方仓库：<https://github.com/mwang87/NP-Classifier>
- 官方在线服务：<https://npclassifier.ucsd.edu/>

## 论文引用

如果研究中使用 NPClassifier 或其分类结果，请引用原始论文：

```bibtex
@article{kim2021npclassifier,
  title   = {NPClassifier: A Deep Neural Network-Based Structural Classification Tool for Natural Products},
  author  = {Kim, Hyun Woo and Wang, Mingxun and Leber, Christopher A. and others},
  journal = {Journal of Natural Products},
  volume  = {84},
  number  = {11},
  pages   = {2795--2807},
  year    = {2021},
  doi     = {10.1021/acs.jnatprod.1c00399}
}
```

- DOI：<https://doi.org/10.1021/acs.jnatprod.1c00399>
- 标题：*NPClassifier: A Deep Neural Network-Based Structural Classification Tool for Natural Products*
- 期刊：*Journal of Natural Products* **2021**, 84, 2795–2807

## 许可证

- 本项目的包装与推理代码使用 **MIT License**，完整文本见仓库根目录的 [`LICENSE`](../../LICENSE)。
- 上游 [NPClassifier](https://github.com/mwang87/NP-Classifier) 软件使用 MIT License。
- 官方模型、数据与 ontology 使用 **CC0**。本仓库不分发这些权重或数据；转换得到的 ONNX 文件由用户自行从官方 HDF5 生成，再分发时请遵守其来源页面的条款并保留上游署名。
