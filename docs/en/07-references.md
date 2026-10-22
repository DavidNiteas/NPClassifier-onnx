# References and notices

[中文](../zh-CN/07-references.md) | **English** · [Documentation home](../../README.md#english)

**Level:** 3 — Reference

**Prerequisites:** [Official mode](./04-official-mode.md).

## Relationship to official NPClassifier

`NPClassifier-onnx` is an independent third-party inference wrapper. It is
**not** the official NPClassifier release, and it is not maintained or endorsed
by the NPClassifier authors. It keeps the official Morgan-count fingerprints,
label decoding, thresholds and hierarchical voting, but uses ONNX Runtime as the
inference backend.

If you need the official implementation, web service, model release or support,
use the upstream project:

- Official repository: <https://github.com/mwang87/NP-Classifier>
- Official web service: <https://npclassifier.ucsd.edu/>

## Citation

If you use NPClassifier or its classifications in research, please cite the
original paper:

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

- DOI: <https://doi.org/10.1021/acs.jnatprod.1c00399>
- Title: *NPClassifier: A Deep Neural Network-Based Structural Classification Tool for Natural Products*
- Journal: *Journal of Natural Products* **2021**, 84, 2795–2807

## License

- The wrapper and inference code in this repository are released under the
  **MIT License**; see [`LICENSE`](../../LICENSE) at the repository root.
- Upstream [NPClassifier](https://github.com/mwang87/NP-Classifier) software is
  MIT licensed.
- The official models, data and ontology are **CC0**. This repository does not
  distribute those weights or data. ONNX files are generated locally by users
  from the official HDF5 files; follow the source terms and keep upstream
  attribution when redistributing a conversion.

**Next:** [Why code-only](./08-why-code-only.md)
