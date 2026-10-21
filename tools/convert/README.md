# Conversion tools

`convert_npc.py` builds ONNX models directly from the official Keras HDF5 files,
without TensorFlow. `verify_parity.py` compares a converted directory with a
trusted reference ONNX export.

```bash
pip install -r requirements.txt

python convert_npc.py \
  --hdf5-dir /path/to/hdf5 \
  --index /path/to/index_v1.json \
  --output-dir /path/to/model_dir \
  --verify

python verify_parity.py \
  --reference-dir /path/to/reference \
  --model-dir /path/to/model_dir
```

See [docs/conversion.md](../../docs/conversion.md) and
[中文转换指南](../../docs/conversion.zh.md) for the full guide.

The tools do not download or redistribute model weights. Output ONNX files are
ignored by Git and are not part of release artifacts.
