from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

import convert_npc
from conftest import make_synthetic_hdf5, reference_output, write_index


def _run(path: Path, formula: np.ndarray, binary: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path))
    feeds = {}
    for meta in session.get_inputs():
        size = int(meta.shape[-1])
        feeds[meta.name] = formula if size == 2048 else binary
    return session.run(None, feeds)[0]


def test_build_onnx_matches_numpy_reference(synthetic_hdf5, tmp_path: Path):
    hdf5_path, weights = synthetic_hdf5
    proto, summary = convert_npc.build_onnx_model(hdf5_path)
    assert summary["classes"] == 3
    onnx.checker.check_model(proto)
    output_path = tmp_path / "class_model.onnx"
    onnx.save_model(proto, str(output_path))

    rng = np.random.default_rng(1)
    formula = rng.random((5, 2048)).astype(np.float32)
    binary = rng.random((5, 4096)).astype(np.float32)
    actual = _run(output_path, formula, binary)
    expected = reference_output(formula, binary, weights)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_convert_models_writes_index_and_manifest(synthetic_hdf5, tmp_path: Path):
    hdf5_path, _ = synthetic_hdf5
    index_path = tmp_path / "index_v1.json"
    write_index(index_path, classes=3)
    output_dir = tmp_path / "converted"

    manifest_path = convert_npc.convert_models(
        hdf5_dir=tmp_path,
        output_dir=output_dir,
        index_path=index_path,
        models=["class"],
    )

    assert (output_dir / "class_model.onnx").is_file()
    assert (output_dir / "index_v1.json").is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["opset"] == 13
    assert manifest["models"]["class"]["classes"] == 3
    assert manifest["source"]["class"]["sha256"]
    assert manifest["index"]["sha256"]

    with pytest.raises(FileExistsError):
        convert_npc.convert_models(
            hdf5_dir=tmp_path,
            output_dir=output_dir,
            index_path=index_path,
            models=["class"],
        )

    convert_npc.convert_models(
        hdf5_dir=tmp_path,
        output_dir=output_dir,
        index_path=index_path,
        models=["class"],
        force=True,
    )


def test_index_class_count_mismatch_is_rejected(synthetic_hdf5, tmp_path: Path):
    hdf5_path, _ = synthetic_hdf5
    index_path = tmp_path / "index_v1.json"
    write_index(index_path, classes=4)
    with pytest.raises(ValueError, match="does not belong together|classes"):
        convert_npc.convert_models(
            hdf5_dir=tmp_path,
            output_dir=tmp_path / "converted",
            index_path=index_path,
            models=["class"],
        )


def test_unsupported_layer_is_rejected(tmp_path: Path):
    hdf5_path = tmp_path / "bad.hdf5"
    make_synthetic_hdf5(hdf5_path)
    import h5py

    with h5py.File(hdf5_path, "r+") as handle:
        config = json.loads(handle.attrs["model_config"])
        config["config"]["layers"].insert(
            3, {"class_name": "LSTM", "name": "lstm", "config": {"units": 2}}
        )
        handle.attrs["model_config"] = json.dumps(config)

    with pytest.raises(ValueError, match="unsupported Keras layer"):
        convert_npc.build_onnx_model(hdf5_path)
