from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import onnx
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONVERT_DIR = ROOT / "tools" / "convert"
if str(CONVERT_DIR) not in sys.path:
    sys.path.insert(0, str(CONVERT_DIR))

import convert_npc  # noqa: E402


def make_synthetic_hdf5(
    path: Path,
    classes: int = 3,
    hidden: int = 4,
    seed: int = 0,
    input_names: Tuple[str, str] = ("input_2048", "input_4096"),
) -> Dict[str, np.ndarray]:
    """Create a tiny Keras HDF5 model that mirrors the NPClassifier MLP."""
    rng = np.random.default_rng(seed)
    input_one, input_two = input_names
    first_width = 2048
    second_width = 4096

    config = {
        "class_name": "Functional",
        "config": {
            "name": "synthetic_npclassifier",
            "layers": [
                {
                    "class_name": "InputLayer",
                    "name": input_one,
                    "config": {
                        "batch_input_shape": [None, first_width],
                        "dtype": "float32",
                    },
                },
                {
                    "class_name": "InputLayer",
                    "name": input_two,
                    "config": {
                        "batch_input_shape": [None, second_width],
                        "dtype": "float32",
                    },
                },
                {
                    "class_name": "Concatenate",
                    "name": "concatenate",
                    "config": {"axis": -1},
                },
                {
                    "class_name": "Dense",
                    "name": "dense",
                    "config": {
                        "units": hidden,
                        "activation": "relu",
                        "use_bias": True,
                    },
                },
                {
                    "class_name": "BatchNormalization",
                    "name": "batch_normalization",
                    "config": {
                        "axis": [1],
                        "epsilon": 0.001,
                        "momentum": 0.99,
                    },
                },
                {
                    "class_name": "Dropout",
                    "name": "dropout",
                    "config": {"rate": 0.1},
                },
                {
                    "class_name": "Dense",
                    "name": "output",
                    "config": {
                        "units": classes,
                        "activation": "sigmoid",
                        "use_bias": True,
                    },
                },
            ],
            "input_layers": [[input_one, 0, 0], [input_two, 0, 0]],
            "output_layers": [["output", 0, 0]],
        },
    }

    weights = {
        "dense_kernel": rng.normal(0, 0.05, (first_width + second_width, hidden)).astype(
            np.float32
        ),
        "dense_bias": rng.normal(0, 0.1, hidden).astype(np.float32),
        "gamma": (1.0 + rng.normal(0, 0.02, hidden)).astype(np.float32),
        "beta": rng.normal(0, 0.02, hidden).astype(np.float32),
        "mean": rng.normal(0, 0.02, hidden).astype(np.float32),
        "variance": (1.0 + rng.random(hidden)).astype(np.float32),
        "output_kernel": rng.normal(0, 0.1, (hidden, classes)).astype(np.float32),
        "output_bias": rng.normal(0, 0.1, classes).astype(np.float32),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["keras_version"] = "2.4.0"
        handle.attrs["backend"] = "tensorflow"
        handle.attrs["model_config"] = json.dumps(config)
        model_weights = handle.create_group("model_weights")

        dense = model_weights.create_group("dense").create_group("dense")
        dense.create_dataset("kernel:0", data=weights["dense_kernel"])
        dense.create_dataset("bias:0", data=weights["dense_bias"])

        batch_norm = model_weights.create_group("batch_normalization").create_group(
            "batch_normalization"
        )
        for parameter in ("gamma", "beta", "moving_mean", "moving_variance"):
            key = {
                "moving_mean": "mean",
                "moving_variance": "variance",
            }.get(parameter, parameter)
            batch_norm.create_dataset("%s:0" % parameter, data=weights[key])

        output = model_weights.create_group("output").create_group("output")
        output.create_dataset("kernel:0", data=weights["output_kernel"])
        output.create_dataset("bias:0", data=weights["output_bias"])

    return weights


def write_index(path: Path, classes: int = 3) -> None:
    labels = ["Alpha", "Beta", "Gamma", "Delta"]
    index = {
        "Class": {labels[i]: i for i in range(classes)},
        "Pathway": {"Pathway A": 0, "Pathway B": 1},
        "Superclass": {"Superclass A": 0, "Superclass B": 1},
        "Class_hierarchy": {
            str(i): {
                "Pathway": [0 if i < 2 else 1],
                "Superclass": [0 if i < 2 else 1],
            }
            for i in range(classes)
        },
        "Super_hierarchy": {
            "0": {"Pathway": [0]},
            "1": {"Pathway": [1]},
        },
    }
    path.write_text(json.dumps(index), encoding="utf-8")


def reference_output(
    formula: np.ndarray,
    binary: np.ndarray,
    weights: Dict[str, np.ndarray],
) -> np.ndarray:
    hidden = np.concatenate([formula, binary], axis=1) @ weights["dense_kernel"]
    hidden = hidden + weights["dense_bias"]
    hidden = np.maximum(hidden, 0.0)
    hidden = (
        (hidden - weights["mean"])
        / np.sqrt(weights["variance"] + 0.001)
        * weights["gamma"]
        + weights["beta"]
    )
    logits = hidden @ weights["output_kernel"] + weights["output_bias"]
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


@pytest.fixture
def synthetic_hdf5(tmp_path: Path):
    hdf5_path = tmp_path / convert_npc.EXPECTED_HDF5["class"]
    weights = make_synthetic_hdf5(hdf5_path, classes=3)
    return hdf5_path, weights


@pytest.fixture
def synthetic_model_dir(tmp_path: Path, synthetic_hdf5):
    hdf5_path, weights = synthetic_hdf5
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    proto, summary = convert_npc.build_onnx_model(hdf5_path)
    assert summary["classes"] == 3
    onnx.save_model(proto, str(model_dir / "class_model.onnx"))
    write_index(model_dir / "index_v1.json", classes=3)
    return model_dir, weights


@pytest.fixture
def synthetic_official_model_dir(tmp_path: Path):
    model_dir = tmp_path / "official_models"
    model_dir.mkdir()
    weights = {}
    for model_name, classes in (
        ("class", 3),
        ("pathway", 2),
        ("superclass", 2),
    ):
        hdf5_path = tmp_path / ("%s.hdf5" % model_name)
        weights[model_name] = make_synthetic_hdf5(
            hdf5_path, classes=classes, seed=classes + 10
        )
        proto, summary = convert_npc.build_onnx_model(hdf5_path)
        assert summary["classes"] == classes
        onnx.save_model(proto, str(model_dir / ("%s_model.onnx" % model_name)))
    write_index(model_dir / "index_v1.json", classes=3)
    return model_dir
