#!/usr/bin/env python3
"""Convert official NPClassifier Keras HDF5 weights to ONNX.

The official NPClassifier models are small fully-connected networks.  Instead of
requiring TensorFlow (the upstream converter was written for TensorFlow 2.3),
this script reads the Keras ``model_config`` and weights directly from HDF5 and
builds an equivalent ONNX graph with NumPy and ONNX.

Supported layers: InputLayer, Concatenate, Dense, BatchNormalization, Dropout
and Activation.  The official V1 models use only these layers.

The script does not download anything and does not include model weights.  Point
``--hdf5-dir`` at the directory containing the official HDF5 files and pass the
``index_v1.json`` from the upstream source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MODEL_NAMES: Tuple[str, ...] = ("class", "pathway", "superclass")
EXPECTED_HDF5 = {
    "class": "NP_classifier_class_V1.hdf5",
    "pathway": "NP_classifier_pathway_V1.hdf5",
    "superclass": "NP_classifier_superclass_V1.hdf5",
}
ACTIVATIONS = {
    "relu": "Relu",
    "sigmoid": "Sigmoid",
    "softmax": "Softmax",
    "tanh": "Tanh",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_config(handle: h5py.File) -> Tuple[dict, dict]:
    raw = handle.attrs.get("model_config")
    if raw is None:
        raise ValueError("HDF5 has no model_config attribute; is this a Keras model?")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    config = json.loads(raw)
    if not isinstance(config, dict) or "config" not in config:
        raise ValueError("unsupported Keras model_config structure")
    return config, config["config"]


def _weight(handle: h5py.File, layer: str, parameter: str) -> np.ndarray:
    candidates = (
        "model_weights/%s/%s/%s:0" % (layer, layer, parameter),
        "model_weights/%s/%s:0" % (layer, parameter),
    )
    for candidate in candidates:
        if candidate in handle:
            return np.asarray(handle[candidate][()])
    raise KeyError(
        "missing weight %r for layer %r; available names: %s"
        % (parameter, layer, ", ".join(sorted(handle["model_weights"].keys())))
    )


def _initializer(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(array, dtype=np.float32), name=name)


def _activation_node(
    activation: Optional[str], source: str, target: str
) -> onnx.NodeProto:
    activation = (activation or "linear").lower()
    if activation in ("linear", "identity", None):
        return helper.make_node("Identity", [source], [target])
    if activation in ACTIVATIONS:
        kwargs = {"axis": 1} if activation == "softmax" else {}
        return helper.make_node(ACTIVATIONS[activation], [source], [target], **kwargs)
    raise ValueError("unsupported activation %r" % activation)


def _input_shape(layer_config: dict) -> List[Optional[int]]:
    shape = layer_config.get("batch_input_shape") or layer_config.get("batch_shape")
    if shape is None:
        raise ValueError("input layer has no batch_input_shape")
    dims = [None if item is None else int(item) for item in shape]
    if len(dims) != 2 or dims[-1] is None:
        raise ValueError(
            "only 2-D fixed-width inputs are supported, got shape %r" % (shape,)
        )
    return [None, dims[-1]]


def _output_name(model_config: dict, fallback: str) -> str:
    output_layers = model_config.get("output_layers")
    if output_layers:
        first = output_layers[0]
        if isinstance(first, (list, tuple)) and first:
            return str(first[0])
        if isinstance(first, str):
            return first
    return fallback


def build_onnx_model(
    hdf5_path: Path,
    opset: int = 13,
) -> Tuple[onnx.ModelProto, Dict[str, int]]:
    """Build an ONNX model from one official NPClassifier HDF5 file.

    Returns the model and a small summary (``classes`` and ``inputs``).
    """
    hdf5_path = Path(hdf5_path)
    with h5py.File(hdf5_path, "r") as handle:
        _, model_config = _read_config(handle)
        layers = model_config.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError("Keras config has no layers")

        nodes: List[onnx.NodeProto] = []
        initializers: List[onnx.TensorProto] = []
        graph_inputs: List[onnx.ValueInfoProto] = []
        current_tensors: List[str] = []
        input_feature_dims: List[int] = []
        last_dim: Optional[int] = None
        produced: Dict[str, bool] = {}
        seen_layers: List[str] = []

        for layer in layers:
            class_name = layer.get("class_name")
            layer_name = str(layer.get("name"))
            config = layer.get("config", {})
            seen_layers.append("%s(%s)" % (class_name, layer_name))

            if class_name == "InputLayer":
                shape = _input_shape(config)
                last_dim = int(shape[1])
                input_feature_dims.append(last_dim)
                graph_inputs.append(
                    helper.make_tensor_value_info(layer_name, TensorProto.FLOAT, shape)
                )
                current_tensors.append(layer_name)
                produced[layer_name] = True
                continue

            if class_name == "Concatenate":
                axis = int(config.get("axis", -1))
                if axis < 0:
                    axis += 2
                if axis != 1:
                    raise ValueError(
                        "unsupported Concatenate axis %r in layer %r; expected "
                        "the feature axis" % (config.get("axis"), layer_name)
                    )
                if len(graph_inputs) < 2:
                    raise ValueError(
                        "Concatenate layer %r requires at least two inputs" % layer_name
                    )
                input_names = [item.name for item in graph_inputs]
                nodes.append(
                    helper.make_node("Concat", input_names, [layer_name], axis=1)
                )
                last_dim = sum(input_feature_dims)
                current_tensors = [layer_name]
                produced[layer_name] = True
                continue

            if len(current_tensors) != 1:
                raise ValueError(
                    "unexpected tensor state before layer %r (%s); only a single "
                    "data tensor is supported" % (layer_name, class_name)
                )
            source = current_tensors[0]

            if class_name == "Dense":
                units = int(config["units"])
                kernel = _weight(handle, layer_name, "kernel")
                if kernel.ndim != 2 or kernel.shape[1] != units:
                    raise ValueError(
                        "unexpected kernel shape %r for %r; expected (features, %d)"
                        % (kernel.shape, layer_name, units)
                    )
                if last_dim is not None and kernel.shape[0] != last_dim:
                    raise ValueError(
                        "kernel for %r expects %d features, previous layer has %d"
                        % (layer_name, kernel.shape[0], last_dim)
                    )
                kernel_name = "%s_kernel" % layer_name
                initializers.append(_initializer(kernel_name, kernel))
                nodes.append(
                    helper.make_node(
                        "MatMul", [source, kernel_name], ["%s_matmul" % layer_name]
                    )
                )
                dense_output = "%s_add" % layer_name
                use_bias = bool(config.get("use_bias", True))
                if use_bias:
                    bias = _weight(handle, layer_name, "bias")
                    if bias.shape != (units,):
                        raise ValueError(
                            "unexpected bias shape %r for %r" % (bias.shape, layer_name)
                        )
                    bias_name = "%s_bias" % layer_name
                    initializers.append(_initializer(bias_name, bias))
                    nodes.append(
                        helper.make_node(
                            "Add", ["%s_matmul" % layer_name, bias_name], [dense_output]
                        )
                    )
                else:
                    nodes.append(
                        helper.make_node(
                            "Identity", ["%s_matmul" % layer_name], [dense_output]
                        )
                    )
                nodes.append(
                    _activation_node(config.get("activation"), dense_output, layer_name)
                )
                last_dim = units
                current_tensors = [layer_name]
                produced[layer_name] = True
                continue

            if class_name == "BatchNormalization":
                axis = config.get("axis", -1)
                if isinstance(axis, (list, tuple)):
                    if len(axis) != 1:
                        raise ValueError(
                            "unsupported BatchNormalization axis %r in %r"
                            % (axis, layer_name)
                        )
                    axis = axis[0]
                if int(axis) not in (1, -1):
                    raise ValueError(
                        "unsupported BatchNormalization axis %r in %r"
                        % (axis, layer_name)
                    )

                gamma = _weight(handle, layer_name, "gamma")
                beta = _weight(handle, layer_name, "beta")
                mean = _weight(handle, layer_name, "moving_mean")
                variance = _weight(handle, layer_name, "moving_variance")
                for parameter, array in (
                    ("gamma", gamma),
                    ("beta", beta),
                    ("mean", mean),
                    ("variance", variance),
                ):
                    if array.shape != (last_dim,):
                        raise ValueError(
                            "unexpected %s shape %r for %r; expected (%d,)"
                            % (parameter, array.shape, layer_name, last_dim)
                        )
                names = {}
                for parameter, array in (
                    ("gamma", gamma),
                    ("beta", beta),
                    ("mean", mean),
                    ("variance", variance),
                ):
                    name = "%s_%s" % (layer_name, parameter)
                    names[parameter] = name
                    initializers.append(_initializer(name, array))
                nodes.append(
                    helper.make_node(
                        "BatchNormalization",
                        [
                            source,
                            names["gamma"],
                            names["beta"],
                            names["mean"],
                            names["variance"],
                        ],
                        [layer_name],
                        epsilon=float(config.get("epsilon", 1e-3)),
                    )
                )
                current_tensors = [layer_name]
                produced[layer_name] = True
                continue

            if class_name == "Dropout":
                nodes.append(helper.make_node("Identity", [source], [layer_name]))
                current_tensors = [layer_name]
                produced[layer_name] = True
                continue

            if class_name == "Activation":
                nodes.append(
                    _activation_node(config.get("activation"), source, layer_name)
                )
                current_tensors = [layer_name]
                produced[layer_name] = True
                continue

            raise ValueError(
                "unsupported Keras layer %r (%s); supported layers are "
                "InputLayer, Concatenate, Dense, BatchNormalization, Dropout, "
                "Activation" % (layer_name, class_name)
            )

        if len(current_tensors) != 1:
            raise ValueError("model must end with exactly one data tensor")
        final_tensor = current_tensors[0]
        output_name = _output_name(model_config, final_tensor)
        if output_name not in produced:
            # Keras sometimes names the output tensor rather than the layer.
            output_name = final_tensor
            nodes.append(helper.make_node("Identity", [final_tensor], ["model_output"]))
            output_name = "model_output"
        output_shape = [None, last_dim]
        graph_outputs = [
            helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)
        ]

        graph = helper.make_graph(
            nodes,
            "npclassifier_%s" % hdf5_path.stem,
            graph_inputs,
            graph_outputs,
            initializers,
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", int(opset))],
            producer_name="npclassifier-onnx-hdf5-converter",
        )
        model.ir_version = 8
        model.metadata_props.append(
            onnx.StringStringEntryProto(key="source_format", value="keras-hdf5")
        )
        model.metadata_props.append(
            onnx.StringStringEntryProto(key="source_file", value=hdf5_path.name)
        )
        keras_version = handle.attrs.get("keras_version", "")
        if isinstance(keras_version, bytes):
            keras_version = keras_version.decode("utf-8", errors="replace")
        model.metadata_props.append(
            onnx.StringStringEntryProto(key="keras_version", value=str(keras_version))
        )
        model.metadata_props.append(
            onnx.StringStringEntryProto(
                key="converter", value="npclassifier-onnx/NPClassifier_onnx/convert.py"
            )
        )
        model.metadata_props.append(
            onnx.StringStringEntryProto(key="layers", value=", ".join(seen_layers))
        )

    onnx.checker.check_model(model)
    if last_dim is None:
        raise ValueError("could not determine the number of output classes")
    return model, {"classes": int(last_dim), "inputs": len(graph_inputs)}


def _find_hdf5(hdf5_dir: Path, model: str) -> Path:
    expected = hdf5_dir / EXPECTED_HDF5[model]
    if expected.is_file():
        return expected
    for path in sorted(hdf5_dir.glob("*.hdf5")):
        if path.name.lower() == EXPECTED_HDF5[model].lower():
            return path
    available = ", ".join(path.name for path in sorted(hdf5_dir.glob("*.hdf5")))
    raise FileNotFoundError(
        "missing %s weights: expected %s in %s (found: %s)"
        % (model, EXPECTED_HDF5[model], hdf5_dir, available or "none")
    )


def _load_index(index_path: Path) -> Dict[str, dict]:
    with index_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("index file must be a JSON object")
    result: Dict[str, dict] = {}
    for key, value in raw.items():
        if str(key).lower() in MODEL_NAMES and isinstance(value, dict):
            result[str(key).lower()] = value
    if not result:
        raise ValueError("index file has no Class/Pathway/Superclass entries")
    return result


def _write_manifest(
    output_dir: Path,
    sources: Dict[str, Path],
    outputs: Dict[str, Path],
    classes: Dict[str, int],
    index_path: Path,
    index_output: Path,
    opset: int,
) -> Path:
    manifest = {
        "format_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "converter": "npclassifier-onnx/NPClassifier_onnx/convert.py",
        "opset": int(opset),
        "source": {},
        "models": {},
        "index": {
            "file": index_output.name,
            "sha256": sha256(index_output),
            "bytes": index_output.stat().st_size,
        },
        "license": {
            "code": "MIT",
            "official_models": "CC0",
            "note": "ONNX files are generated from the official weights and are not distributed by this repository.",
        },
    }
    for model in MODEL_NAMES:
        if model not in sources:
            continue
        manifest["source"][model] = {
            "file": sources[model].name,
            "sha256": sha256(sources[model]),
            "bytes": sources[model].stat().st_size,
        }
        manifest["models"][model] = {
            "file": outputs[model].name,
            "sha256": sha256(outputs[model]),
            "bytes": outputs[model].stat().st_size,
            "classes": classes[model],
        }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def convert_models(
    hdf5_dir: Path,
    output_dir: Path,
    index_path: Path,
    models: Sequence[str] = MODEL_NAMES,
    opset: int = 13,
    force: bool = False,
    explicit_sources: Optional[Dict[str, Path]] = None,
) -> Path:
    """Convert one or more official models and write ``manifest.json``."""
    hdf5_dir = Path(hdf5_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    index_path = Path(index_path).expanduser()
    if not hdf5_dir.is_dir():
        raise NotADirectoryError("weights directory does not exist: %s" % hdf5_dir)
    if not index_path.is_file():
        raise FileNotFoundError("index file does not exist: %s" % index_path)
    if int(opset) < 11:
        raise ValueError("opset must be at least 11")

    normalised: List[str] = []
    for model in models:
        name = str(model).strip().lower()
        if name not in MODEL_NAMES:
            raise ValueError(
                "unknown model %r; choose from %s" % (model, ", ".join(MODEL_NAMES))
            )
        if name not in normalised:
            normalised.append(name)
    if not normalised:
        raise ValueError("at least one model is required")

    index = _load_index(index_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources: Dict[str, Path] = {}
    outputs: Dict[str, Path] = {}
    classes: Dict[str, int] = {}
    explicit_sources = explicit_sources or {}

    for model in normalised:
        source = explicit_sources.get(model)
        if source is None:
            source = _find_hdf5(hdf5_dir, model)
        source = Path(source).expanduser()
        if not source.is_file():
            raise FileNotFoundError("weights file does not exist: %s" % source)
        if model not in index:
            raise ValueError(
                "index file lacks a %r entry; use the index_v1.json from the "
                "official source tree" % model
            )
        output = output_dir / ("%s_model.onnx" % model)
        if output.exists() and not force:
            raise FileExistsError(
                "%s already exists; pass --force to overwrite" % output
            )
        proto, summary = build_onnx_model(source, opset=opset)
        expected_classes = len(index[model])
        if summary["classes"] != expected_classes:
            raise ValueError(
                "%s outputs %d classes but the index has %d; the HDF5 and index "
                "do not belong together"
                % (source.name, summary["classes"], expected_classes)
            )
        onnx.save_model(proto, str(output))
        sources[model] = source
        outputs[model] = output
        classes[model] = summary["classes"]
        print(
            "converted %s -> %s (%d classes, %.1f MiB)"
            % (
                source.name,
                output.name,
                summary["classes"],
                output.stat().st_size / 1048576,
            )
        )

    index_output = output_dir / "index_v1.json"
    if index_path.resolve() != index_output.resolve():
        shutil.copyfile(index_path, index_output)
    manifest = _write_manifest(
        output_dir, sources, outputs, classes, index_path, index_output, opset
    )
    print("wrote %s" % manifest)
    return manifest


def _verify_outputs(output_dir: Path, models: Sequence[str]) -> None:
    import onnxruntime as ort

    for model in models:
        path = output_dir / ("%s_model.onnx" % model)
        session = ort.InferenceSession(str(path))
        feeds = {}
        for input_meta in session.get_inputs():
            size = int(input_meta.shape[-1])
            feeds[input_meta.name] = np.zeros((2, size), dtype=np.float32)
        output = session.run(None, feeds)[0]
        print(
            "verified %s: inputs=%s output=%s"
            % (path.name, sorted(feeds), output.shape)
        )


def _parse_models(value: str) -> List[str]:
    models = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [item for item in models if item not in MODEL_NAMES]
    if unknown:
        raise argparse.ArgumentTypeError("unknown model(s): %s" % ", ".join(unknown))
    if not models:
        raise argparse.ArgumentTypeError("at least one model is required")
    return [name for name in MODEL_NAMES if name in models]


def _default_index(hdf5_dir: Path) -> Optional[Path]:
    candidates = (
        hdf5_dir / "index_v1.json",
        hdf5_dir.parent / "index_v1.json",
        hdf5_dir.parent / "dict" / "index_v1.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="convert_npc",
        description=(
            "Convert official NPClassifier Keras HDF5 models to ONNX without "
            "TensorFlow. No weights are downloaded or distributed."
        ),
    )
    parser.add_argument(
        "--hdf5-dir",
        required=True,
        help="directory containing NP_classifier_*_V1.hdf5",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory for <model>_model.onnx, index_v1.json and manifest.json",
    )
    parser.add_argument(
        "--index",
        default=None,
        help="path to the official index_v1.json (auto-detected when omitted)",
    )
    parser.add_argument(
        "--models",
        type=_parse_models,
        default=list(MODEL_NAMES),
        help="comma-separated models: class,pathway,superclass (default: all)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version (default: 13)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing ONNX files",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="run the converted models once with ONNX Runtime",
    )
    args = parser.parse_args(argv)

    hdf5_dir = Path(args.hdf5_dir).expanduser()
    index_path = (
        Path(args.index).expanduser() if args.index else _default_index(hdf5_dir)
    )
    if index_path is None:
        print(
            "convert_npc: error: no index_v1.json found; pass --index "
            "path/to/index_v1.json (it is part of the official source tree)",
            file=sys.stderr,
        )
        return 2

    try:
        convert_models(
            hdf5_dir=hdf5_dir,
            output_dir=Path(args.output_dir).expanduser(),
            index_path=index_path,
            models=args.models,
            opset=args.opset,
            force=args.force,
        )
        if args.verify:
            _verify_outputs(Path(args.output_dir).expanduser(), args.models)
    except (
        FileNotFoundError,
        NotADirectoryError,
        FileExistsError,
        ValueError,
        KeyError,
        OSError,
    ) as exc:
        print("convert_npc: error: %s" % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
