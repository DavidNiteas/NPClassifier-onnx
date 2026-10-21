from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
import pytest

from NPClassifier_onnx import NPClassifier
from conftest import reference_output, write_index


def test_prediction_rows_and_status(synthetic_model_dir):
    model_dir, weights = synthetic_model_dir
    classifier = NPClassifier(model_dir, ["class"])

    frame = classifier.batch_predict_and_decode(
        ["CCO", "not-a-smiles", "c1ccccc1"], top_k=2
    )
    assert len(frame) == 3
    assert frame["status"].tolist() == ["ok", "invalid", "ok"]
    assert frame.loc[1, "class_model_prob_1"] != frame.loc[1, "class_model_prob_1"]  # NaN
    assert frame.loc[0, "class_model_label_1"] in {"Alpha", "Beta", "Gamma"}

    results, status = classifier.batch_predict(
        ["CCO", "not-a-smiles"], return_status=True
    )
    assert status == ["ok", "invalid"]
    assert results["class_model"].shape == (2, 3)
    assert np.all(results["class_model"][1] == 0)


def test_probabilities_match_onnx_and_numpy(synthetic_model_dir):
    model_dir, weights = synthetic_model_dir
    classifier = NPClassifier(model_dir, ["class"])
    formula, binary = classifier.molecules2fingerprints(["CCO", "c1ccccc1"])
    results = classifier.batch_predict(["CCO", "c1ccccc1"])["class_model"]
    expected = reference_output(formula, binary, weights)
    np.testing.assert_allclose(results, expected, rtol=1e-5, atol=1e-6)


def test_legacy_input_names_are_mapped_by_shape(synthetic_model_dir):
    model_dir, _ = synthetic_model_dir
    path = model_dir / "class_model.onnx"
    model = onnx.load(str(path))
    rename = {"input_2048": "inputs", "input_4096": "inputs_1"}
    for graph_input in model.graph.input:
        if graph_input.name in rename:
            graph_input.name = rename[graph_input.name]
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name in rename:
                node.input[index] = rename[name]
    onnx.save_model(model, str(path))

    classifier = NPClassifier(model_dir, ["class"])
    frame = classifier.batch_predict_and_decode(["CCO"], top_k=1)
    assert frame.loc[0, "status"] == "ok"
    assert frame.loc[0, "class_model_label_1"] in {"Alpha", "Beta", "Gamma"}


def test_top_k_validation(synthetic_model_dir):
    model_dir, _ = synthetic_model_dir
    classifier = NPClassifier(model_dir, ["class"])
    with pytest.raises(ValueError, match="larger than the number of classes"):
        classifier.batch_predict_and_decode(["CCO"], top_k=4)
    with pytest.raises(ValueError, match="top_k"):
        classifier.batch_predict_and_decode(["CCO"], top_k=0)


def test_table_file(synthetic_model_dir, tmp_path: Path):
    model_dir, _ = synthetic_model_dir
    csv_path = tmp_path / "molecules.csv"
    pd.DataFrame({"smiles": ["CCO", "bad", "c1ccccc1"]}).to_csv(csv_path, index=False)
    classifier = NPClassifier(model_dir, ["class"])
    frame = classifier.predict_from_table_file(
        csv_path, tag_column="smiles", batch_size=2, top_k=1, progress=False
    )
    assert len(frame) == 3
    assert frame.index.tolist() == ["CCO", "bad", "c1ccccc1"]
    assert frame["status"].tolist() == ["ok", "invalid", "ok"]


def test_loading_errors(synthetic_model_dir, tmp_path: Path):
    model_dir, _ = synthetic_model_dir
    with pytest.raises(FileNotFoundError, match="model directory"):
        NPClassifier(tmp_path / "missing", ["class"])

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="label mapping"):
        NPClassifier(empty_dir, ["class"])

    no_model_dir = tmp_path / "no_model"
    no_model_dir.mkdir()
    write_index(no_model_dir / "index_v1.json", classes=3)
    with pytest.raises(FileNotFoundError, match="missing class model"):
        NPClassifier(no_model_dir, ["class"])

    # The synthetic class model has three outputs; this index claims four.
    mismatch_dir = tmp_path / "mismatch"
    mismatch_dir.mkdir()
    (mismatch_dir / "class_model.onnx").write_bytes(
        (model_dir / "class_model.onnx").read_bytes()
    )
    write_index(mismatch_dir / "index_v1.json", classes=4)
    with pytest.raises(ValueError, match="classes but the label mapping"):
        NPClassifier(mismatch_dir, ["class"])


def test_load_mapping_ignores_hierarchy(tmp_path: Path):
    from conftest import write_index

    path = tmp_path / "index_v1.json"
    write_index(path, classes=3)
    mapping = NPClassifier.load_mapping(path)
    assert set(mapping) == {"class", "pathway", "superclass"}
    assert mapping["class"].iloc[0] == "Alpha"
