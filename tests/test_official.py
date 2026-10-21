from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from NPClassifier_onnx import NPClassifier
from NPClassifier_onnx.official import (
    OFFICIAL_THRESHOLDS,
    is_glycoside,
    vote_classification,
)
from NPClassifier_onnx import cli


ONTOLOGY = {
    "Class": {"C0": 0, "C1": 1, "C2": 2},
    "Superclass": {"S0": 0, "S1": 1},
    "Pathway": {"P0": 0, "P1": 1},
    "Class_hierarchy": {
        "0": {"Pathway": [0], "Superclass": [0]},
        "1": {"Pathway": [0], "Superclass": [0]},
        "2": {"Pathway": [1], "Superclass": [1]},
    },
    "Super_hierarchy": {
        "0": {"Pathway": [0]},
        "1": {"Pathway": [1]},
    },
}


def test_official_thresholds_and_glycoside():
    assert OFFICIAL_THRESHOLDS == {"class": 0.1, "superclass": 0.3, "pathway": 0.5}
    assert is_glycoside("OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O") is True
    assert is_glycoside("CCO") is False
    assert is_glycoside("not-a-smiles") == "Input_error"


def test_vote_classification_three_votes():
    pathways, superclasses, classes, glycoside = vote_classification(
        n_path=[0],
        n_class=[0, 1],
        n_super=[0],
        pred_class=np.array([0.9, 0.8, 0.1]),
        pred_super=np.array([0.9, 0.1]),
        path_from_class=[0],
        path_from_superclass=[0],
        isglycoside=False,
        ontology=ONTOLOGY,
    )
    assert pathways == ["P0"]
    assert superclasses == ["S0"]
    assert classes == ["C0", "C1"]
    assert glycoside is False


def test_vote_classification_two_votes_and_no_path():
    pathways, superclasses, classes, _ = vote_classification(
        n_path=[0],
        n_class=[0],
        n_super=[],
        pred_class=np.array([0.9, 0.1, 0.1]),
        pred_super=np.array([0.9, 0.1]),
        path_from_class=[0],
        path_from_superclass=[],
        isglycoside=False,
        ontology=ONTOLOGY,
    )
    assert pathways == ["P0"]
    assert superclasses == ["S0"]
    assert classes == ["C0"]

    pathways, superclasses, classes, _ = vote_classification(
        n_path=[1],
        n_class=[],
        n_super=[],
        pred_class=np.array([0.1, 0.1, 0.1]),
        pred_super=np.array([0.1, 0.1]),
        path_from_class=[],
        path_from_superclass=[],
        isglycoside=False,
        ontology=ONTOLOGY,
    )
    assert pathways == ["P1"]
    assert superclasses == []
    assert classes == []


def test_official_threshold_override(synthetic_official_model_dir):
    classifier = NPClassifier(synthetic_official_model_dir)
    frame = classifier.predict_official(
        ["CCO"], thresholds={"class": 0.0, "superclass": 0.0, "pathway": 0.0}
    )
    assert frame.loc[0, "status"] == "ok"
    assert isinstance(frame.loc[0, "class_results"], list)
    with pytest.raises(ValueError, match="unknown threshold"):
        classifier.predict_official(["CCO"], thresholds={"nope": 0.1})


def test_official_prediction_api(synthetic_official_model_dir):
    classifier = NPClassifier(synthetic_official_model_dir)
    frame = classifier.predict_official(["CCO", "not-a-smiles"])
    assert list(frame.columns) == [
        "status",
        "isglycoside",
        "pathway_results",
        "superclass_results",
        "class_results",
    ]
    assert frame.loc[0, "status"] == "ok"
    assert frame.loc[1, "status"] == "invalid"
    assert frame.loc[1, "isglycoside"] is None
    assert isinstance(frame.loc[0, "pathway_results"], list)


def test_official_cli(synthetic_official_model_dir, tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "official.csv"
    pd.DataFrame({"smiles": ["CCO"]}).to_csv(input_path, index=False)
    code = cli.main(
        [
            "--model-dir",
            str(synthetic_official_model_dir),
            "--input",
            str(input_path),
            "--mode",
            "official",
            "--output",
            str(output_path),
            "--quiet",
        ]
    )
    assert code == 0
    frame = pd.read_csv(output_path, index_col=0)
    assert list(frame.columns) == [
        "status",
        "isglycoside",
        "pathway_results",
        "superclass_results",
        "class_results",
    ]
