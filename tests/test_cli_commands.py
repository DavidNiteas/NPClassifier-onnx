from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from NPClassifier_onnx import cli
from conftest import write_index


def _write_input(path: Path) -> Path:
    pd.DataFrame({"smiles": ["CCO", "bad", "c1ccccc1"]}).to_csv(path, index=False)
    return path


def test_predict_subcommand(synthetic_model_dir, tmp_path: Path):
    model_dir, _ = synthetic_model_dir
    output = tmp_path / "out.csv"
    code = cli.main(
        [
            "predict",
            "--model-dir",
            str(model_dir),
            "--input",
            str(_write_input(tmp_path / "input.csv")),
            "--model-set",
            "class",
            "--output",
            str(output),
            "--quiet",
        ]
    )
    assert code == 0
    frame = pd.read_csv(output, index_col=0)
    assert list(frame["status"]) == ["ok", "invalid", "ok"]


def test_convert_subcommand(synthetic_hdf5, tmp_path: Path):
    _, _ = synthetic_hdf5
    write_index(tmp_path / "index_v1.json", classes=3)
    output_dir = tmp_path / "converted"
    code = cli.main(
        [
            "convert",
            "--hdf5-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--models",
            "class",
            "--quiet",
        ]
    )
    assert code == 0
    assert (output_dir / "class_model.onnx").is_file()
    assert (output_dir / "index_v1.json").is_file()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["models"]["class"]["classes"] == 3


def test_verify_subcommand(synthetic_model_dir, tmp_path: Path):
    model_dir, _ = synthetic_model_dir
    code = cli.main(
        [
            "verify",
            "--reference-dir",
            str(model_dir),
            "--model-dir",
            str(model_dir),
            "--models",
            "class",
            "--quiet",
        ]
    )
    assert code == 0


def test_doctor_json(synthetic_model_dir, tmp_path: Path, capsys):
    model_dir, _ = synthetic_model_dir
    code = cli.main(
        [
            "doctor",
            "--model-dir",
            str(model_dir),
            "--models",
            "class",
            "--json",
        ]
    )
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["version"]
    assert any(
        item["name"] == "fingerprint" and item["ok"] for item in report["checks"]
    )
    assert report["models"]["class_model"]["outputs"]


def test_help_lists_subcommands(capsys):
    code = cli.main(["--help"])
    assert code == 0
    output = capsys.readouterr().out
    for command in ("predict", "convert", "verify", "doctor"):
        assert command in output
