from __future__ import annotations

from pathlib import Path

import pandas as pd

from NPClassifier_onnx import cli


def test_cli_writes_results(synthetic_model_dir, tmp_path: Path, capsys):
    model_dir, _ = synthetic_model_dir
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "results.csv"
    pd.DataFrame({"smiles": ["CCO", "bad", "c1ccccc1"]}).to_csv(
        input_path, index=False
    )

    code = cli.main(
        [
            "--model-dir",
            str(model_dir),
            "--input",
            str(input_path),
            "--column",
            "smiles",
            "--model-set",
            "class",
            "--batch-size",
            "2",
            "--top-k",
            "1",
            "--output",
            str(output_path),
            "--quiet",
        ]
    )
    assert code == 0
    frame = pd.read_csv(output_path, index_col=0)
    assert len(frame) == 3
    assert list(frame["status"]) == ["ok", "invalid", "ok"]
    assert "class_model_prob_1" in frame.columns
    assert "class_model_label_1" in frame.columns


def test_cli_reports_missing_model(synthetic_model_dir, tmp_path: Path, capsys):
    model_dir, _ = synthetic_model_dir
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"smiles": ["CCO"]}).to_csv(input_path, index=False)

    code = cli.main(
        [
            "--model_dir",
            str(model_dir),
            "--input_file",
            str(input_path),
            "--tag_column",
            "smiles",
            "--output",
            str(tmp_path / "out.csv"),
            "--quiet",
        ]
    )
    assert code == 2
    captured = capsys.readouterr()
    assert "missing pathway model" in captured.err


def test_cli_rejects_unknown_model(tmp_path: Path):
    parser = cli.build_parser()
    try:
        parser.parse_args(
            [
                "--model-dir",
                str(tmp_path),
                "--input",
                "in.csv",
                "--model-set",
                "not-a-model",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("unknown model should be rejected")
