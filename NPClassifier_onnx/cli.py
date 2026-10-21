"""Command line interface for NPClassifier ONNX inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from . import __version__
from .npc import MODEL_NAMES, NPClassifier


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("expected an integer, got %r" % value) from None
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer, got %d" % number)
    return number


def _model_list(value: str) -> List[str]:
    models = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [item for item in models if item not in MODEL_NAMES]
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown model(s): %s; choose from %s"
            % (", ".join(unknown), ", ".join(MODEL_NAMES))
        )
    if not models:
        raise argparse.ArgumentTypeError("at least one model is required")
    # Keep the canonical order and remove duplicates.
    return [name for name in MODEL_NAMES if name in models]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="npc",
        description=(
            "Classify natural products with NPClassifier ONNX models. "
            "Model files are not bundled; see docs/conversion.md."
        ),
    )
    parser.add_argument(
        "--model-dir",
        "--model_dir",
        dest="model_dir",
        required=True,
        help="directory containing <model>_model.onnx and index*.json",
    )
    parser.add_argument(
        "--input",
        "-i",
        "--input_file",
        dest="input_file",
        required=True,
        help="input table (csv, tsv/txt, xlsx, pkl, parquet or h5)",
    )
    parser.add_argument(
        "--column",
        "--tag-column",
        "--tag_column",
        dest="tag_column",
        default="smiles",
        help="column containing SMILES/InChI strings (default: smiles)",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=_positive_int,
        default=256,
        help="molecules per inference batch (default: 256)",
    )
    parser.add_argument(
        "--mode",
        choices=["scores", "official"],
        default="scores",
        help=(
            "scores: per-class probabilities/top-k (default); official: "
            "NPClassifier class 0.1 / superclass 0.3 / pathway 0.5 + voting"
        ),
    )
    parser.add_argument(
        "--top-k",
        "--top_k",
        dest="top_k",
        type=_positive_int,
        default=1,
        help="number of labels to return per model in scores mode (default: 1)",
    )
    parser.add_argument(
        "--model-set",
        "--model_set",
        dest="model_set",
        type=_model_list,
        default=list(MODEL_NAMES),
        help="comma-separated models: class,pathway,superclass (default: all)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="npc_results.csv",
        help="output table; extension selects csv, tsv, xlsx, pkl or parquet",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="disable the progress bar and the summary line",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + __version__,
    )
    return parser


def _write_output(frame: pd.DataFrame, output: str) -> Path:
    path = Path(output).expanduser()
    if not path.suffix:
        path = path.with_suffix(".csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Official mode returns Python lists; table formats cannot represent them
    # reliably, so join them with a semicolon.  Pickle/parquet keep the lists.
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv", ".tab", ".txt", ".xlsx", ".xls"):
        frame = frame.copy()
        for column in frame.columns:
            if frame[column].map(lambda value: isinstance(value, list)).any():
                frame[column] = frame[column].map(
                    lambda value: ";".join(map(str, value))
                    if isinstance(value, list)
                    else value
                )

    if suffix == ".csv":
        frame.to_csv(path)
    elif suffix in (".tsv", ".tab", ".txt"):
        frame.to_csv(path, sep="\t")
    elif suffix in (".xlsx", ".xls"):
        try:
            frame.to_excel(path)
        except ImportError as exc:
            raise ImportError(
                "writing Excel output requires openpyxl; install with: "
                "pip install 'NPClassifier_onnx[excel]'"
            ) from exc
    elif suffix == ".pkl":
        frame.to_pickle(path)
    elif suffix == ".parquet":
        frame.to_parquet(path)
    else:
        raise ValueError(
            "unsupported output extension %r; use csv, tsv, xlsx, pkl or parquet"
            % suffix
        )
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        classifier = NPClassifier(args.model_dir, args.model_set)
        if args.mode == "official":
            frame = classifier.predict_official_from_table_file(
                data_path=args.input_file,
                tag_column=args.tag_column,
                batch_size=args.batch_size,
                progress=not args.quiet,
            )
        else:
            frame = classifier.predict_from_table_file(
                data_path=args.input_file,
                tag_column=args.tag_column,
                batch_size=args.batch_size,
                top_k=args.top_k,
                progress=not args.quiet,
            )
        output_path = _write_output(frame, args.output)
    except (FileNotFoundError, NotADirectoryError, ValueError, KeyError, ImportError, OSError) as exc:
        print("npc: error: %s" % exc, file=sys.stderr)
        return 2

    if not args.quiet:
        invalid = int((frame["status"] == "invalid").sum())
        print(
            "npc: wrote %d rows (%d invalid) to %s"
            % (len(frame), invalid, output_path),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
