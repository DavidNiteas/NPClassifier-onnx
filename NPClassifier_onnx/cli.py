"""Command line interface for NPClassifier ONNX inference.

The primary interface is organized as ``npc <command>``::

    npc predict --model-dir ... --input molecules.csv
    npc convert --hdf5-dir ... --output-dir ...
    npc verify --reference-dir ... --model-dir ...
    npc doctor --model-dir ...

The historical single-command form (``npc --model-dir ... --input ...``) is
still accepted for compatibility.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from . import __version__
from .npc import MODEL_NAMES, NPClassifier

COMMANDS = ("predict", "convert", "verify", "doctor")


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "expected an integer, got %r" % value
        ) from None
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
    # Keep canonical order and remove duplicates.
    return [name for name in MODEL_NAMES if name in models]


def _add_predict_arguments(parser: argparse.ArgumentParser) -> None:
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
        "--class-threshold",
        type=float,
        default=None,
        help="official mode: override the class threshold (default: 0.1)",
    )
    parser.add_argument(
        "--superclass-threshold",
        type=float,
        default=None,
        help="official mode: override the superclass threshold (default: 0.3)",
    )
    parser.add_argument(
        "--pathway-threshold",
        type=float,
        default=None,
        help="official mode: override the pathway threshold (default: 0.5)",
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


def build_predict_parser(prog: str = "npc predict") -> argparse.ArgumentParser:
    """Build the parser used by ``npc predict`` and the legacy CLI."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Classify natural products with NPClassifier ONNX models. "
            "Model files are not bundled; see docs/conversion.md."
        ),
    )
    _add_predict_arguments(parser)
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Backward-compatible alias for the historical single-command parser."""
    return build_predict_parser(prog="npc")


def _add_convert_arguments(parser: argparse.ArgumentParser) -> None:
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
        help="path to index_v1.json (auto-detected from --hdf5-dir when omitted)",
    )
    parser.add_argument(
        "--models",
        type=_model_list,
        default=list(MODEL_NAMES),
        help="comma-separated models: class,pathway,superclass (default: all)",
    )
    parser.add_argument(
        "--opset", type=int, default=13, help="ONNX opset (default: 13)"
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing ONNX files"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="run the converted models once with ONNX Runtime",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")


def _add_verify_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--models",
        type=_model_list,
        default=list(MODEL_NAMES),
        help="comma-separated models (default: all)",
    )
    parser.add_argument(
        "--samples",
        type=_positive_int,
        default=8,
        help="number of random fingerprint rows (default: 8)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="maximum allowed absolute difference (default: 1e-5)",
    )
    parser.add_argument(
        "--top-k",
        type=_positive_int,
        default=1,
        help="compare top-k label agreement (default: 1)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress success lines")


def _add_doctor_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-dir",
        default=None,
        help="optional model directory to inspect",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="optional input table to inspect",
    )
    parser.add_argument(
        "--models",
        type=_model_list,
        default=list(MODEL_NAMES),
        help="models to inspect when --model-dir is given",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of human-readable lines",
    )


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="npc",
        description=(
            "NPClassifier ONNX command line tools. Use 'npc predict' for inference, "
            "'npc convert' for HDF5-to-ONNX conversion, 'npc verify' for parity "
            "checks and 'npc doctor' for environment diagnostics."
        ),
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s " + __version__
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    predict = sub.add_parser("predict", help="run inference on a table")
    _add_predict_arguments(predict)
    convert = sub.add_parser("convert", help="convert official HDF5 models to ONNX")
    _add_convert_arguments(convert)
    verify = sub.add_parser("verify", help="compare two ONNX model directories")
    _add_verify_arguments(verify)
    doctor = sub.add_parser("doctor", help="check the local installation")
    _add_doctor_arguments(doctor)
    return parser


def _write_output(frame: pd.DataFrame, output: str) -> Path:
    path = Path(output).expanduser()
    if not path.suffix:
        path = path.with_suffix(".csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Official mode returns Python lists; table formats cannot represent them
    # reliably, so join them with a semicolon. Pickle/parquet keep the lists.
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


def _run_predict(args: argparse.Namespace) -> int:
    classifier = NPClassifier(args.model_dir, args.model_set)
    if args.mode == "official":
        thresholds = {}
        for key, value in (
            ("class", args.class_threshold),
            ("superclass", args.superclass_threshold),
            ("pathway", args.pathway_threshold),
        ):
            if value is not None:
                if value < 0 or value > 1:
                    raise ValueError("thresholds must be in [0, 1]")
                thresholds[key] = value
        frame = classifier.predict_official_from_table_file(
            data_path=args.input_file,
            tag_column=args.tag_column,
            batch_size=args.batch_size,
            progress=not args.quiet,
            thresholds=thresholds or None,
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
    if not args.quiet:
        invalid = int((frame["status"] == "invalid").sum())
        print(
            "npc: wrote %d rows (%d invalid) to %s"
            % (len(frame), invalid, output_path),
            file=sys.stderr,
        )
    return 0


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


def _run_convert(args: argparse.Namespace) -> int:
    try:
        from .convert import _verify_outputs, convert_models
    except ImportError as exc:
        raise ImportError(
            "conversion requires the 'convert' extra; install with "
            "'pip install \"npclassifier-onnx[convert]\"'"
        ) from exc

    hdf5_dir = Path(args.hdf5_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    index_path = (
        Path(args.index).expanduser() if args.index else _default_index(hdf5_dir)
    )
    if index_path is None:
        raise FileNotFoundError(
            "no index_v1.json found; pass --index path/to/index_v1.json "
            "(it is part of the official NPClassifier source tree)"
        )
    convert_models(
        hdf5_dir=hdf5_dir,
        output_dir=output_dir,
        index_path=index_path,
        models=args.models,
        opset=args.opset,
        force=args.force,
    )
    if args.verify:
        _verify_outputs(output_dir, args.models)
    if not args.quiet:
        print("npc: converted models into %s" % output_dir)
    return 0


def _run_verify(args: argparse.Namespace) -> int:
    from .parity import verify_dirs

    results = verify_dirs(
        Path(args.reference_dir),
        Path(args.model_dir),
        models=args.models,
        samples=args.samples,
        seed=args.seed,
        atol=args.atol,
        top_k=args.top_k,
    )
    failures = 0
    for result in results:
        if result["status"] != "ok":
            failures += 1
        if result["status"] == "error":
            print(json.dumps(result, ensure_ascii=False), file=sys.stderr)
        elif not args.quiet or result["status"] != "ok":
            print(json.dumps(result, ensure_ascii=False))
    return 0 if failures == 0 else 1


def _package_version(name: str) -> Optional[str]:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _run_doctor(args: argparse.Namespace) -> int:
    report: Dict[str, Any] = {
        "package": "npclassifier-onnx",
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "pandas", "onnxruntime", "rdkit", "onnx", "h5py")
        },
        "checks": [],
        "errors": [],
    }

    def check(name: str, ok: bool, detail: str = "") -> None:
        report["checks"].append(
            {"name": name, "ok": bool(ok), "detail": "" if ok else detail}
        )
        if not ok:
            report["errors"].append("%s: %s" % (name, detail))

    for name in ("numpy", "pandas", "onnxruntime", "rdkit"):
        check(
            "dependency:%s" % name,
            report["packages"][name] is not None,
            "not installed",
        )

    # Fingerprint smoke test is independent of model weights.
    try:
        from .npc import FINGERPRINT_RADIUS, FINGERPRINT_SIZE

        mol = NPClassifier.get_mol("CCO")
        formula, binary = NPClassifier.calculate_fingerprint(mol)
        check(
            "fingerprint",
            formula.shape == (1, FINGERPRINT_SIZE)
            and binary.shape == (1, FINGERPRINT_SIZE * FINGERPRINT_RADIUS),
            "formula=%r binary=%r" % (formula.shape, binary.shape),
        )
    except Exception as exc:
        check("fingerprint", False, "%s: %s" % (type(exc).__name__, exc))

    if args.input:
        input_path = Path(args.input).expanduser()
        check("input", input_path.is_file(), "not found: %s" % input_path)

    if args.model_dir:
        model_dir = Path(args.model_dir).expanduser()
        check("model_dir", model_dir.is_dir(), "not found: %s" % model_dir)
        if model_dir.is_dir():
            try:
                classifier = NPClassifier(model_dir, args.models)
                report["models"] = {}
                for key, session in classifier.sessions.items():
                    inputs = [
                        {"name": item.name, "shape": list(item.shape)}
                        for item in session.get_inputs()
                    ]
                    outputs = [
                        {"name": item.name, "shape": list(item.shape)}
                        for item in session.get_outputs()
                    ]
                    report["models"][key] = {
                        "inputs": inputs,
                        "outputs": outputs,
                    }
                check("models", True, ", ".join(sorted(classifier.sessions)))
                probabilities, status = classifier.batch_predict(
                    ["CCO"], used_model=list(classifier.sessions), return_status=True
                )
                check(
                    "smoke_prediction",
                    status == ["ok"]
                    and all(array.shape[0] == 1 for array in probabilities.values()),
                    "status=%r shapes=%r"
                    % (
                        status,
                        {key: value.shape for key, value in probabilities.items()},
                    ),
                )
            except Exception as exc:
                check("models", False, "%s: %s" % (type(exc).__name__, exc))

    errors = bool(report["errors"])
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print("npclassifier-onnx %s" % report["version"])
        print("Python %s on %s" % (report["python"], report["platform"]))
        for name, version in report["packages"].items():
            print("  %-12s %s" % (name, version or "not installed"))
        for item in report["checks"]:
            print(
                "  [%s] %-24s %s"
                % ("ok" if item["ok"] else "FAIL", item["name"], item["detail"])
            )
        print("status: %s" % ("FAIL" if errors else "ok"))
    return 1 if errors else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] in ("-h", "--help"):
        build_cli_parser().print_help()
        return 0
    command = raw[0] if raw else None

    if command in COMMANDS:
        parser = build_cli_parser()
        args = parser.parse_args(raw)
    else:
        # Historical single-command interface, plus ``npc --help`` and
        # ``npc --version``.
        parser = build_parser()
        args = parser.parse_args(raw)
        return _execute(parser, args, legacy=True)

    return _execute(parser, args, legacy=False)


def _execute(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    legacy: bool,
) -> int:
    try:
        if legacy or args.command == "predict":
            return _run_predict(args)
        if args.command == "convert":
            return _run_convert(args)
        if args.command == "verify":
            return _run_verify(args)
        if args.command == "doctor":
            return _run_doctor(args)
        parser.error("unknown command")
    except (
        FileNotFoundError,
        NotADirectoryError,
        FileExistsError,
        ValueError,
        KeyError,
        ImportError,
        OSError,
    ) as exc:
        print("npc: error: %s" % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
