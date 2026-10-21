"""Compare a converted NPClassifier ONNX directory with a reference directory.

This helper is optional. It is useful when you already have a trusted ONNX
export and want to check that a new conversion produces the same outputs. It
does not ship or download model weights.

The implementation is also exposed through ``npc verify``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort

MODEL_NAMES: Tuple[str, ...] = ("class", "pathway", "superclass")


def _resolve(directory: Path, model: str) -> Path:
    for name in ("%s_model.onnx" % model, "%s.onnx" % model):
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError("no ONNX model for %r in %s" % (model, directory))


def _input_names(session: ort.InferenceSession) -> Dict[str, str]:
    found: Dict[str, str] = {}
    for meta in session.get_inputs():
        size = int(meta.shape[-1])
        if size == 2048:
            found["formula"] = meta.name
        elif size == 4096:
            found["binary"] = meta.name
        else:
            raise ValueError(
                "unsupported input %r with last dimension %r" % (meta.name, meta.shape)
            )
    if set(found) != {"formula", "binary"}:
        raise ValueError("model must have one 2048-wide and one 4096-wide input")
    return found


def _run(path: Path, formula: np.ndarray, binary: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path))
    names = _input_names(session)
    output = session.run(
        None,
        {
            names["formula"]: formula,
            names["binary"]: binary,
        },
    )[0]
    return np.asarray(output, dtype=np.float32)


def _parse_models(value: str) -> List[str]:
    models = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [item for item in models if item not in MODEL_NAMES]
    if unknown:
        raise argparse.ArgumentTypeError("unknown model(s): %s" % ", ".join(unknown))
    if not models:
        raise argparse.ArgumentTypeError("at least one model is required")
    return [name for name in MODEL_NAMES if name in models]


def verify_dirs(
    reference_dir: Path,
    model_dir: Path,
    *,
    models: Sequence[str] = MODEL_NAMES,
    samples: int = 8,
    seed: int = 0,
    atol: float = 1e-5,
    top_k: int = 1,
) -> List[dict]:
    """Run the parity comparison and return one result dict per model."""
    if samples < 1:
        raise ValueError("samples must be positive")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if atol < 0:
        raise ValueError("atol must be non-negative")
    reference_dir = Path(reference_dir).expanduser()
    model_dir = Path(model_dir).expanduser()
    if not reference_dir.is_dir():
        raise FileNotFoundError("reference directory not found: %s" % reference_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError("model directory not found: %s" % model_dir)

    rng = np.random.default_rng(seed)
    # Non-negative integer counts resemble the Morgan count fingerprints.
    formula = rng.integers(0, 4, size=(samples, 2048)).astype(np.float32)
    binary = rng.integers(0, 4, size=(samples, 4096)).astype(np.float32)

    results: List[dict] = []
    for model in models:
        try:
            reference_path = _resolve(reference_dir, model)
            converted_path = _resolve(model_dir, model)
            reference = _run(reference_path, formula, binary)
            converted = _run(converted_path, formula, binary)
        except Exception as exc:
            results.append(
                {
                    "model": model,
                    "status": "error",
                    "error": "%s: %s" % (type(exc).__name__, exc),
                }
            )
            continue

        if reference.shape != converted.shape:
            results.append(
                {
                    "model": model,
                    "status": "fail",
                    "error": "shape %r != %r" % (reference.shape, converted.shape),
                }
            )
            continue

        max_diff = float(np.max(np.abs(reference - converted)))
        selected_top_k = min(top_k, reference.shape[1])
        reference_top = np.argsort(reference, axis=1)[:, -selected_top_k:][:, ::-1]
        converted_top = np.argsort(converted, axis=1)[:, -selected_top_k:][:, ::-1]
        # Random fingerprints can put a model far outside its training
        # distribution, producing all-zero or tied probabilities. Only compare
        # top-k labels where the reference has a clear, non-zero winner.
        if reference.shape[1] > 1:
            sorted_reference = np.sort(reference, axis=1)
            margin = sorted_reference[:, -1] - sorted_reference[:, -2]
        else:
            margin = np.full(reference.shape[0], np.inf)
        informative = (np.max(reference, axis=1) > atol) & (margin > 10.0 * atol)
        if np.any(informative):
            top_match = bool(
                np.array_equal(reference_top[informative], converted_top[informative])
            )
        else:
            top_match = True
        ok = max_diff <= atol and top_match
        results.append(
            {
                "model": model,
                "status": "ok" if ok else "fail",
                "rows": int(reference.shape[0]),
                "classes": int(reference.shape[1]),
                "max_abs_diff": max_diff,
                "atol": float(atol),
                "top_k": int(selected_top_k),
                "rows_compared_for_top_k": int(np.sum(informative)),
                "top_k_match": top_match,
            }
        )
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verify_parity",
        description="Compare converted NPClassifier ONNX outputs with a reference export.",
    )
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--models",
        type=_parse_models,
        default=list(MODEL_NAMES),
        help="comma-separated models (default: all)",
    )
    parser.add_argument(
        "--samples",
        type=int,
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
        type=int,
        default=1,
        help="compare top-k label agreement (default: 1)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        results = verify_dirs(
            Path(args.reference_dir),
            Path(args.model_dir),
            models=args.models,
            samples=args.samples,
            seed=args.seed,
            atol=args.atol,
            top_k=args.top_k,
        )
    except (FileNotFoundError, ValueError) as exc:
        print("verify_parity: error: %s" % exc, file=sys.stderr)
        return 2

    failures = 0
    for result in results:
        if result["status"] != "ok":
            failures += 1
        stream = sys.stderr if result["status"] == "error" else sys.stdout
        print(json.dumps(result, ensure_ascii=False), file=stream)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
