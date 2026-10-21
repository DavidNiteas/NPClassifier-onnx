#!/usr/bin/env python3
"""Compare a converted NPClassifier ONNX directory with a reference directory.

This helper is optional.  It is useful when you already have a trusted ONNX
export and want to check that a new conversion produces the same outputs.  It
does not ship or download model weights.

Example::

    python tools/convert/verify_parity.py \
        --reference-dir /path/to/reference \
        --model-dir /path/to/converted \
        --models class,pathway,superclass
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


def main(argv: Optional[Sequence[str]] = None) -> int:
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
    args = parser.parse_args(argv)

    if args.samples < 1:
        print("verify_parity: error: --samples must be positive", file=sys.stderr)
        return 2
    if args.top_k < 1:
        print("verify_parity: error: --top-k must be positive", file=sys.stderr)
        return 2

    reference_dir = Path(args.reference_dir).expanduser()
    model_dir = Path(args.model_dir).expanduser()
    if not reference_dir.is_dir():
        print("verify_parity: error: reference directory not found: %s" % reference_dir, file=sys.stderr)
        return 2
    if not model_dir.is_dir():
        print("verify_parity: error: model directory not found: %s" % model_dir, file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    # Non-negative integer counts resemble the Morgan count fingerprints.
    formula = rng.integers(0, 4, size=(args.samples, 2048)).astype(np.float32)
    binary = rng.integers(0, 4, size=(args.samples, 4096)).astype(np.float32)

    failures = 0
    for model in args.models:
        try:
            reference_path = _resolve(reference_dir, model)
            converted_path = _resolve(model_dir, model)
            reference = _run(reference_path, formula, binary)
            converted = _run(converted_path, formula, binary)
        except (FileNotFoundError, ValueError) as exc:
            print("%s: ERROR %s" % (model, exc), file=sys.stderr)
            failures += 1
            continue

        if reference.shape != converted.shape:
            print(
                "%s: FAIL shape %r != %r" % (model, reference.shape, converted.shape),
                file=sys.stderr,
            )
            failures += 1
            continue

        max_diff = float(np.max(np.abs(reference - converted)))
        top_k = min(args.top_k, reference.shape[1])
        reference_top = np.argsort(reference, axis=1)[:, -top_k:][:, ::-1]
        converted_top = np.argsort(converted, axis=1)[:, -top_k:][:, ::-1]
        # Random fingerprints can put a model far outside its training
        # distribution, producing all-zero or tied probabilities.  Only compare
        # top-k labels where the reference has a clear, non-zero winner.
        if reference.shape[1] > 1:
            sorted_reference = np.sort(reference, axis=1)
            margin = sorted_reference[:, -1] - sorted_reference[:, -2]
        else:
            margin = np.full(reference.shape[0], np.inf)
        informative = (np.max(reference, axis=1) > args.atol) & (
            margin > 10.0 * args.atol
        )
        if np.any(informative):
            top_match = bool(
                np.array_equal(
                    reference_top[informative], converted_top[informative]
                )
            )
        else:
            top_match = True
        ok = max_diff <= args.atol and top_match
        print(
            json.dumps(
                {
                    "model": model,
                    "rows": int(reference.shape[0]),
                    "classes": int(reference.shape[1]),
                    "max_abs_diff": max_diff,
                    "atol": args.atol,
                    "top_k": top_k,
                    "rows_compared_for_top_k": int(np.sum(informative)),
                    "top_k_match": top_match,
                    "status": "ok" if ok else "fail",
                },
                ensure_ascii=False,
            )
        )
        if not ok:
            failures += 1

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
