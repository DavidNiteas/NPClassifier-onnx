#!/usr/bin/env python3
"""Compatibility entry point for the packaged HDF5-to-ONNX converter.

The implementation lives in :mod:`NPClassifier_onnx.convert` so that
``npc convert`` works after a normal wheel install.  This script remains for
users who work directly from a source checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NPClassifier_onnx.convert import (  # noqa: E402,F401
    ACTIVATIONS,
    EXPECTED_HDF5,
    MODEL_NAMES,
    build_onnx_model,
    convert_models,
    main,
    sha256,
)

if __name__ == "__main__":
    raise SystemExit(main())
