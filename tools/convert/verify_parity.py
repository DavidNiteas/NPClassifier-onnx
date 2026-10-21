#!/usr/bin/env python3
"""Compatibility entry point for the packaged ONNX parity checker.

The implementation lives in :mod:`NPClassifier_onnx.parity`; prefer
``npc verify`` after installation.  This script remains for source checkouts.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NPClassifier_onnx.parity import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
