"""NPClassifier inference with ONNX Runtime.

The public objects are imported lazily so that the conversion module can be
used with only the conversion dependencies (NumPy, ONNX, h5py and ONNX Runtime)
without pulling in RDKit or pandas.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.3.0"
__all__ = ["NPClassifier", "OFFICIAL_THRESHOLDS", "__version__"]


def __getattr__(name: str) -> Any:
    if name == "NPClassifier":
        from .npc import NPClassifier

        return NPClassifier
    if name == "OFFICIAL_THRESHOLDS":
        from .official import OFFICIAL_THRESHOLDS

        return OFFICIAL_THRESHOLDS
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
