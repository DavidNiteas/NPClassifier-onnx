"""NPClassifier inference with ONNX Runtime."""

from .npc import NPClassifier
from .official import OFFICIAL_THRESHOLDS

__version__ = "0.3.0"
__all__ = ["NPClassifier", "OFFICIAL_THRESHOLDS", "__version__"]
