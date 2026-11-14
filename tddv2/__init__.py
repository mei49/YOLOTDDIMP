"""TDD-YOLO v2: fixed CSAF/CAFE modules + DySample, trainable with stock ultralytics."""
from .register import register

register(verbose=False)

from .modules import CAFE2, CSAF2, DySample  # noqa: E402

__all__ = ("register", "CAFE2", "CSAF2", "DySample")
