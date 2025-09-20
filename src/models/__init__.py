"""
Fake News Detection Models

This module contains implementations and wrappers for various fake news detection models
that will be targeted by the adversarial attacks.
"""

from .base_detector import FakeNewsDetector
from .transformer_detector import TransformerBasedDetector

__all__ = [
    "FakeNewsDetector",
    "TransformerBasedDetector"
]