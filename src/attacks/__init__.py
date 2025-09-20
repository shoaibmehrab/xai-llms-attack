"""
Adversarial Attack Methods

This module contains implementations of various adversarial attack methods
for generating adversarial comments against fake news detectors.
"""

from .base_attack import AdversarialAttack
from .xai_guided_attack import XAIGuidedAttack
from .llm_attack import LLMAdversarialAttack

__all__ = [
    "AdversarialAttack",
    "XAIGuidedAttack", 
    "LLMAdversarialAttack"
]