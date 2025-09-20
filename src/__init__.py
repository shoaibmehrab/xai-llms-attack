"""
XAI-guided Adversarial Comment Generation with LLMs

This package contains the implementation for the research paper:
"A New Attack Surface: XAI-guided Adversarial Comment Generation with LLMs to Attack Fake News Detectors"
"""

__version__ = "0.1.0"
__author__ = "XAI Attack Research Team"
__email__ = ""

from . import models
from . import attacks
from . import xai
from . import utils

__all__ = [
    "models",
    "attacks", 
    "xai",
    "utils"
]