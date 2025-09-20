"""
Base class for fake news detection models.
"""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class FakeNewsDetector(ABC):
    """
    Abstract base class for fake news detection models.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    @abstractmethod
    def predict(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Predict whether texts are fake news.
        
        Args:
            texts: Input text(s) to classify
            
        Returns:
            Prediction probabilities [real_prob, fake_prob] for each text
        """
        pass
        
    @abstractmethod
    def predict_proba(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            texts: Input text(s) to classify
            
        Returns:
            Prediction probabilities
        """
        pass
        
    def __str__(self):
        return f"FakeNewsDetector({self.model_name})"