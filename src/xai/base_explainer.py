"""
Base class for XAI explainers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import numpy as np


class XAIExplainer(ABC):
    """
    Abstract base class for XAI explanation methods.
    """
    
    def __init__(self, explainer_name: str):
        self.explainer_name = explainer_name
        
    @abstractmethod
    def explain(self, 
                text: str,
                model: Any,
                **kwargs) -> Dict[str, Any]:
        """
        Generate explanations for model predictions.
        
        Args:
            text: Input text to explain
            model: The model to explain
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing explanation results
        """
        pass
        
    @abstractmethod
    def get_important_features(self,
                             explanation: Dict[str, Any],
                             top_k: int = 10) -> List[str]:
        """
        Extract the most important features from explanations.
        
        Args:
            explanation: Explanation results
            top_k: Number of top features to return
            
        Returns:
            List of important feature names/tokens
        """
        pass
        
    def __str__(self):
        return f"XAIExplainer({self.explainer_name})"