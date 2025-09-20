"""
Base class for adversarial attacks.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class AdversarialAttack(ABC):
    """
    Abstract base class for adversarial attacks against fake news detectors.
    """
    
    def __init__(self, attack_name: str):
        self.attack_name = attack_name
        
    @abstractmethod
    def generate_adversarial_comment(self, 
                                   original_text: str,
                                   target_model: Any,
                                   **kwargs) -> str:
        """
        Generate an adversarial comment for the given text.
        
        Args:
            original_text: The original text to attack
            target_model: The target fake news detection model
            **kwargs: Additional attack parameters
            
        Returns:
            Generated adversarial comment
        """
        pass
        
    @abstractmethod
    def evaluate_attack_success(self,
                              original_text: str,
                              adversarial_comment: str,
                              target_model: Any) -> Dict[str, float]:
        """
        Evaluate the success of the adversarial attack.
        
        Args:
            original_text: Original text
            adversarial_comment: Generated adversarial comment
            target_model: Target model
            
        Returns:
            Dictionary with attack success metrics
        """
        pass
        
    def __str__(self):
        return f"AdversarialAttack({self.attack_name})"