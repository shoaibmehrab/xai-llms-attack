"""
Tests for the models module.
"""

import pytest
import numpy as np
from src.models import FakeNewsDetector, TransformerBasedDetector


class TestTransformerBasedDetector:
    """Test cases for TransformerBasedDetector."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = TransformerBasedDetector()
        assert detector.model_name == "bert-base-uncased"
        assert detector.max_length == 512
        
    def test_predict_single_text(self, sample_text):
        """Test prediction on single text."""
        detector = TransformerBasedDetector()
        predictions = detector.predict(sample_text)
        
        assert predictions.shape == (1, 2)
        assert np.allclose(predictions.sum(axis=1), 1.0)  # Probabilities should sum to 1
        
    def test_predict_multiple_texts(self, sample_text, sample_fake_text):
        """Test prediction on multiple texts."""
        detector = TransformerBasedDetector()
        texts = [sample_text, sample_fake_text]
        predictions = detector.predict(texts)
        
        assert predictions.shape == (2, 2)
        assert np.allclose(predictions.sum(axis=1), 1.0)  # All probabilities should sum to 1
        
    def test_predict_proba(self, sample_text):
        """Test predict_proba method."""
        detector = TransformerBasedDetector()
        proba = detector.predict_proba(sample_text)
        predictions = detector.predict(sample_text)
        
        assert np.array_equal(proba, predictions)