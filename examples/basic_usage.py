#!/usr/bin/env python3
"""
Example script demonstrating the basic usage of the XAI-guided adversarial attack framework.

This script shows how to:
1. Load a fake news detection model
2. Generate XAI explanations
3. Create adversarial attacks
4. Evaluate attack success

Run with: python examples/basic_usage.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import TransformerBasedDetector
from xai import XAIExplainer  
from attacks import AdversarialAttack


def main():
    """Main example function."""
    print("XAI-guided Adversarial Attack Example")
    print("=" * 40)
    
    # Sample text
    sample_text = """
    Breaking news: Local university researchers have made a groundbreaking discovery 
    in artificial intelligence that could revolutionize how we detect misinformation online.
    The team has developed a new method that combines explainable AI with advanced 
    language models to identify fake news with unprecedented accuracy.
    """
    
    print(f"Original text: {sample_text.strip()}")
    print()
    
    # Initialize fake news detector
    print("1. Initializing fake news detector...")
    detector = TransformerBasedDetector(model_name="bert-base-uncased")
    
    # Get baseline prediction
    baseline_prediction = detector.predict(sample_text)
    print(f"Baseline prediction: Real={baseline_prediction[0][0]:.3f}, Fake={baseline_prediction[0][1]:.3f}")
    print()
    
    # Note: Full implementation would include actual XAI explanation and attack generation
    print("2. XAI explanation generation...")
    print("   [This would generate explanations showing which parts of the text")
    print("    are most important for the model's decision]")
    print()
    
    print("3. Adversarial attack generation...")
    print("   [This would use XAI insights to guide the generation of adversarial comments")
    print("    that could fool the fake news detector]")
    print()
    
    print("4. Attack evaluation...")
    print("   [This would measure the success rate and quality of generated attacks]")
    print()
    
    print("Example completed! Check the src/ directory for full implementations.")


if __name__ == "__main__":
    main()