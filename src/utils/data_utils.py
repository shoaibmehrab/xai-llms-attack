"""
Data utility functions.
"""

import pandas as pd
from typing import Union, List, Dict, Any
import re


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load dataset from file.
    
    Args:
        filepath: Path to the dataset file
        
    Returns:
        Loaded dataset as pandas DataFrame
    """
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.json'):
        return pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def preprocess_text(text: str) -> str:
    """
    Preprocess text for model input.
    
    Args:
        text: Raw text to preprocess
        
    Returns:
        Preprocessed text
    """
    # Basic preprocessing
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
    text = text.strip()
    
    return text


def split_dataset(data: pd.DataFrame, 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_state: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Split dataset into train/validation/test sets.
    
    Args:
        data: Dataset to split
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_state: Random seed
        
    Returns:
        Dictionary with train/val/test splits
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    data_shuffled = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    n = len(data_shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    return {
        'train': data_shuffled[:train_end],
        'val': data_shuffled[train_end:val_end],
        'test': data_shuffled[val_end:]
    }