# Data Directory

This directory contains datasets used in the research.

## Structure

- `raw/` - Original, unprocessed datasets
- `processed/` - Cleaned and preprocessed datasets ready for experiments
- `results/` - Experimental results and outputs

## Datasets

[Add information about specific datasets used]

### Raw Data
- Place original datasets in the `raw/` directory
- Common formats: CSV, JSON, TXT

### Processed Data
- Preprocessed datasets go in the `processed/` directory
- Should include train/validation/test splits
- Standardized format for compatibility

### Results
- Experimental outputs and attack results
- Performance metrics and analysis
- Generated adversarial examples

## Usage

Use the data utilities in `src/utils/data_utils.py` to load and preprocess datasets.

```python
from src.utils import load_dataset

train_data = load_dataset("data/processed/train.csv")
```