# Experiments

This directory contains experimental scripts and Jupyter notebooks for reproducing the research results.

## Structure

- `notebooks/` - Jupyter notebooks for interactive analysis
- `scripts/` - Python scripts for running experiments
- `configs/` - Experiment-specific configuration files

## Running Experiments

1. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up configuration:
   ```bash
   cp docs/config.yaml experiments/configs/my_experiment.yaml
   # Edit the configuration as needed
   ```

3. Run experiments:
   ```bash
   python experiments/scripts/run_attack.py --config experiments/configs/my_experiment.yaml
   ```

## Notebooks

- `01_data_exploration.ipynb` - Dataset analysis and exploration
- `02_model_evaluation.ipynb` - Baseline model performance
- `03_xai_analysis.ipynb` - XAI explanation analysis
- `04_attack_generation.ipynb` - Adversarial attack experiments
- `05_results_analysis.ipynb` - Results analysis and visualization

## Scripts

- `run_attack.py` - Main script for running adversarial attacks
- `evaluate_models.py` - Script for evaluating fake news detectors
- `generate_explanations.py` - Script for generating XAI explanations
- `analyze_results.py` - Script for analyzing experimental results