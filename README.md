# XAI-Guided  Adversarial Comment Generation with LLMs
## Overview
This repository reproduces the SHAP-guided comment-attack pipeline. We:
1. Train a BERT detector and extract token importance with SHAP.
2. Use influential tokens to prompt an LLM that generates contextually consistent comments.
3. Append generated comments to news articles and evaluate transfer attacks against TextCNN, RoBERTa, and dEFEND.
4. Compare against retrieval or template baselines (MALCOM, CopyCat, etc.).

## Repository Layout
- `models/`: trained checkpoints, vocabularies, prediction CSVs, and attack outputs for each victim model.
- `src/`: all Python modules grouped by component (BERT, TextCNN, RoBERTa, dEFEND).

## Requirements
- Python 3.11 (match the BERT project; adjust if you target a different version).
- CUDA-capable GPU recommended but optional.
- Conda or mamba for the provided `environment.yml` (alternatively use `pip install -r requirements.txt`) for common library.
- Each model has different environment file provided under the `environments` folder with `name.yml` that need to be installed before each execution. 

```powershell
# Windows + conda
conda env create -f environment.yml
conda activate fake-news-llm
```

<!-- ## Data Preparation
1. Place `gossipcop_train.csv`, `gossipcop_test.csv`, `gossipcop_val.csv` and their Politifact equivalents under `data/fake_news_data/`.
2. Ensure attack comment CSVs (`specific_attack_comments_<dataset>_v2.csv`, `generic_attack_comments_<dataset>.csv`, etc.) live in the same directory.
3. Copy SHAP exports (`token_confidence/*.csv`, `summary.csv`, etc.) into `data/bert_shap_tokens/<dataset>/`. -->

## Step 1: Train BERT & Compute SHAP
```powershell
cd src/bert
python train.py --dataset gossipcop
python evaluate.py --dataset gossipcop
python explain_shap.py --dataset gossipcop --output-dir ../../data/bert_shap_tokens/gossipcop
```
Repeat for Politifact.

## Step 2: Generate / Evaluate TextCNN Attacks
TextCNN configuration assumes pretrained checkpoints under `models/textcnn/saved/<dataset>/`.

```powershell
# Predictions (stored in models/textcnn/logs/explanations/<dataset>/predictions.csv)
python src/textcnn/generic_comment_attack.py --dataset gossipcop
python src/textcnn/specific_comment_attack.py --dataset gossipcop
python src/textcnn/attack_textcnn_with_bert.py --dataset gossipcop --top-n 300
```

Outputs land in `models/textcnn/logs/explanations/<dataset>/generic_adversarial_attacks/`,
`specific_adversarial_attacks/`, and `bert_token_attacks/` (ensure directories exist).

## Step 3: RoBERTa Attacks
RoBERTa checkpoints must be under `models/roberta/ROB_<dataset>_CLF/`.

```powershell
python src/roberta/generic_comment_attack.py --dataset gossipcop
python src/roberta/specific_comment_attack.py --dataset gossipcop
python src/roberta/attack_roberta_with_bert.py --dataset gossipcop --top-n 300
```
Generated comments and metrics will be stored in `models/roberta/attack_results_*`.

## Step 4: dEFEND Attacks
Place each `<dataset>_defend_model.h5` inside `models/defend/saved-models/`.

```powershell
python src/defend/generic_comment_attack.py --dataset gossipcop
python src/defend/specific_comment_attack.py --dataset gossipcop
python src/defend/attack_defend_with_bert.py --dataset gossipcop --top-n 300
```

## Baseline Comparisons
- MALCOM / CopyCat / retrieval scripts should log into their respective folders under `models/<model>/attack_results_<baseline>/`.
- Use `src/common/attack_metrics.py` to aggregate success rates and naturalness metrics.

## Logging & Reproducibility
- Each attack script writes per-article JSON and summary CSVs with timestamps.
- Set `TORCH_SEED`, `PYTHONHASHSEED`, and `random.seed()` inside scripts for deterministic runs.

## Adding New Victim Models
1. Extend `src/<model>/` with a new `attack_<model>_with_bert.py`.
2. Ensure `predict_with_<model>` returns label string, probabilities, and metadata.
3. Update README instructions and scripts.

## Troubleshooting
- **Missing vocab or predictions:** regenerate via TextCNN training/evaluation pipelines.
- **Path errors:** verify `constants.py` paths point to `data/` and `models/`.

<!-- ## Citation
If you publish, cite SHAP, MALCOM, CopyCat, and any datasets accordingly. -->
