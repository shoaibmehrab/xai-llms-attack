import os
import argparse
import pandas as pd
import torch
import shap
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from config import Config
from model import BERTClassifier
from data import FakeNewsDataset
from transformers import BertForSequenceClassification
from constants import DATASET_FOLDER, MODEL_LOG_FOLDER, BEST_MODEL_FILENAME, MODEL_RESULTS_FOLDER


class BertTextWrapper(torch.nn.Module):
    def __init__(self, model, tokenizer, device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def forward(self, texts):
        # Ensure input is a list of strings
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(texts, list):
            # Convert any non-str elements to str
            texts = [str(t) for t in texts]
        else:
            raise ValueError("Input to BertTextWrapper must be a string or list of strings.")

        enc = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].to(self.device)
        attention_mask = enc['attention_mask'].to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

def ensure_dir(folder_path):
    os.makedirs(folder_path, exist_ok=True)

def get_test_dataset(tokenizer, max_length, comment_mode):
    test_csv = os.path.join(DATASET_FOLDER, "test.csv")
    data = pd.read_csv(test_csv)
    return FakeNewsDataset(data, tokenizer, max_length=max_length, comment_mode=comment_mode)

def get_background_data(dataset, num_background=20):
    indices = torch.randperm(len(dataset))[:num_background]
    background = [dataset[int(i)] for i in indices]  
    input_ids = torch.stack([item['input_ids'] for item in background])
    attention_mask = torch.stack([item['attention_mask'] for item in background])
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

def get_tokens_from_ids(tokenizer, input_ids):
    return tokenizer.convert_ids_to_tokens(input_ids.tolist())

def save_token_shap_csv(tokens, shap_values, label, pred, confidence, article_id, out_dir):
    df = pd.DataFrame({
        'token': tokens,
        'shap_value': shap_values,
        'label': label,
        'prediction': pred,
        'confidence': confidence
    })
    ensure_dir(out_dir)
    df.to_csv(os.path.join(out_dir, f"{article_id}.csv"), index=False)

def aggregate_global_shap(global_dict, tokens, shap_values):
    for t, s in zip(tokens, shap_values):
        if t not in global_dict:
            global_dict[t] = []
        global_dict[t].append(s)

def save_global_shap_csv(global_dict, out_path):
    rows = []
    for token, vals in global_dict.items():
        rows.append({'token': token, 'mean_shap': sum(vals)/len(vals), 'count': len(vals)})
    df = pd.DataFrame(rows)
    df = df.sort_values('mean_shap', ascending=False)
    df.to_csv(out_path, index=False)


def save_token_confidence(tokens, shap_values, pred, label, confidence, article_id, out_dir):
    """
    Save per-token SHAP impact/confidence for BERT predictions.
    """
    # Calculate impact and confidence
    values = np.array(shap_values)
    confidences = np.abs(values)
    if np.max(confidences) > 0:
        confidences = confidences / np.max(confidences)
    # Direction: FAKE if pred==1 and value>0 or pred==0 and value<0, else REAL
    directions = [
        'FAKE' if (pred == 1 and v > 0) or (pred == 0 and v < 0) else 'REAL'
        for v in values
    ]
    df = pd.DataFrame({
        'Token': tokens,
        'Impact': values,
        'Confidence': confidences,
        'Direction': directions
    })

    # Filter tokens based on actual label
    is_actually_fake = (label == 1)
    if is_actually_fake:
        filtered_df = df[df['Direction'] == 'FAKE'].copy()
        specific_dir = os.path.join(out_dir, "fake_tokens")
    else:
        filtered_df = df[df['Direction'] == 'REAL'].copy()
        specific_dir = os.path.join(out_dir, "real_tokens")
    os.makedirs(specific_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "all_tokens"), exist_ok=True)

    # Sort by confidence
    filtered_df = filtered_df.sort_values('Confidence', ascending=False)
    filtered_df.to_csv(os.path.join(specific_dir, f"{article_id}.csv"), index=False)
    df.sort_values('Confidence', ascending=False).to_csv(
        os.path.join(out_dir, "all_tokens", f"{article_id}.csv"),
        index=False
    )

def main():
    parser = argparse.ArgumentParser(description="SHAP analysis for BERTClassifier")
    parser.add_argument("--dataset", type=str, default="gossipcop", choices=["gossipcop", "politifact"], help="Which dataset to use")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--comment-mode", type=str, default="all", choices=["all", "none", "limited"])
    parser.add_argument("--num-background", type=int, default=20, help="Number of background samples for SHAP")
    parser.add_argument("--max-examples", type=int, default=0, help="Max number of test examples (0=all)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config.model_name)

    # Use dataset argument for paths
    test_csv = os.path.join(DATASET_FOLDER, f"{args.dataset}_test.csv")
    model_path = os.path.join(MODEL_LOG_FOLDER, args.dataset, BEST_MODEL_FILENAME)
    results_dir = os.path.join(MODEL_RESULTS_FOLDER, args.dataset)
    per_article_dir = os.path.join(results_dir, f"token_confidence/{BEST_MODEL_FILENAME}")
    ensure_dir(per_article_dir)

    # Load model
    model = BertForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load test dataset
    data = pd.read_csv(test_csv)
    test_dataset = FakeNewsDataset(data, tokenizer, max_length=args.max_length, comment_mode=args.comment_mode)
    if args.max_examples > 0:
        indices = list(range(min(args.max_examples, len(test_dataset))))
    else:
        indices = list(range(len(test_dataset)))

    # Prepare SHAP background
    background = get_background_data(test_dataset, num_background=args.num_background)
    background_input_ids = background['input_ids'].to(device)
    background_attention_mask = background['attention_mask'].to(device)

    # def bert_predict(inputs):
    #     with torch.no_grad():
    #         logits = model(
    #             input_ids=inputs['input_ids'].to(device),
    #             attention_mask=inputs['attention_mask'].to(device)
    #         )
    #         probs = torch.softmax(logits, dim=1)
    #     return probs.cpu().numpy()

    # explainer = shap.Explainer(
    #     bert_predict,
    #     {'input_ids': background_input_ids, 'attention_mask': background_attention_mask},
    #     algorithm="auto"
    # )

    masker = shap.maskers.Text(tokenizer)
    wrapped_model = BertTextWrapper(model, tokenizer, device)
    explainer = shap.Explainer(wrapped_model, masker, output_names=["REAL", "FAKE"])

    global_shap = {}

    print("Running SHAP analysis...")
    for idx in tqdm(indices):
        item = test_dataset[idx]
        article_id = data.iloc[idx, 0]
        input_ids = item['input_ids'].unsqueeze(0)
        attention_mask = item['attention_mask'].unsqueeze(0)
        label = int(item['labels'].item())

        # Decode input_ids to text
        text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
        shap_values = explainer([text])

        with torch.no_grad():
            output = model(input_ids.to(device), attention_mask=attention_mask.to(device))
            logits = output.logits
            probs = torch.softmax(logits, dim=1)
            pred = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, pred].item())

        tokens = shap_values.data[0]  # SHAP returns tokens here
        shap_vals = shap_values.values[0][:, pred].tolist()

        # save_token_shap_csv(tokens, shap_vals, label, pred, confidence, article_id, per_article_dir)
        save_token_confidence(tokens, shap_vals, pred, label, confidence, article_id, per_article_dir)
        aggregate_global_shap(global_shap, tokens, shap_vals)

    save_global_shap_csv(global_shap, os.path.join(results_dir, "shap_global.csv"))
    print(f"SHAP analysis complete. Results saved to {results_dir}")

if __name__ == "__main__":
    main()