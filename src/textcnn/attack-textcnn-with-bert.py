import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
import requests
from tqdm import tqdm
import datetime
from collections import defaultdict
import string

from model import TextCNN
from config import Config
from data import nltk_tokenizer, PAD_TOKEN, UNK_TOKEN, collate_fn
from constants import BERT_TOKENIZER_FOLDER, MODEL_LOG_FOLDER, MODEL_VOCAB_FOLDER, BEST_MODEL_FILENAME, DATASET_FOLDER


# Filenames
MODEL_FORMAT = ".pt"
MODEL_FILENAME = f"bert-fake-news-model{MODEL_FORMAT}"
BEST_MODEL_FILENAME_FOR_BERT_TOKENIZER = f"{MODEL_FILENAME}_prioritized_comment_first_style-best{MODEL_FORMAT}"


class TextCNNAttackerBertTokens:
    def __init__(self, dataset_name, device=None):
        self.dataset_name = dataset_name
        self.config = Config()
        if device:
            self.config.device = device

        # Load vocabulary
        vocab_path = os.path.join(MODEL_VOCAB_FOLDER, dataset_name, "vocab.pkl")
        with open(vocab_path, "rb") as f:
            import pickle
            self.stoi, self.itos = pickle.load(f)
        self.config.vocab_size = len(self.stoi)
        self.pad_idx = self.stoi[PAD_TOKEN]

        # Load model
        best_model_path = os.path.join(MODEL_LOG_FOLDER, dataset_name, BEST_MODEL_FILENAME)
        self.model = TextCNN(self.config.vocab_size, self.config).to(self.config.device)
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.config.device))
        self.model.eval()

        print(f"Model loaded from {best_model_path}")

        # Setup paths
        self.results_dir = os.path.join(MODEL_LOG_FOLDER, "explanations", self.dataset_name)
        self.attack_dir = os.path.join(self.results_dir, "bert-attack-result")
        os.makedirs(self.attack_dir, exist_ok=True)

        # BERT SHAP token folder
        self.bert_token_dir = os.path.join(
            BERT_TOKENIZER_FOLDER,
            self.dataset_name, "token_confidence", BEST_MODEL_FILENAME_FOR_BERT_TOKENIZER, "all_tokens"
        )

    def load_bert_tokens(self, article_id, is_fake, top_n=200):
        """
        Load tokens for a specific article ID based on its classification (fake or real)
        from the BERT SHAP all_tokens directory.
        """
        tokens_path = os.path.join(self.bert_token_dir, f"{article_id}.csv")
        if os.path.exists(tokens_path):
            tokens_df = pd.read_csv(tokens_path)

            # Identify tokens that appear in both directions
            duplicate_tokens = tokens_df.groupby("Token")["Direction"].nunique()
            conflicting_tokens = duplicate_tokens[duplicate_tokens > 1].index.tolist()

            # Filter tokens based on the target direction
            target_direction = "REAL" if is_fake else "FAKE"
            tokens_df = tokens_df[(tokens_df["Direction"] == target_direction) & (~tokens_df["Token"].isin(conflicting_tokens))]

            # Sort tokens by Confidence in descending order
            tokens_df = tokens_df.sort_values("Confidence", ascending=False)

            # Get the top N tokens
            tokens = tokens_df.head(top_n)["Token"].tolist()

            # Filter out irrelevant tokens (e.g., punctuation, symbols)
            tokens = [str(token) for token in tokens if pd.notnull(token) and str(token) not in string.punctuation and len(str(token).strip()) > 0]

            # Remove duplicates while preserving order
            seen = set()
            tokens = [token for token in tokens if not (token in seen or seen.add(token))]
        else:
            print(f"BERT tokens file not found for article ID {article_id}: {tokens_path}")
            tokens = []
        return tokens

    def get_article_content(self, article_id):
        """Get original article content and metadata"""
        test_csv_path = os.path.join(DATASET_FOLDER, f"{self.dataset_name}_test.csv")
        df = pd.read_csv(test_csv_path)
        article = df[df.iloc[:, 0] == article_id]
        if len(article) == 0:
            return None, None, None, None
        content = str(article['content'].values[0])
        comments = str(article['comments'].values[0])
        label = int(article['label'].values[0])
        title = str(article['title'].values[0])
        return content, comments, label, title

    def process_generated_comment(self, generated_comment):

        if not generated_comment:
            return None

        print(f"Original generated comment: {generated_comment}")

        # Check for the presence of the pattern \n\n\" and extract the text after it
        split_marker = '\n\n\"'
        if split_marker in generated_comment:
            print(f"Found split marker '{split_marker}'. Extracting text after it.")
            generated_comment = generated_comment.split(split_marker, 1)[-1].strip()

        # Remove unwanted introductory phrases
        unwanted_phrases = [
            "Here is a new single line comment that aligns with the REAL classification:",
            "Here is a new single line comment that aligns with the FAKE classification:",
            "Here's a new single-line comment that aligns with the REAL classification:",
            "Here's a new single-line comment that aligns with the FAKE classification:",
            "Here is a single short new comment that aligns with the REAL classification:",
            "Here is a new single line comment that aligns with the overall idea of the title and uses several global tokens for REAL classification:",
            "Here's a new single line comment that aligns with the REAL classification:",
            "Here's the generated comment based on the given input:",
            "Here's a comment that aligns with the FAKE classification:",
            "Here's a comment that aligns with the REAL classification:",
            "Here is a single new comment that aligns with the FAKE classification:",
            "Here is a new comment that aligns with the REAL classification:",
            "Here's a generated comment that aligns with the REAL classification:",
            "Here is a generated comment that aligns with the REAL classification:",
            "Here is a single short new comment that aligns with the REAL classification:",
            "Here's a new short comment that aligns with the REAL classification:",
            "Here is a new single-line comment that aligns with the REAL classification:",
            "Here's a fake comment that aligns with the FAKE classification:"
            "Here's a generated comment that aligns with the FAKE classification:",
            "Here is a single short new comment that aligns with the FAKE classification:",
            "Here is a single new comment that aligns with the REAL classification:",
            "Here is a generated single new comment that aligns with the REAL classification:",
            "Here is a single short new comment that aligns with the FAKE classification:",
            "Here is a new comment that aligns with the FAKE classification:",
            "Here is a new single-line comment that aligns with the overall idea of the post title",
            "Here is a new single line comment that aligns with the overall idea of the title and uses many of the global tokens for REAL classification:"
        ]
        for phrase in unwanted_phrases:
            if phrase in generated_comment:
                print(f"Removing phrase: {phrase}")
                generated_comment = generated_comment.replace(phrase, "").strip()

        # Remove surrounding brackets or quotes
        if generated_comment.startswith("[") and generated_comment.endswith("]"):
            generated_comment = generated_comment[1:-1].strip()
        if generated_comment.startswith("\"") and generated_comment.endswith("\""):
            generated_comment = generated_comment[1:-1].strip()

        # Extract only the first sentence if the comment has multiple sentences
        if "." in generated_comment:
            print("Multiple sentences detected. Extracting the first sentence.")
            generated_comment = generated_comment.split(".", 1)[0].strip() + "."

        print(f"Processed comment: {generated_comment}")
        # Return the cleaned comment
        return generated_comment.strip()

    def predict_with_model(self, text):
        if not text or len(text.strip()) < 10:
            return {
                'prediction': 'FAKE',
                'confidence': 0.99,
                'fake_prob': 0.99,
                'real_prob': 0.01
            }
        tokens = nltk_tokenizer(text)
        if len(tokens) < 5:
            tokens += ["<pad>"] * (5 - len(tokens))
        class SingleExampleDataset(torch.utils.data.Dataset):
            def __init__(self, text):
                self.samples = [(" ".join(text), 0)]
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return self.samples[idx]
        dataset = SingleExampleDataset(text)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, self.stoi, self.pad_idx)
        )
        self.model.eval()
        try:
            with torch.no_grad():
                for texts, _ in loader:
                    texts = texts.to(self.config.device)
                    outputs = self.model(texts)
                    probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
                    pred_class = np.argmax(probs)
                    confidence = probs[pred_class]
            return {
                'prediction': 'FAKE' if pred_class == 1 else 'REAL',
                'confidence': float(confidence),
                'fake_prob': float(probs[1]),
                'real_prob': float(probs[0])
            }
        except RuntimeError as e:
            print(f"Error during prediction: {e}")
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'fake_prob': 0.0,
                'real_prob': 0.0
            }
        
    def create_comment_generation_prompt(self, title, content, comments, include_tokens, avoid_tokens, is_fake, attack_type):
        """Create a prompt to generate a new comment using SHAP tokens."""
        target_class = "REAL" if is_fake else "FAKE"

        prompt = f"""
        Given the following post title: "{title}", news article: {content} and the news commenets are: "{comments}".
        Write a new single line comment that aligns with the overall idea of the title and alters its classification to "{target_class}".

        The comment is considered as when people share one news article with a comment in a social media.

        WORDS MUST INCLUDE FOR {target_class} CLASSIFICATION:
        {', '.join(include_tokens)}

        WORDS MUST AVOID FOR {target_class} CLASSIFICATION:
        {', '.join(avoid_tokens)}

        INSTRUCTIONS:
        1. Generate a single line very short new comment that aligns with the {target_class} classification.
        2. Include as many of the WORDS for {target_class} classification as possible.
        3. Avoid using the WORDS listed under "WORDS TO AVOID FOR {target_class} CLASSIFICATION".
        4. IMPORTANT: Give ONLY the generated comment as plain text, don't give any introductory and conlusion text.
        
        
        """
        return prompt

    
    def query_ollama(self, prompt, attack_type, model_name="llama3:8b"):

        temperature = 0.9 if attack_type == "Real" else 0.7

        """Query Ollama API with the given prompt"""
        url = "http://localhost:11434/api/generate"
        
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
            }
        }
        
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                generated_text = response.json()["response"]
            
                # Check for refusal messages
                if "I can't fulfill this request" in generated_text or "violates my policy" in generated_text:
                    print(f"Model {model_name} refused to generate a comment.")
                    return None

                # Return the generated text
                return generated_text.strip()
            else:
                print(f"Error querying Ollama: {response.status_code}")
                print(response.text)
                return None
        except Exception as e:
            print(f"Exception when querying Ollama: {e}")
            return None
        
    def verify_token_usage(self, processed_comment, tokens):
        """Verify how many tokens are present in the processed comment."""
        if not processed_comment:
            return [], 0

        # # Tokenize the processed comment
        # comment_tokens = nltk_tokenizer(processed_comment)

        # Find tokens from the provided list that are present in the comment
        used_tokens = [token for token in tokens if token in processed_comment]

        # Return the list of used tokens and their count
        return used_tokens, len(used_tokens)

    def compare_models(self, article_ids, predictions, models=None, top_n=200):
        """Attack using BERT SHAP tokens, aggregate and save summary by model, dataset, scenario, and attack type."""
        if models is None:
            models = ["llama3:8b"]

        # Prepare output directory
        comparison_dir = os.path.join(self.attack_dir, f"model_comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(comparison_dir, exist_ok=True)
        print(f"Comparison directory created: {comparison_dir}")

        # Track results for each model
        comparison_results = defaultdict(list)

        # Process each article
        for _, row in tqdm(predictions.iterrows(), total=len(predictions)):
            article_id = row['id']
            predicted_label = row['predicted']
            is_fake_pred = predicted_label == 'FAKE'
            target_class = 'REAL' if is_fake_pred else 'FAKE'
            attack_type = 'Fake' if is_fake_pred else 'Real'

            # Get article content
            content, comments, label, title = self.get_article_content(article_id)
            if content is None:
                continue

            # Load BERT SHAP tokens for this article
            include_tokens = self.load_bert_tokens(article_id, is_fake_pred, top_n)
            avoid_tokens = self.load_bert_tokens(article_id, not is_fake_pred, top_n)
            # Compose a new comment using BERT tokens (simple join for demonstration)
            # In practice, you may want to use a language model or prompt as in your original script
            for model in models:
                print(f"Generating comment and testing attack for article {article_id} with model {model}")

                # Create prompt and generate a new comment
                attack_type = 'Fake' if is_fake_pred else 'Real'
                prompt = self.create_comment_generation_prompt(title, content, comments, include_tokens, avoid_tokens, is_fake_pred, attack_type)
                generated_comment = self.query_ollama(prompt, attack_type, model_name=model)
                processed_comment = self.process_generated_comment(generated_comment)

                if not processed_comment:
                    print(f"Skipping article {article_id} for model {model} due to refusal or invalid comment.")
                    continue

                used_tokens, num_used_tokens = self.verify_token_usage(processed_comment, include_tokens)
                non_used_tokens, num_non_used_tokens = self.verify_token_usage(processed_comment, avoid_tokens) 
                # Scenario 1: News Content + Generated Comment
                scenario_1_text = content + " " + processed_comment

                # Scenario 2: News Content + Original Comments + Generated Comment
                scenario_2_text = content + " " + comments + " " + processed_comment

                # Test both scenarios (success = prediction flips to target_class)
                scenario_1_result = self.test_attack_success(content, scenario_1_text, is_fake_pred)
                scenario_2_result = self.test_attack_success(content + " " + comments, scenario_2_text, is_fake_pred)

                # Record result
                result = {
                    'id': article_id,
                    'model': model,
                    'dataset': self.dataset_name.capitalize(),
                    'attack_type': attack_type,
                    'original_predicted': predicted_label,
                    'target_class': target_class,
                    'generated_comment': processed_comment,
                    'used_tokens': used_tokens,
                    'num_used_tokens': num_used_tokens,
                    'avoid_tokens': non_used_tokens,
                    'num_avoid_tokens': num_non_used_tokens,
                    'scenario_1': {
                        'success': scenario_1_result['modified_prediction']['prediction'] == target_class,
                        'original_confidence': scenario_1_result['original_prediction']['confidence'],
                        'modified_confidence': scenario_1_result['modified_prediction']['confidence'],
                    },
                    'scenario_2': {
                        'success': scenario_2_result['modified_prediction']['prediction'] == target_class,
                        'original_confidence': scenario_2_result['original_prediction']['confidence'],
                        'modified_confidence': scenario_2_result['modified_prediction']['confidence'],
                    }
                }
                comparison_results[model].append(result)

                # Save individual result as JSON
                with open(os.path.join(comparison_dir, f"attack_{article_id}_{model}.json"), 'w') as f:
                    json.dump(result, f, indent=2)

        # Save all results as CSV
        for model, results in comparison_results.items():
            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(comparison_dir, f"{model}_attack_results.csv"), index=False)

        # Prepare summary
        summary_rows = []
        for model, results in comparison_results.items():
            for scenario in ['scenario_1', 'scenario_2']:
                for attack_type in ['Fake', 'Real']:
                    filtered = [r for r in results if r['attack_type'] == attack_type]
                    if not filtered:
                        continue
                    success_count = sum(1 for r in filtered if r[scenario]['success'])
                    total = len(filtered)
                    success_rate = 100 * success_count / total if total else 0
                    summary_rows.append({
                        'Model': model,
                        'Dataset': filtered[0]['dataset'] if filtered else self.dataset_name.capitalize(),
                        'Attack_Scenario': 'Scenario 1 (zero comments)' if scenario == 'scenario_1' else 'Scenario 2 (ten comments)',
                        'Attack_Type': attack_type,
                        'Attack_Success_Rate': f"{success_rate:.2f}%"
                    })

        # Save and print summary
        summary_df = pd.DataFrame(summary_rows)
        summary_csv_path = os.path.join(comparison_dir, "attack_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)

        print("\nAttack Summary Table:")
        print(summary_df.to_string(index=False))

        return comparison_results

    def test_attack_success(self, original_content, modified_content, is_fake):
        original_pred = self.predict_with_model(original_content)
        modified_pred = self.predict_with_model(modified_content)
        target_class = 'REAL' if is_fake else 'FAKE'
        success = modified_pred['prediction'] == target_class
        return {
            'original_prediction': original_pred,
            'modified_prediction': modified_pred,
            'success': success
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TextCNN Adversarial Attack using BERT SHAP tokens")
    parser.add_argument(
        "--dataset",
        type=str,
        default="gossipcop", 
        choices=["gossipcop", "politifact"],
        help="Which dataset to use"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Maximum number of examples to attack (use 0 for all)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=200,
        help="Number of top influential tokens to use from BERT SHAP"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3:8b",
        help="Ollama model to use (for compatibility, not used in this script)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare base model with additional models"
    )
    parser.add_argument(
        "--compare-with",
        type=str,
        default="",
        help="Additional models to compare with (comma-separated, e.g. 'mistral,yi')"
    )
    args = parser.parse_args()
    max_examples = None if args.max_examples == 0 else args.max_examples

    attacker = TextCNNAttackerBertTokens(args.dataset)

    predictions_file = os.path.join(attacker.results_dir, "predictions.csv")
    predictions = pd.read_csv(predictions_file)
    predictions = predictions[predictions['correct'] == 1]
    if max_examples:
        predictions = predictions.head(max_examples)
    article_ids = predictions['id'].tolist()

    if args.compare:
        models = [args.model]
        if args.compare_with:
            models.extend(args.compare_with.split(','))
        print(f"Comparing models: {', '.join(models)}")
        attacker.compare_models(article_ids, predictions, models, args.top_n)
    else:
        # For single-model attack, just use compare_models with one model
        attacker.compare_models(article_ids, predictions, [args.model], args.top_n)
        