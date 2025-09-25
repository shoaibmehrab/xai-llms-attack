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
from ast import literal_eval # To handle list-like strings in CSV
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline

# --- Configuration ---

# Assumes script is run from a directory containing 'models' and potentially 'bert_shap_tokens'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Path to the folder containing your trained RoBERTa models
#    Expected structure: ./models/ROB_<dataset>_CLF/
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
BERT_TOKENIZER_FOLDER = "/home/shoaib/attack-on-fake-news/bert-fake-news/logs/results"
PRE_GENERATED_COMMENTS_BASE_DIR = "/home/shoaib/attack-on-fake-news/TextCNN-v2/logs/saved-models/explanations"
# 2. Path to the folder containing the BERT SHAP token analysis results
#    This should point to the output of the BERT SHAP analysis script.
#    Expected structure: <BERT_SHAP_OUTPUT_DIR>/<dataset>/token_confidence/<BERT_MODEL_NAME>/all_tokens/
#    Example: ./bert_shap_tokens/gossipcop/token_confidence/bert-fake-news-model.pt_prioritized_comment_first_style-best.pt/all_tokens/

# Filenames
MODEL_FORMAT = ".pt"
MODEL_FILENAME = f"bert-fake-news-model{MODEL_FORMAT}"
BEST_MODEL_FILENAME_FOR_BERT_TOKENIZER = f"{MODEL_FILENAME}_prioritized_comment_first_style-best{MODEL_FORMAT}"

BERT_TOKEN_DIR_BASE = os.path.join(BERT_TOKENIZER_FOLDER) # CHANGE if BERT tokens are elsewhere
BERT_MODEL_FILENAME_FOR_TOKENS = "bert-fake-news-model.pt_prioritized_comment_first_style-best.pt" # CHANGE if BERT model name used for tokens is different

# 3. Path to the directory containing RoBERTa's predictions CSV file.
#    The CSV file itself should be named 'predictions.csv'.
#    It needs columns like 'id', 'predicted' ('FAKE'/'REAL'), 'correct' (1/0).
#    Example: ./roberta_prediction_results/gossipcop/predictions.csv
ROBERTA_PREDICTIONS_DIR_BASE = os.path.join(BASE_DIR, "prediction_results_roberta") # CHANGE if predictions are elsewhere

# 4. Path to the original datasets
DATASET_FOLDER = os.path.expanduser("~/fake_news_data") # Standard location from previous scripts

# 5. Output directory for attack results
ATTACK_OUTPUT_FOLDER = os.path.join(BASE_DIR, "attack_results_roberta_vs_bert_tokens")

# --- Attacker Class ---

class RobertaAttackerWithBertTokens:
    def __init__(self, dataset_name, device=None):
        self.dataset_name = dataset_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # --- Load RoBERTa Model ---
        self.roberta_model_path = os.path.join(MODEL_FOLDER, f'ROB_{self.dataset_name}_CLF')
        if not os.path.exists(self.roberta_model_path):
            raise FileNotFoundError(f"RoBERTa model directory not found at {self.roberta_model_path}.")

        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(self.roberta_model_path)
        self.roberta_model = RobertaForSequenceClassification.from_pretrained(self.roberta_model_path).to(self.device)
        self.roberta_model.eval()

        # Create RoBERTa prediction pipeline
        pipeline_device = 0 if self.device == "cuda" else -1
        self.roberta_pipeline = pipeline(
            'text-classification',
            model=self.roberta_model,
            tokenizer=self.roberta_tokenizer,
            device=pipeline_device,
            return_all_scores=True # Get scores for both labels
        )
        print(f"RoBERTa model and tokenizer loaded from {self.roberta_model_path}")
        print(f"Using device: {self.device}")

        # --- Setup Paths ---
        # Path to RoBERTa's predictions directory (used in main block)
        self.results_dir = os.path.join(ROBERTA_PREDICTIONS_DIR_BASE, self.dataset_name)
        # self.predictions_file = os.path.join(self.results_dir, "predictions.csv") # File is loaded in main

        # Path to BERT SHAP tokens
        self.bert_token_dir = os.path.join(
            BERT_TOKEN_DIR_BASE,
            self.dataset_name, "token_confidence", BERT_MODEL_FILENAME_FOR_TOKENS, "all_tokens"
        )
        if not os.path.exists(self.bert_token_dir):
             print(f"WARNING: BERT token directory not found at {self.bert_token_dir}. Token loading will fail.")

        # Path for saving attack results (base directory)
        self.attack_dir = os.path.join(ATTACK_OUTPUT_FOLDER, self.dataset_name)
        os.makedirs(self.attack_dir, exist_ok=True)


    def load_bert_tokens(self, article_id, is_fake_target, top_n=200):
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
            target_direction = "REAL" if is_fake_target else "FAKE"
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
        """Get original article content and metadata from the test CSV"""
        test_csv_path = os.path.join(DATASET_FOLDER, f"{self.dataset_name}_test.csv")
        try:
            # Use literal_eval for list-like columns as potentially used in RoBERTa training
            df = pd.read_csv(test_csv_path, converters={'title': literal_eval, 'content': literal_eval, 'comments': literal_eval})
            # Assuming the first column is the ID
            article = df[df.iloc[:, 0] == article_id]

            if article.empty:
                print(f"Article ID {article_id} not found in {test_csv_path}")
                return None, None, None, None

            # Extract data, handling potential list/string formats
            row = article.iloc[0]
            content_raw = row['content']
            comments_raw = row['comments']
            title_raw = row['title']

            content = ' '.join(content_raw) if isinstance(content_raw, list) else str(content_raw)
            comments = ' '.join(comments_raw) if isinstance(comments_raw, list) else str(comments_raw)
            title = ' '.join(title_raw) if isinstance(title_raw, list) else str(title_raw) # Join title tokens if needed
            label = int(row['label']) # Actual label

            return content, comments, label, title
        except FileNotFoundError:
            print(f"Error: Test dataset file not found at {test_csv_path}")
            return None, None, None, None
        except Exception as e:
            print(f"Error reading or processing article {article_id} from {test_csv_path}: {e}")
            return None, None, None, None


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
        # if "." in generated_comment:
        #     print("Multiple sentences detected. Extracting the first sentence.")
        #     generated_comment = generated_comment.split(".", 1)[0].strip() + "."

        print(f"Processed comment: {generated_comment}")
        # Return the cleaned comment
        return generated_comment.strip()


    def predict_with_roberta(self, text):
        """Gets prediction from the RoBERTa pipeline."""
        # Renamed from predict_with_model to be specific
        if not text or not isinstance(text, str) or len(text.strip()) < 5:
             print(f"Warning: Input text is too short or invalid for RoBERTa prediction: '{text[:50]}...'")
             # Return structure matching TextCNN version for compatibility with compare_models
             return {'prediction': 'ERROR', 'confidence': 0.0, 'fake_prob': 0.0, 'real_prob': 0.0}
        try:
            with torch.no_grad():
                # Pipeline expects a list of strings
                results = self.roberta_pipeline([text], truncation=True, padding=True, max_length=self.roberta_tokenizer.model_max_length)

            # Process pipeline output: results = [[{'label': 'LABEL_0', 'score': 0.1}, {'label': 'LABEL_1', 'score': 0.9}]]
            scores_dict = {item['label']: item['score'] for item in results[0]}
            # IMPORTANT: Verify RoBERTa label mapping. Assuming LABEL_0=REAL, LABEL_1=FAKE. Adjust if necessary.
            real_prob = scores_dict.get('LABEL_0', 0.0) 
            fake_prob = scores_dict.get('LABEL_1', 0.0)

            pred_class_idx = np.argmax([fake_prob, real_prob]) 
            pred_label = 'REAL' if pred_class_idx == 1 else 'FAKE' 
            confidence = max(real_prob, fake_prob)

            # Return structure matching TextCNN version
            return {
                'prediction': pred_label,
                'confidence': float(confidence),
                'fake_prob': float(fake_prob),
                'real_prob': float(real_prob)
            }
        except Exception as e:
            print(f"Error during RoBERTa prediction for text '{text[:100]}...': {e}")
            return {'prediction': 'ERROR', 'confidence': 0.0, 'fake_prob': 0.0, 'real_prob': 0.0}


    def create_comment_generation_prompt(self, title, content, comments, include_tokens, avoid_tokens, target_class, generated_comments_folder_suffix):
        
        if generated_comments_folder_suffix == "title":
            initaize_prompt = f"Given the following post title: {title}"
        elif generated_comments_folder_suffix == "title_article":
            initaize_prompt = f"Given the following post title: {title} and news article: {content}"
        elif generated_comments_folder_suffix == "title_cmnts":
            initaize_prompt = f"Given the following post title: {title} and the comments are: {comments}"
        else:
            initaize_prompt = f"Given the following post title: {title}, news article: {content} and the comments are: {comments}"
        
        """Create a prompt to generate a new comment using SHAP tokens."""

        prompt = f"""
        {initaize_prompt}.
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

        # target_class = "REAL" if target_class == "REAL" else "FAKE"

        # prompt = f"""

        # Given the following post title: {title}, news article: {content} and the associated comments are: {comments}.
        # Write a new comment that aligns with the overall idea of the title and alters its classification from "{label}".
        # """

        # return
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
        """Verify how many tokens are present in the processed comment. (Copied from attack-textcnn-with-bert.py)"""
        if not processed_comment:
            return [], 0
        # Find tokens from the provided list that are present in the comment (case-insensitive substring check)
        comment_lower = processed_comment.lower()
        used_tokens = [token for token in tokens if token.lower() in comment_lower]
        return used_tokens, len(used_tokens)


    def compare_models(self, article_ids, predictions, generated_comments_folder_suffix, models=None, top_n=200):
        """Attack using BERT SHAP tokens, aggregate and save summary by model, dataset, scenario, and attack type.
           (Copied structure from attack-textcnn-with-bert.py)"""
        if models is None:
            models = ["llama3:8b"] # Default model if none provided

        # Prepare output directory (using self.attack_dir as base)
        comparison_dir = os.path.join(self.attack_dir, f"model_comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{generated_comments_folder_suffix}")
        os.makedirs(comparison_dir, exist_ok=True)
        print(f"Comparison directory created: {comparison_dir}")

        # Track results for each model
        comparison_results = defaultdict(list)

        # Process each article based on the filtered predictions DataFrame
        for _, row in tqdm(predictions.iterrows(), total=len(predictions), desc="Comparing Models"):
            article_id = row['id']
            # Use RoBERTa's prediction as the basis
            predicted_label = row['predicted'] # 'FAKE' or 'REAL'
            is_fake_pred = predicted_label == 'FAKE'
            target_class = 'REAL' if is_fake_pred else 'FAKE'
            # Determine attack_type ('Real' or 'Fake') based on the *target* class
            attack_type = 'Real' if target_class == 'REAL' else 'Fake'

            # Get article content
            content, comments, label, title = self.get_article_content(article_id)
            if content is None:
                print(f"Skipping article {article_id}: Content not found.")
                continue

            # Load BERT SHAP tokens for this article based on the *target* class
            include_tokens = self.load_bert_tokens(article_id, is_fake_target=(target_class == 'FAKE'), top_n=top_n)
            avoid_tokens = self.load_bert_tokens(article_id, is_fake_target=(target_class == 'REAL'), top_n=top_n) # Load tokens for the opposite class to avoid

            if not include_tokens:
                 print(f"Skipping article {article_id}: No BERT 'include' tokens found.")
                 continue

            # Iterate through each LLM model specified
            for model_name in models:
                # print(f"\n-- Article {article_id}, LLM: {model_name} --")

                # # Create prompt and generate a new comment
                # prompt = self.create_comment_generation_prompt(title, content, comments, include_tokens, avoid_tokens, target_class)
                # generated_comment_raw = self.query_ollama(prompt, attack_type, model_name=model_name) # Pass attack_type for temperature
                # processed_comment = self.process_generated_comment(generated_comment_raw)

                print(f"\n-- Article {article_id}, LLM: {model_name}, Loading Pre-generated Comment --")

                # Construct path to the pre-generated comment JSON file
                comment_folder_name = f"model_comparison_{generated_comments_folder_suffix}"
                comment_json_path = os.path.join(
                    PRE_GENERATED_COMMENTS_BASE_DIR,
                    self.dataset_name,
                    "bert-attack-result",
                    comment_folder_name,
                    f"attack_{article_id}_{model_name}.json"
                )

                processed_comment = None
                try:
                    with open(comment_json_path, 'r') as f_json:
                        comment_data = json.load(f_json)
                        processed_comment = comment_data.get("generated_comment")
                    if processed_comment:
                        print(f"Loaded pre-generated comment for article {article_id}, model {model_name}: {processed_comment[:100]}...")
                    else:
                        print(f"Warning: 'generated_comment' key not found or empty in {comment_json_path}")
                except FileNotFoundError:
                    print(f"Warning: Pre-generated comment file not found initially, now generating a comment: {comment_json_path}")
                    
                    os.makedirs(os.path.dirname(comment_json_path), exist_ok=True)

                    prompt = self.create_comment_generation_prompt(title, content, comments, include_tokens, avoid_tokens, target_class, generated_comments_folder_suffix)
                    generated_comment_from_llms = self.query_ollama(prompt, attack_type, model_name=model_name) # Pass attack_type for temperature
                    processed_comment = self.process_generated_comment(generated_comment_from_llms)

                    if processed_comment:
                        print(f"Successfully generated a new comment for {article_id}, model {model_name}.")
                        # Construct the full result object to save it back to the original path
                        # This requires performing the attack tests here to get all necessary data
                        temp_used_tokens, temp_num_used_tokens = self.verify_token_usage(processed_comment, include_tokens)
                        temp_non_used_tokens, temp_num_non_used_tokens = self.verify_token_usage(processed_comment, avoid_tokens)

                        temp_original_text_scenario_1 = content
                        temp_original_text_scenario_2 = content + " " + comments
                        temp_scenario_1_text = content + " " + processed_comment
                        temp_scenario_2_text = content + " " + comments + " " + processed_comment

                        temp_scenario_1_result = self.test_attack_success(temp_original_text_scenario_1, temp_scenario_1_text, is_fake_pred)
                        temp_scenario_2_result = self.test_attack_success(temp_original_text_scenario_2, temp_scenario_2_text, is_fake_pred)

                        comment_to_save_data = {
                            'id': article_id,
                            'model': model_name,
                            'dataset': self.dataset_name.capitalize(),
                            'attack_type': attack_type,
                            'original_predicted': predicted_label,
                            'target_class': target_class,
                            'generated_comment': processed_comment,
                            'used_tokens': temp_used_tokens,
                            'num_used_tokens': temp_num_used_tokens,
                            'avoid_tokens': temp_non_used_tokens,
                            'num_avoid_tokens': temp_num_non_used_tokens,
                            'scenario_1': {
                                'success': temp_scenario_1_result['success'],
                                'original_confidence': temp_scenario_1_result['original_prediction']['confidence'],
                                'modified_confidence': temp_scenario_1_result['modified_prediction']['confidence'],
                            },
                            'scenario_2': {
                                'success': temp_scenario_2_result['success'],
                                'original_confidence': temp_scenario_2_result['original_prediction']['confidence'],
                                'modified_confidence': temp_scenario_2_result['modified_prediction']['confidence'],
                            }
                        }
                        try:
                            with open(comment_json_path, 'w') as f_out:
                                json.dump(comment_to_save_data, f_out, indent=2)
                            print(f"Saved newly generated comment and results to: {comment_json_path}")
                        except Exception as e_save:
                            print(f"Error saving newly generated comment to {comment_json_path}: {e_save}")
                    else:
                        print(f"Failed to generate a new comment for {article_id}, model {model_name} after FileNotFoundError.")
                        # processed_comment remains None, will be handled by the 'if not processed_comment:' block later
                        
                except json.JSONDecodeError:
                    print(f"Warning: Error decoding JSON from {comment_json_path}")
                except Exception as e:
                    print(f"Error loading pre-generated comment from {comment_json_path}: {e}")

                if not processed_comment:
                    print(f"Skipping article {article_id} for model {model_name} due to refusal or invalid comment.")
                    # Optionally record failure
                    result = {
                        'id': article_id, 'model': model_name, 'status': 'Comment Generation Failed',
                        'dataset': self.dataset_name.capitalize(), 'attack_type': attack_type,
                        'original_predicted': predicted_label, 'target_class': target_class,
                    }
                    comparison_results[model_name].append(result)
                    continue

                # Verify token usage (using BERT tokens)
                used_tokens, num_used_tokens = self.verify_token_usage(processed_comment, include_tokens)
                non_used_tokens, num_non_used_tokens = self.verify_token_usage(processed_comment, avoid_tokens)

                # Define original texts for RoBERTa prediction scenarios
                original_text_scenario_1 = content # Content only
                original_text_scenario_2 = content + " " + comments # Content + Original Comments

                # Define modified texts for RoBERTa prediction scenarios
                scenario_1_text = content + " " + processed_comment
                scenario_2_text = content + " " + comments + " " + processed_comment

                # Test both scenarios against RoBERTa (success = prediction flips to target_class)
                scenario_1_result = self.test_attack_success(original_text_scenario_1, scenario_1_text, is_fake_pred)
                scenario_2_result = self.test_attack_success(original_text_scenario_2, scenario_2_text, is_fake_pred)

                # Record result structure matching attack-textcnn-with-bert.py
                result = {
                    'id': article_id,
                    'model': model_name, # LLM model used for generation
                    'dataset': self.dataset_name.capitalize(),
                    'attack_type': attack_type, # 'Real' or 'Fake' (based on target)
                    'original_predicted': predicted_label, # RoBERTa's original prediction
                    'target_class': target_class,
                    'generated_comment': processed_comment,
                    'used_tokens': used_tokens, # BERT include tokens found
                    'num_used_tokens': num_used_tokens,
                    'avoid_tokens': non_used_tokens, # BERT avoid tokens found
                    'num_avoid_tokens': num_non_used_tokens,
                    'scenario_1': { # Content + Generated Comment
                        'success': scenario_1_result['success'],
                        'original_confidence': scenario_1_result['original_prediction']['confidence'],
                        'modified_confidence': scenario_1_result['modified_prediction']['confidence'],
                        # Add prediction labels if needed
                        # 'original_prediction_label': scenario_1_result['original_prediction']['prediction'],
                        # 'modified_prediction_label': scenario_1_result['modified_prediction']['prediction'],
                    },
                    'scenario_2': { # Content + Original Comments + Generated Comment
                        'success': scenario_2_result['success'],
                        'original_confidence': scenario_2_result['original_prediction']['confidence'],
                        'modified_confidence': scenario_2_result['modified_prediction']['confidence'],
                        # Add prediction labels if needed
                        # 'original_prediction_label': scenario_2_result['original_prediction']['prediction'],
                        # 'modified_prediction_label': scenario_2_result['modified_prediction']['prediction'],
                    }
                }

                comparison_results[model_name].append(result)
                # Save individual result as JSON (optional, good for debugging)
                with open(os.path.join(comparison_dir, f"attack_{article_id}_{model_name}.json"), 'w') as f:
                    json.dump(result, f, indent=2)

        # --- Save Aggregate Results (Matching attack-textcnn-with-bert.py) ---
        # all_results_list = []
        # for model_name, results in comparison_results.items():
        #     if results: # Only process if there are results for this model
        #          all_results_list.extend(results)
        #          # Save results per model as CSV
        #          results_df = pd.DataFrame(results)
        #          # Flatten nested scenario dictionaries for easier CSV reading
        #          results_df['scenario_1_success'] = results_df['scenario_1'].apply(lambda x: x.get('success'))
        #          results_df['scenario_1_orig_conf'] = results_df['scenario_1'].apply(lambda x: x.get('original_confidence'))
        #          results_df['scenario_1_mod_conf'] = results_df['scenario_1'].apply(lambda x: x.get('modified_confidence'))
        #          results_df['scenario_2_success'] = results_df['scenario_2'].apply(lambda x: x.get('success'))
        #          results_df['scenario_2_orig_conf'] = results_df['scenario_2'].apply(lambda x: x.get('original_confidence'))
        #          results_df['scenario_2_mod_conf'] = results_df['scenario_2'].apply(lambda x: x.get('modified_confidence'))
        #          results_df = results_df.drop(columns=['scenario_1', 'scenario_2']) # Drop original nested columns
        #          results_df.to_csv(os.path.join(comparison_dir, f"{model_name}_attack_results.csv"), index=False)

        # if not all_results_list:
        #      print("\nNo valid attack results generated across all models.")
        #      return comparison_results # Return empty dict

        # --- Prepare Summary (Matching attack-textcnn-with-bert.py) ---
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


    def test_attack_success(self, original_content, modified_content, is_fake_original_pred):
        """Compares RoBERTa predictions on original and modified text.
           Matches the return structure of attack-textcnn-with-bert.py's version."""
        original_pred = self.predict_with_roberta(original_content)
        modified_pred = self.predict_with_roberta(modified_content)

        # Determine target class based on the original prediction
        target_class = 'REAL' if is_fake_original_pred else 'FAKE'

        # Check if prediction flipped to the target class
        success = (original_pred['prediction'] != 'ERROR' and
                   modified_pred['prediction'] == target_class)

        # Return structure matching attack-textcnn-with-bert.py
        return {
            'original_prediction': original_pred,
            'modified_prediction': modified_pred,
            'success': success
            # Note: 'target_class' is not explicitly returned here, matching the source script
        }

# --- Main Execution Block (Matching attack-textcnn-with-bert.py) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoBERTa Adversarial Attack using BERT SHAP tokens and LLM Comparison")
    parser.add_argument(
        "--dataset",
        type=str,
        default="gossipcop",
        choices=["gossipcop", "politifact"],
        help="Which dataset to use (must match RoBERTa model and BERT token source)"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Maximum number of correctly predicted examples (by RoBERTa) to attack (use 0 for all)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=200, # Default matches attack-textcnn-with-bert.py
        help="Number of top influential tokens to use from BERT SHAP"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3:8b", # Default matches attack-textcnn-with-bert.py
        help="Base Ollama model to use for generating comments"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare base model with additional models specified by --compare-with"
    )
    parser.add_argument(
        "--compare-with",
        type=str,
        default="",
        help="Additional Ollama models to compare with (comma-separated, e.g. 'mistral,phi3')"
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        required=True,
        default="title",
        choices=["title", "title_article", "title_cmnts", "title_article_cmnts"],
        help="Suffix for the folder under bert-attack-result containing pre-generated comments (e.g., 'title_cmnts'). This will be used to form 'model_comparison_<suffix>'."
    )
    args = parser.parse_args()

    # Determine max_examples
    max_examples = None if args.max_examples <= 0 else args.max_examples

    # Initialize the RoBERTa attacker
    attacker = RobertaAttackerWithBertTokens(args.dataset)

    # --- Load RoBERTa Predictions ---
    # Path based on attacker's results_dir
    predictions_file = os.path.join(attacker.results_dir, "predictions.csv")
    try:
        predictions_df = pd.read_csv(predictions_file)
        # Ensure 'correct' column exists and filter for correctly predicted examples by RoBERTa
        if 'correct' not in predictions_df.columns:
             raise ValueError("RoBERTa predictions CSV must contain a 'correct' column (1 for correct, 0 for incorrect).")
        if 'predicted' not in predictions_df.columns:
             raise ValueError("RoBERTa predictions CSV must contain a 'predicted' column ('FAKE'/'REAL').")

        predictions = predictions_df[predictions_df['correct'] == 1].copy()
        if predictions.empty:
            print(f"No correctly predicted examples found in {predictions_file}. Exiting.")
            exit()
        print(f"Loaded {len(predictions_df)} total predictions, {len(predictions)} correctly predicted examples by RoBERTa.")
    except FileNotFoundError:
        print(f"Error: RoBERTa predictions file not found at {predictions_file}. Please generate it first.")
        exit()
    except Exception as e:
        print(f"Error reading RoBERTa predictions file {predictions_file}: {e}")
        exit()

    # Apply max_examples limit if specified
    if max_examples:
        predictions = predictions.head(max_examples)
        print(f"Limiting attack to the first {len(predictions)} correctly predicted examples.")

    # Extract article IDs to process (though compare_models uses the DataFrame directly)
    article_ids = predictions['id'].tolist()

    # --- Determine Models to Compare ---
    models_to_compare = [args.model.strip()] # Start with the base model
    if args.compare and args.compare_with:
        additional_models = [m.strip() for m in args.compare_with.split(',') if m.strip()]
        models_to_compare.extend(additional_models)
        # Remove duplicates just in case
        seen_models = set()
        models_to_compare = [m for m in models_to_compare if not (m in seen_models or seen_models.add(m))]
        print(f"Comparing models: {', '.join(models_to_compare)}")
    elif args.compare:
         print(f"Comparing with base model only: {args.model}") # --compare used but --compare-with empty
    else:
        print(f"Running attack with single model: {args.model}") # No --compare flag

    # --- Run the Comparison ---
    print(f"\n--- Starting RoBERTa Attack using BERT Tokens ---")
    print(f"Dataset: {args.dataset}")
    print(f"Max Examples: {'All Correct' if max_examples is None else max_examples}")
    print(f"Top N BERT Tokens: {args.top_n}")
    print(f"LLM Models for Attack: {', '.join(models_to_compare)}")
    print(f"RoBERTa Model Path: {attacker.roberta_model_path}")
    print(f"BERT Token Path: {attacker.bert_token_dir}")
    print(f"RoBERTa Predictions Path: {predictions_file}")
    print(f"Output Directory Base: {attacker.attack_dir}")
    print(f"-------------------------------------------------")

    # Call compare_models with the filtered predictions DataFrame
    attacker.compare_models(
        article_ids=None, # Pass None as compare_models uses the DataFrame
        predictions=predictions,
        models=models_to_compare,
        top_n=args.top_n,
        generated_comments_folder_suffix=args.prompt_style
    )

    print("\n--- Attack Run Finished ---")