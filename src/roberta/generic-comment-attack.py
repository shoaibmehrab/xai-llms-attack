import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import datetime
from collections import defaultdict
from ast import literal_eval
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
DATASET_FOLDER = "/home/shoaib/attack-on-fake-news/fake_news_data"
ATTACK_OUTPUT_FOLDER = os.path.join(BASE_DIR, "attack_results_roberta_generic")
ROBERTA_PREDICTIONS_DIR_BASE = os.path.join(BASE_DIR, "prediction_results_roberta")

class GenericCommentAttackerRoBERTa:
    def __init__(self, dataset_name, device=None):
        """Initialize attacker with the trained RoBERTa model"""
        self.dataset_name = dataset_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load RoBERTa model and tokenizer
        self.roberta_model_path = os.path.join(MODEL_FOLDER, f'ROB_{self.dataset_name}_CLF')
        if not os.path.exists(self.roberta_model_path):
            raise FileNotFoundError(f"RoBERTa model directory not found at {self.roberta_model_path}.")

        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(self.roberta_model_path)
        self.roberta_model = RobertaForSequenceClassification.from_pretrained(self.roberta_model_path).to(self.device)
        self.roberta_model.eval()
        
        print(f"RoBERTa model and tokenizer loaded from {self.roberta_model_path}")
        print(f"Using device: {self.device}")
        
        # Setup paths
        self.results_dir = os.path.join(ROBERTA_PREDICTIONS_DIR_BASE, self.dataset_name)
        self.attack_dir = os.path.join(ATTACK_OUTPUT_FOLDER, self.dataset_name, "generic_adversarial_attacks")
        os.makedirs(self.attack_dir, exist_ok=True)

        # Create ID mapping
        self.id_mapping = self.create_id_mapping()
        
        # Load generic comments
        self.generic_comments = self.load_generic_comments()
        print(f"Loaded {len(self.generic_comments)} generic comments for {dataset_name}")

    def load_generic_comments(self):
        """Load generic comments from CSV file"""
        comments_file = os.path.join(DATASET_FOLDER, f"generic_attack_comments_{self.dataset_name}_v2.csv")

        if not os.path.exists(comments_file):
            raise FileNotFoundError(f"Generic comments file not found: {comments_file}")

        # Load comments and create a dictionary with article IDs as keys
        comments_df = pd.read_csv(comments_file)
        comments_dict = {}

        for _, row in comments_df.iterrows():
            article_id = row['id']
            comment = row['comment']
            label = row['label']

            if article_id not in comments_dict:
                comments_dict[article_id] = {
                    'comment': comment,
                    'label': label
                }

        return comments_dict

    def get_article_content(self, article_id):
        """Get original article content and metadata from the test CSV"""
        test_csv_path = os.path.join(DATASET_FOLDER, f"{self.dataset_name}_test.csv")
        try:
            # Use literal_eval for list-like columns as potentially used in RoBERTa training
            df = pd.read_csv(test_csv_path, converters={'title': literal_eval, 'content': literal_eval, 'comments': literal_eval})
            
            # Find article by ID
            article = df[df['id'] == article_id]
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
            title = ' '.join(title_raw) if isinstance(title_raw, list) else str(title_raw)
            label = int(row['label'])

            return content, comments, label, title
        except FileNotFoundError:
            print(f"Error: Test dataset file not found at {test_csv_path}")
            return None, None, None, None
        except Exception as e:
            print(f"Error reading or processing article {article_id} from {test_csv_path}: {e}")
            return None, None, None, None

    def predict_with_roberta(self, text):
        """Make prediction with RoBERTa model"""
        if not text or not isinstance(text, str) or len(text.strip()) < 5:
            print(f"Warning: Input text is too short or invalid for RoBERTa prediction: '{text[:50]}...'")
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'fake_prob': 0.0,
                'real_prob': 0.0
            }
        
        try:
            with torch.no_grad():
                # Tokenize with truncation to handle very long texts
                inputs = self.roberta_tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True,
                    max_length=self.roberta_tokenizer.model_max_length
                ).to(self.device)
                
                outputs = self.roberta_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                
                # Assuming LABEL_0=REAL, LABEL_1=FAKE. Adjust if needed.
                real_prob = probs[0]
                fake_prob = probs[1]
                
                pred_class_idx = np.argmax(probs)
                pred_label = 'FAKE' if pred_class_idx == 1 else 'REAL'
                confidence = float(probs[pred_class_idx])

            return {
                'prediction': pred_label,
                'confidence': confidence,
                'fake_prob': float(fake_prob),
                'real_prob': float(real_prob)
            }
        except Exception as e:
            print(f"Error during RoBERTa prediction for text '{text[:100]}...': {e}")
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'fake_prob': 0.0,
                'real_prob': 0.0
            }
        
    def create_id_mapping(self):
        """Create a mapping between numeric IDs and string IDs"""
        test_csv_path = os.path.join(DATASET_FOLDER, f"{self.dataset_name}_test.csv")
        df = pd.read_csv(test_csv_path)
    
        # Create a mapping dictionary
        id_mapping = {}
    
        # The first column (index 0) contains numeric IDs (row numbers)
        # The second column (index 1) contains string IDs like "politifact13887"
        for index, row in df.iterrows():
            # Get the numeric ID from the first column (index)
            numeric_id = row.iloc[0]  # This is the row number in the CSV
            
            # Get the string ID from the second column (usually named 'id')
            string_id = str(row['id'])  # This is the string ID like "politifact13887"
            
            # Create mappings
            id_mapping[numeric_id] = string_id
            id_mapping[str(numeric_id)] = string_id
            
            # Also map the row index, which might be used in predictions.csv
            id_mapping[index] = string_id
            id_mapping[str(index)] = string_id
    
        return id_mapping
    
    def test_attack_success(self, original_content, modified_content, is_fake):
        """Test if attack was successful"""
        # Predict on original content
        original_pred = self.predict_with_roberta(original_content)
        
        # Predict on modified content
        modified_pred = self.predict_with_roberta(modified_content)
        
        # If original was FAKE, attack succeeds if new prediction is REAL
        # If original was REAL, attack succeeds if new prediction is FAKE
        target_class = 'REAL' if is_fake else 'FAKE'
        success = modified_pred['prediction'] == target_class
        
        return {
            'original_prediction': original_pred,
            'modified_prediction': modified_pred,
            'success': success
        }

    def execute_attack(self, max_examples=None, filter_correct=True):
        """Execute adversarial attack using pre-generated generic comments."""
        # Load predictions
        predictions_file = os.path.join(self.results_dir, "predictions.csv")
        predictions = pd.read_csv(predictions_file)

        # Filter to correctly classified examples if requested
        if filter_correct:
            predictions = predictions[predictions['correct'] == 1]

        if max_examples:
            predictions = predictions.head(max_examples)

        # Prepare results tracking
        results = []

        # Process each example
        for _, row in tqdm(predictions.iterrows(), total=len(predictions)):

            numeric_id = row['id']

            # Convert numeric ID to string ID
            article_id = self.id_mapping.get(numeric_id) or self.id_mapping.get(str(numeric_id))
            
            if not article_id:
                print(f"No mapping found for numeric ID: {numeric_id}, skipping")
                continue

            predicted_label = row['predicted']
            is_fake_pred = predicted_label == 'FAKE'
            target_class = 'REAL' if is_fake_pred else 'FAKE'
            attack_type = 'Fake' if is_fake_pred else 'Real'

            # Skip if no generic comment available for this article
            if article_id not in self.generic_comments:
                print(f"No generic comment available for article {article_id}, skipping")
                continue

            # Get content and comments
            content, original_comments, label, title = self.get_article_content(article_id)
            if content is None:
                continue

            # Get the generic comment
            generic_comment = self.generic_comments[article_id]['comment']

            # Scenario 1: News Content + Generic Comment
            scenario_1_text = content + " " + generic_comment

            # Scenario 2: News Content + Original Comments + Generic Comment
            scenario_2_text = content + " " + original_comments + " " + generic_comment

            # Test both scenarios
            scenario_1_result = self.test_attack_success(content, scenario_1_text, is_fake_pred)
            scenario_2_result = self.test_attack_success(content + " " + original_comments, scenario_2_text, is_fake_pred)

            # Record results
            result = {
                'id': article_id,
                'original_predicted': predicted_label,
                'attack_type': attack_type,
                'target_class': target_class,
                'generic_comment': generic_comment,
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

            results.append(result)

            # Save individual result
            with open(os.path.join(self.attack_dir, f"attack_{article_id}.json"), 'w') as f:
                json.dump(result, f, indent=2)

            # Print progress
            print(f"Article {article_id}: Scenario 1 - {'✓' if scenario_1_result['success'] else '✗'}, "
                  f"Scenario 2 - {'✓' if scenario_2_result['success'] else '✗'}")

        # Save all results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.attack_dir, "generic_attack_results.csv"), index=False)

        # Print summary
        scenario_1_success = sum(1 for r in results if r['scenario_1']['success'])
        scenario_2_success = sum(1 for r in results if r['scenario_2']['success'])
        
        if len(results) > 0:
            print(f"\nAttack Summary:")
            print(f"Scenario 1 - Total examples: {len(results)}, Successful attacks: {scenario_1_success}, "
                f"Success rate: {100 * scenario_1_success / len(results):.2f}%")
            print(f"Scenario 2 - Total examples: {len(results)}, Successful attacks: {scenario_2_success}, "
                f"Success rate: {100 * scenario_2_success / len(results):.2f}%")
        else:
            print("No results to summarize")

        return results

    def compare_attack_types(self, max_examples=None, filter_correct=True):
        """Compare attack effectiveness between Fake and Real articles."""
        # Load predictions
        predictions_file = os.path.join(self.results_dir, "predictions.csv")
        predictions = pd.read_csv(predictions_file)

        # Filter to correctly classified examples if requested
        if filter_correct:
            predictions = predictions[predictions['correct'] == 1]

        if max_examples:
            predictions = predictions.head(max_examples)

        # Prepare results tracking
        fake_results = []
        real_results = []

        # Process each example
        for _, row in tqdm(predictions.iterrows(), total=len(predictions)):
            numeric_id = row['id']
            predicted_label = row['predicted']
            is_fake_pred = predicted_label == 'FAKE'
            target_class = 'REAL' if is_fake_pred else 'FAKE'
            attack_type = 'Fake' if is_fake_pred else 'Real'

            # Convert numeric ID to string ID
            article_id = self.id_mapping.get(numeric_id) or self.id_mapping.get(str(numeric_id))
            
            if not article_id:
                print(f"No mapping found for numeric ID: {numeric_id}, skipping")
                continue

            # Skip if no generic comment available for this article
            if article_id not in self.generic_comments:
                print(f"No generic comment available for article {article_id}, skipping")
                continue

            # Get content and comments
            content, original_comments, label, title = self.get_article_content(article_id)
            if content is None:
                continue

            # Get the generic comment
            generic_comment = self.generic_comments[article_id]['comment']

            # Scenario 1: News Content + Generic Comment
            scenario_1_text = content + " " + generic_comment

            # Scenario 2: News Content + Original Comments + Generic Comment
            scenario_2_text = content + " " + original_comments + " " + generic_comment

            # Test both scenarios
            scenario_1_result = self.test_attack_success(content, scenario_1_text, is_fake_pred)
            scenario_2_result = self.test_attack_success(content + " " + original_comments, scenario_2_text, is_fake_pred)

            # Record results
            result = {
                'id': article_id,
                'attack_type': attack_type,
                'original_predicted': predicted_label,
                'target_class': target_class,
                'generic_comment': generic_comment,
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

            # Add to appropriate result list
            if is_fake_pred:
                fake_results.append(result)
            else:
                real_results.append(result)

            # Save individual result
            with open(os.path.join(self.attack_dir, f"attack_{article_id}.json"), 'w') as f:
                json.dump(result, f, indent=2)

        # Prepare summary data
        comparison_dir = os.path.join(self.attack_dir, f"comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(comparison_dir, exist_ok=True)

        # Calculate statistics
        fake_s1_success = sum(1 for r in fake_results if r['scenario_1']['success'])
        fake_s2_success = sum(1 for r in fake_results if r['scenario_2']['success'])
        real_s1_success = sum(1 for r in real_results if r['scenario_1']['success'])
        real_s2_success = sum(1 for r in real_results if r['scenario_2']['success'])

        # Save all results in CSV format
        if fake_results:
            pd.DataFrame(fake_results).to_csv(os.path.join(comparison_dir, "fake_attack_results.csv"), index=False)
        if real_results:
            pd.DataFrame(real_results).to_csv(os.path.join(comparison_dir, "real_attack_results.csv"), index=False)

        # Prepare summary table
        summary_rows = []
        
        if fake_results:
            fake_s1_rate = 100 * fake_s1_success / len(fake_results)
            fake_s2_rate = 100 * fake_s2_success / len(fake_results)
            
            summary_rows.append({
                'Dataset': self.dataset_name.capitalize(),
                'Attack_Scenario': 'Scenario 1 (no comments)',
                'Attack_Type': 'Fake',
                'Total_Articles': len(fake_results),
                'Successful_Attacks': fake_s1_success,
                'Attack_Success_Rate': f"{fake_s1_rate:.2f}%"
            })
            
            summary_rows.append({
                'Dataset': self.dataset_name.capitalize(),
                'Attack_Scenario': 'Scenario 2 (with comments)',
                'Attack_Type': 'Fake',
                'Total_Articles': len(fake_results),
                'Successful_Attacks': fake_s2_success,
                'Attack_Success_Rate': f"{fake_s2_rate:.2f}%"
            })
        
        if real_results:
            real_s1_rate = 100 * real_s1_success / len(real_results)
            real_s2_rate = 100 * real_s2_success / len(real_results)
            
            summary_rows.append({
                'Dataset': self.dataset_name.capitalize(),
                'Attack_Scenario': 'Scenario 1 (no comments)',
                'Attack_Type': 'Real',
                'Total_Articles': len(real_results),
                'Successful_Attacks': real_s1_success,
                'Attack_Success_Rate': f"{real_s1_rate:.2f}%"
            })
            
            summary_rows.append({
                'Dataset': self.dataset_name.capitalize(),
                'Attack_Scenario': 'Scenario 2 (with comments)',
                'Attack_Type': 'Real',
                'Total_Articles': len(real_results),
                'Successful_Attacks': real_s2_success,
                'Attack_Success_Rate': f"{real_s2_rate:.2f}%"
            })

        # Save and print summary
        summary_df = pd.DataFrame(summary_rows)
        summary_csv_path = os.path.join(comparison_dir, "attack_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)

        print("\nAttack Summary Table:")
        print(summary_df.to_string(index=False))

        return fake_results, real_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoBERTa Generic Comment Attack")
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
        "--all-predictions",
        action="store_true",
        help="Use all predictions, not just correctly classified ones"
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare attack success rates between fake and real articles"
    )

    args = parser.parse_args()
    max_examples = None if args.max_examples == 0 else args.max_examples
    
    attacker = GenericCommentAttackerRoBERTa(args.dataset)
    
    if args.compare:
        attacker.compare_attack_types(max_examples=max_examples, filter_correct=not args.all_predictions)
    else:
        attacker.execute_attack(max_examples=max_examples, filter_correct=not args.all_predictions)