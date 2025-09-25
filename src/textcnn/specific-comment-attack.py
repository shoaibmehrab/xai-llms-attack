import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import datetime
import random
import ast
from collections import defaultdict

from model import TextCNN
from config import Config
from data import nltk_tokenizer, PAD_TOKEN, UNK_TOKEN, collate_fn
from constants import MODEL_LOG_FOLDER, MODEL_VOCAB_FOLDER, BEST_MODEL_FILENAME, DATASET_FOLDER

class SpecificCommentAttacker:
    def __init__(self, dataset_name, device=None):
        """Initialize attacker with the trained model"""
        self.dataset_name = dataset_name
        
        # Load configuration
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
        self.attack_dir = os.path.join(self.results_dir, "specific_adversarial_attacks")
        os.makedirs(self.attack_dir, exist_ok=True)
        
        # Create ID mapping
        self.id_mapping = self.create_id_mapping()

        # Load specific comments
        self.specific_comments = self.load_specific_comments()
        print(f"Loaded specific comments for {len(self.specific_comments)} articles from {dataset_name}")

    def create_id_mapping(self):
        """Create a mapping between numeric IDs and string IDs"""
        test_csv_path = os.path.join(DATASET_FOLDER, f"{self.dataset_name}_test.csv")
        df = pd.read_csv(test_csv_path)
    
        # Create a mapping dictionary
        id_mapping = {}
    
        # Map between indices and string IDs
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

    def load_specific_comments(self):
        """Load specific comments from CSV file and randomly select one comment for each article"""
        comments_file = os.path.join(DATASET_FOLDER, f"specific_attack_comments_{self.dataset_name}_v2.csv")

        if not os.path.exists(comments_file):
            raise FileNotFoundError(f"Specific comments file not found: {comments_file}")

        # Load comments
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
        """Get original article content and metadata"""
        test_csv_path = os.path.join(DATASET_FOLDER, f"{self.dataset_name}_test.csv")
        df = pd.read_csv(test_csv_path)
        
        # Find article by ID
        article = df[df.iloc[:, 1] == article_id]
        if len(article) == 0:
            return None, None, None, None
        
        content = str(article['content'].values[0])
        comments = str(article['comments'].values[0])
        label = int(article['label'].values[0])
        title = str(article['title'].values[0])
        
        return content, comments, label, title

    def test_attack_success(self, original_content, modified_content, is_fake):
        """Test if attack was successful"""
        # Predict on original content
        original_pred = self.predict_with_model(original_content)
        
        # Predict on modified content
        modified_pred = self.predict_with_model(modified_content)
        
        # If original was FAKE, attack succeeds if new prediction is REAL
        # If original was REAL, attack succeeds if new prediction is FAKE
        target_class = 'REAL' if is_fake else 'FAKE'
        success = modified_pred['prediction'] == target_class
        
        return {
            'original_prediction': original_pred,
            'modified_prediction': modified_pred,
            'success': success
        }

    def predict_with_model(self, text):
        """Make prediction with proper input validation"""
        if not text or len(text.strip()) < 10:
            print("Warning: Input text is too short - predictions may be unreliable")
            # For very short inputs, default to the majority class to avoid errors
            return {
                'prediction': 'FAKE',  # Adjust based on your dataset's majority class
                'confidence': 0.99,
                'fake_prob': 0.99,
                'real_prob': 0.01
            }

        # Tokenize and check length
        tokens = nltk_tokenizer(text)
        if len(tokens) < 5:  # Minimum length check
            print(f"Warning: Text only has {len(tokens)} tokens, padding to minimum length")
            # Add padding tokens to reach minimum length
            tokens += ["<pad>"] * (5 - len(tokens))

        # Create dataset and loader
        class SingleExampleDataset(torch.utils.data.Dataset):
            def __init__(self, text):
                self.samples = [(" ".join(text), 0)]  # Ensure minimum text length

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return self.samples[idx]

        # Process with dataloader
        dataset = SingleExampleDataset(text)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, self.stoi, self.pad_idx)
        )

        # Get prediction
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
            # Return default value on error
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'fake_prob': 0.0,
                'real_prob': 0.0
            }

    def execute_attack(self, max_examples=None, filter_correct=True):
        """Execute adversarial attack using specific comments."""
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

            # Skip if no specific comment available for this article
            if article_id not in self.specific_comments:
                print(f"No specific comment available for article {article_id} (numeric ID: {numeric_id}), skipping")
                continue

            # Get content and comments
            content, original_comments, label, title = self.get_article_content(article_id)
            if content is None:
                continue

            # Get the specific comment
            specific_comment = self.specific_comments[article_id]['comment']

            # Scenario 1: News Content + Specific Comment
            scenario_1_text = content + " " + specific_comment

            # Scenario 2: News Content + Original Comments + Specific Comment
            scenario_2_text = content + " " + original_comments + " " + specific_comment

            # Test both scenarios
            scenario_1_result = self.test_attack_success(content, scenario_1_text, is_fake_pred)
            scenario_2_result = self.test_attack_success(content + " " + original_comments, scenario_2_text, is_fake_pred)

            # Record results
            result = {
                'id': article_id,
                'original_predicted': predicted_label,
                'attack_type': attack_type,
                'target_class': target_class,
                'specific_comment': specific_comment,
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
        results_df.to_csv(os.path.join(self.attack_dir, "specific_attack_results.csv"), index=False)

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

            # Skip if no specific comment available for this article
            if article_id not in self.specific_comments:
                print(f"No specific comment available for article {article_id} (numeric ID: {numeric_id}), skipping")
                continue

            # Get content and comments
            content, original_comments, label, title = self.get_article_content(article_id)
            if content is None:
                continue

            # Get the specific comment
            specific_comment = self.specific_comments[article_id]['comment']

            # Scenario 1: News Content + Specific Comment
            scenario_1_text = content + " " + specific_comment

            # Scenario 2: News Content + Original Comments + Specific Comment
            scenario_2_text = content + " " + original_comments + " " + specific_comment

            # Test both scenarios
            scenario_1_result = self.test_attack_success(content, scenario_1_text, is_fake_pred)
            scenario_2_result = self.test_attack_success(content + " " + original_comments, scenario_2_text, is_fake_pred)

            # Record results
            result = {
                'id': article_id,
                'attack_type': attack_type,
                'original_predicted': predicted_label,
                'target_class': target_class,
                'specific_comment': specific_comment,
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
    parser = argparse.ArgumentParser(description="TextCNN Specific Comment Attack")
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
    
    attacker = SpecificCommentAttacker(args.dataset)
    
    if args.compare:
        attacker.compare_attack_types(max_examples=max_examples, filter_correct=not args.all_predictions)
    else:
        attacker.execute_attack(max_examples=max_examples, filter_correct=not args.all_predictions)