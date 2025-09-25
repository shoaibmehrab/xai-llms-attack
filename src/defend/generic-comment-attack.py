import os
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
from collections import defaultdict
from ast import literal_eval

# Assuming defend.py is in the same directory or accessible via PYTHONPATH
from defend import Defend  # dEFEND model class

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Should be dEFEND-v2/raw/
DEFEND_MODEL_DIR = os.path.join(BASE_DIR, "saved-models")
DATASET_FOLDER = "/home/shoaib/attack-on-fake-news/fake_news_data"
ATTACK_OUTPUT_FOLDER = os.path.join(BASE_DIR, "attack_results_defend_generic")
DEFEND_PREDICTIONS_DIR_BASE = os.path.join(BASE_DIR, "prediction_results_defend")

class GenericCommentAttackerDefend:
    def __init__(self, dataset_name):
        """Initialize attacker with the trained dEFEND model"""
        self.dataset_name = dataset_name
        
        # Load dEFEND model
        self.defend_model_path_dir = DEFEND_MODEL_DIR
        self.defend_model_filename = f"{self.dataset_name}_defend_model.h5"
        
        if not os.path.exists(os.path.join(self.defend_model_path_dir, self.defend_model_filename)):
            raise FileNotFoundError(
                f"dEFEND model file not found at "
                f"{os.path.join(self.defend_model_path_dir, self.defend_model_filename)}"
            )
        
        # Defend class loads its own tokenizer internally based on the dataset
        self.defend_model = Defend(self.dataset_name)  # This initializes tokenizer etc.
        try:
            self.defend_model.load_weights(
                saved_model_dir=self.defend_model_path_dir,
                saved_model_filename=self.defend_model_filename
            )
            print(f"dEFEND model loaded for dataset: {self.dataset_name} from {self.defend_model_path_dir}")
        except Exception as e:
            print(f"Error loading dEFEND model weights: {e}")
            raise
        
        # Setup paths
        self.results_dir = os.path.join(DEFEND_PREDICTIONS_DIR_BASE, self.dataset_name)
        self.attack_dir = os.path.join(ATTACK_OUTPUT_FOLDER, self.dataset_name, "generic_adversarial_attacks")
        os.makedirs(self.attack_dir, exist_ok=True)

        # Create ID mapping
        self.id_mapping = self.create_id_mapping()
        
        # Load generic comments
        self.generic_comments = self.load_generic_comments()
        print(f"Loaded {len(self.generic_comments)} generic comments for {dataset_name}")

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
        """
        Get original article content and metadata from the test CSV.
        Returns content as a list of sentences and comments as a list of strings.
        """
        test_csv_path = os.path.join(DATASET_FOLDER, f"{self.dataset_name}_test.csv")
        try:
            df = pd.read_csv(test_csv_path, converters={'title': literal_eval, 'content': literal_eval, 'comments': literal_eval})
            
            # Find article by ID
            article = df[df['id'] == article_id]
            if article.empty:
                print(f"Article ID {article_id} not found in {test_csv_path}")
                return None, None, None, None

            # Extract data, handling potential list/string formats
            row = article.iloc[0]
            # dEFEND expects content as list of sentences, comments as list of comment strings
            content_sentences = row['content'] if isinstance(row['content'], list) else [str(row['content'])]
            original_comments_list = row['comments'] if isinstance(row['comments'], list) else [str(row['comments'])]
            
            title_raw = row['title']
            title_str = ' '.join(title_raw) if isinstance(title_raw, list) else str(title_raw)
            actual_label_numeric = int(row['label'])  # 0 for REAL, 1 for FAKE

            return content_sentences, original_comments_list, actual_label_numeric, title_str
        except FileNotFoundError:
            print(f"Error: Test dataset file not found at {test_csv_path}")
            return None, None, None, None
        except Exception as e:
            print(f"Error reading or processing article {article_id} from {test_csv_path}: {e}")
            return None, None, None, None

    def predict_with_defend(self, article_content_sentences_list, article_comments_strings_list):
        """
        Gets prediction from the dEFEND model.
        dEFEND's predict method expects a batch:
        - article_content_sentences_list: e.g., [['sent1', 'sent2']] for a single article batch
        - article_comments_strings_list: e.g., [['comment1', 'comment2']] for a single article batch
        """
        if not article_content_sentences_list or not article_content_sentences_list[0]:  # Check if content is empty
            print(f"Warning: Input content is empty for dEFEND prediction.")
            return {'prediction': 'ERROR', 'confidence': 0.0, 'fake_prob': 0.0, 'real_prob': 0.0}
        try:
            # dEFEND model's predict method handles tokenization and padding internally.
            # It expects a list of articles, where each article's content is a list of sentences,
            # and each article's comments is a list of comment strings.
            probabilities_batch = self.defend_model.predict(
                article_content_sentences_list,
                article_comments_strings_list
            )
            
            if probabilities_batch is None or len(probabilities_batch) == 0:
                print("Error: dEFEND prediction returned None or empty.")
                return {'prediction': 'ERROR', 'confidence': 0.0, 'fake_prob': 0.0, 'real_prob': 0.0}

            probas_for_item = probabilities_batch[0]  # We are predicting one item at a time here

            # Assuming dEFEND output: [prob_REAL, prob_FAKE] (0 for REAL, 1 for FAKE)
            real_prob = float(probas_for_item[0])
            fake_prob = float(probas_for_item[1])
            
            predicted_idx = np.argmax(probas_for_item)
            pred_label = 'FAKE' if predicted_idx == 1 else 'REAL'
            confidence = float(probas_for_item[predicted_idx])

            return {
                'prediction': pred_label,
                'confidence': confidence,
                'fake_prob': fake_prob,
                'real_prob': real_prob
            }
        except Exception as e:
            # Catching content for logging, be mindful of very long content.
            log_content = str(article_content_sentences_list[0][:2]) if article_content_sentences_list and article_content_sentences_list[0] else "N/A"
            print(f"Error during dEFEND prediction for content '{log_content}...': {e}")
            return {'prediction': 'ERROR', 'confidence': 0.0, 'fake_prob': 0.0, 'real_prob': 0.0}

    def test_attack_success(self, article_content_sentences, original_comments_for_scenario, comments_to_add_for_attack, is_fake):
        """
        Compares dEFEND predictions with and without the attack.
        - article_content_sentences: list of sentences for the article.
        - original_comments_for_scenario: list of comment strings for the 'before attack' state.
        - comments_to_add_for_attack: list containing the single generic comment string.
        - is_fake: boolean, True if article was originally predicted as FAKE.
        """
        # Predict on original state for this scenario (batch of 1)
        original_pred_input_content_batch = [article_content_sentences]
        original_pred_input_comments_batch = [original_comments_for_scenario]
        original_pred = self.predict_with_defend(original_pred_input_content_batch, original_pred_input_comments_batch)

        # Predict on modified state for this scenario (batch of 1)
        modified_comments_list = original_comments_for_scenario + comments_to_add_for_attack
        modified_pred_input_content_batch = [article_content_sentences]
        modified_pred_input_comments_batch = [modified_comments_list]
        modified_pred = self.predict_with_defend(modified_pred_input_content_batch, modified_pred_input_comments_batch)

        # If original was FAKE, attack succeeds if new prediction is REAL
        # If original was REAL, attack succeeds if new prediction is FAKE
        target_class = 'REAL' if is_fake else 'FAKE'
        success = (original_pred['prediction'] != 'ERROR' and
                  modified_pred['prediction'] != 'ERROR' and
                  modified_pred['prediction'] == target_class)
        
        return {
            'original_prediction': original_pred,
            'modified_prediction': modified_pred,
            'success': success
        }

    def execute_attack(self, max_examples=None, filter_correct=True):
        """Execute adversarial attack using generic comments."""
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
            article_id = row['id']
            predicted_label = row['predicted']
            is_fake_pred = predicted_label == 'FAKE'
            target_class = 'REAL' if is_fake_pred else 'FAKE'
            attack_type = 'Fake' if is_fake_pred else 'Real'

            # Skip if no generic comment available for this article
            if article_id not in self.generic_comments:
                print(f"No generic comment available for article {article_id}, skipping")
                continue

            # Get content and comments
            content_sentences, original_comments, _, _ = self.get_article_content(article_id)
            if content_sentences is None:
                continue

            # Get the generic comment
            generic_comment = self.generic_comments[article_id]['comment']

            # Scenario 1: News Content + Generic Comment
            scenario_1_original_comments = []  # Empty list for no original comments
            scenario_1_comments_to_add = [generic_comment]  # List with single generic comment

            # Scenario 2: News Content + Original Comments + Generic Comment
            scenario_2_original_comments = original_comments  # Original comments
            scenario_2_comments_to_add = [generic_comment]  # List with single generic comment

            # Test both scenarios
            scenario_1_result = self.test_attack_success(
                content_sentences,
                scenario_1_original_comments,
                scenario_1_comments_to_add,
                is_fake_pred
            )
            
            scenario_2_result = self.test_attack_success(
                content_sentences,
                scenario_2_original_comments,
                scenario_2_comments_to_add,
                is_fake_pred
            )

            # Record results
            result = {
                'id': article_id,
                'original_predicted': predicted_label,
                'attack_type': attack_type,
                'target_class': target_class,
                'generic_comment': generic_comment,
                'scenario_1': {
                    'success': scenario_1_result['success'],
                    'original_confidence': scenario_1_result['original_prediction']['confidence'],
                    'modified_confidence': scenario_1_result['modified_prediction']['confidence'],
                },
                'scenario_2': {
                    'success': scenario_2_result['success'],
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
            content_sentences, original_comments, _, _ = self.get_article_content(article_id)
            if content_sentences is None:
                continue

            # Get the generic comment
            generic_comment = self.generic_comments[article_id]['comment']

            # Scenario 1: News Content + Generic Comment
            scenario_1_original_comments = []  # Empty list for no original comments
            scenario_1_comments_to_add = [generic_comment]  # List with single generic comment

            # Scenario 2: News Content + Original Comments + Generic Comment
            scenario_2_original_comments = original_comments  # Original comments
            scenario_2_comments_to_add = [generic_comment]  # List with single generic comment

            # Test both scenarios
            scenario_1_result = self.test_attack_success(
                content_sentences,
                scenario_1_original_comments,
                scenario_1_comments_to_add,
                is_fake_pred
            )
            
            scenario_2_result = self.test_attack_success(
                content_sentences,
                scenario_2_original_comments,
                scenario_2_comments_to_add,
                is_fake_pred
            )

            # Record results
            result = {
                'id': article_id,
                'attack_type': attack_type,
                'original_predicted': predicted_label,
                'target_class': target_class,
                'generic_comment': generic_comment,
                'scenario_1': {
                    'success': scenario_1_result['success'],
                    'original_confidence': scenario_1_result['original_prediction']['confidence'],
                    'modified_confidence': scenario_1_result['modified_prediction']['confidence'],
                },
                'scenario_2': {
                    'success': scenario_2_result['success'],
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
    parser = argparse.ArgumentParser(description="dEFEND Generic Comment Attack")
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
    
    attacker = GenericCommentAttackerDefend(args.dataset)
    
    if args.compare:
        attacker.compare_attack_types(max_examples=max_examples, filter_correct=not args.all_predictions)
    else:
        attacker.execute_attack(max_examples=max_examples, filter_correct=not args.all_predictions)