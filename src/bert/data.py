import pandas as pd
import torch
import json
import ast
import random
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

# class FakeNewsDataset(Dataset):
#     def __init__(self, data, tokenizer, max_length=512, comment_mode='all'):
#         """
#         Args:
#             data: DataFrame containing the dataset
#             tokenizer: BERT tokenizer
#             max_length: Maximum sequence length
#             comment_mode: 'all' (content + all comments), 'none' (only content), 
#                           'limited' (content + limited comments)
#         """
#         self.data = data
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.comment_mode = comment_mode
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
        
#         # Parse content (handle list format)
#         try:
#             content = row['content']
#             if isinstance(content, str):
#                 if content.startswith('[') and content.endswith(']'):
#                     content = ast.literal_eval(content)
#                     content = ' '.join(content)
#         except:
#             content = ""
            
#         # Process comments based on comment_mode
#         comments = ""
#         if self.comment_mode != 'none':
#             try:
#                 comment_data = row['comments']
#                 if isinstance(comment_data, str):
#                     if comment_data.startswith('[') and comment_data.endswith(']'):
#                         comment_list = ast.literal_eval(comment_data)
                        
#                         # For limited mode, use only first 10 comments
#                         if self.comment_mode == 'limited' and len(comment_list) > 10:
#                             comment_list = comment_list[:10]
                            
#                         comments = ' '.join(comment_list)
#             except:
#                 comments = ""
        
#         # Combine content and comments
#         if comments:
#             text = content + " " + comments
#         else:
#             text = content
            
#         # Get label
#         label = int(row['label'])
        
#         # Tokenize text
#         encoding = self.tokenizer(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_length,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )
        
#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'labels': torch.tensor(label, dtype=torch.long)
#         }

# class FakeNewsDataset(Dataset):
#     def __init__(self, data, tokenizer, max_length=512, comment_mode='all', content_token_limit=400): # Add content_token_limit
#         """
#         Args:
#             data: DataFrame containing the dataset
#             tokenizer: BERT tokenizer
#             max_length: Maximum sequence length for the combined input
#             comment_mode: 'all' (content + all comments), 'none' (only content),
#                           'limited' (content + limited comments)
#             content_token_limit: Max tokens to keep from the start of the content
#         """
#         self.data = data
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.comment_mode = comment_mode
#         # Calculate max comment tokens, accounting for [CLS], [SEP], [SEP]
#         self.content_token_limit = content_token_limit
#         self.comment_token_limit = max_length - content_token_limit - 3 # Reserve space for [CLS] content [SEP] comments [SEP]

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]

#         # Parse content
#         try:
#             content = row['content']
#             if isinstance(content, str):
#                 if content.startswith('[') and content.endswith(']'):
#                     content = ast.literal_eval(content)
#                     content = ' '.join(content)
#             elif not isinstance(content, str): # Handle potential non-string types like float NaN
#                  content = ""
#         except Exception:
#             content = ""

#         # Process comments
#         comments = ""
#         if self.comment_mode != 'none':
#             try:
#                 comment_data = row['comments']
#                 if isinstance(comment_data, str):
#                     if comment_data.startswith('[') and comment_data.endswith(']'):
#                         comment_list = ast.literal_eval(comment_data)
#                         if self.comment_mode == 'limited' and len(comment_list) > 10:
#                             comment_list = comment_list[:10]
#                         comments = ' '.join(comment_list)
#                 elif not isinstance(comment_data, str): # Handle potential non-string types
#                     comments = ""

#             except Exception:
#                 comments = ""

#         # --- Prioritized Tokenization ---
#         # Tokenize content and comments separately, limiting their length
#         content_tokens = self.tokenizer.tokenize(content)
#         content_tokens = content_tokens[:self.content_token_limit] # Truncate content if needed

#         comment_tokens = []
#         if comments and self.comment_mode != 'none' and self.comment_token_limit > 0:
#             comment_tokens = self.tokenizer.tokenize(comments)
#             comment_tokens = comment_tokens[:self.comment_token_limit] # Truncate comments if needed

#         # Combine tokens with special tokens: [CLS] content [SEP] comments [SEP]
#         # Note: tokenizer.encode_plus handles adding special tokens, truncation, and padding
#         # if you provide text pairs. Let's use that for simplicity and robustness.

#         encoding = self.tokenizer.encode_plus(
#             text=content,                      # First sequence (content)
#             text_pair=comments if self.comment_mode != 'none' else None, # Second sequence (comments)
#             add_special_tokens=True,           # Add '[CLS]' and '[SEP]'
#             max_length=self.max_length,        # Pad & truncate to this length
#             padding='max_length',              # Pad to `max_length`
#             truncation='only_first',           # Truncate the first sequence (content) if necessary *after* considering the pair
#                                                # Or use 'longest_first' or implement custom logic if needed
#             return_attention_mask=True,
#             return_tensors='pt',               # Return PyTorch tensors
#         )

#         label = int(row['label'])

#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'labels': torch.tensor(label, dtype=torch.long)
#         }

# ... Priotrized Style Start ...

class FakeNewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, comment_mode='all', content_token_limit=400):
        """
        Args:
            data: DataFrame containing the dataset
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length for the combined input
            comment_mode: 'all' (content + all comments), 'none' (only content),
                          'limited' (content + limited comments)
            content_token_limit: Max tokens to keep from the start of the content (used conceptually, actual truncation handled by encode_plus)
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.comment_mode = comment_mode
        # Note: content_token_limit is less directly used now with encode_plus handling truncation strategies
        # self.content_token_limit = content_token_limit
        # self.comment_token_limit = max_length - content_token_limit - 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Parse content (ensure robustness)
        try:
            content = row['content']
            if isinstance(content, str):
                if content.startswith('[') and content.endswith(']'):
                    content = ast.literal_eval(content)
                    content = ' '.join(content)
                # Ensure content is not empty or just whitespace
                content = content.strip() if content else ""
            elif pd.isna(content):
                 content = ""
            else: # Handle other potential non-string types
                 content = str(content).strip()
        except Exception:
            content = ""
        # Ensure content is never None for tokenizer
        if content is None: content = ""


        # Process comments (ensure robustness)
        comments = ""
        if self.comment_mode != 'none':
            try:
                comment_data = row['comments']
                if isinstance(comment_data, str):
                    if comment_data.startswith('[') and comment_data.endswith(']'):
                        comment_list = ast.literal_eval(comment_data)
                        if self.comment_mode == 'limited' and len(comment_list) > 10:
                            comment_list = comment_list[:10]
                        comments = ' '.join(comment_list).strip()
                    else: # Handle case where it's a string but not a list representation
                        comments = comment_data.strip()
                elif pd.isna(comment_data):
                    comments = ""
                else: # Handle other potential non-string types
                    comments = str(comment_data).strip()

            except Exception:
                comments = ""
        # Ensure comments are never None for tokenizer
        if comments is None: comments = ""


        # --- Input Formatting ---
        # Always use text_pair structure, even for 'none' mode, but pass an empty string for comments.
        text_content = content
        text_comments = comments if self.comment_mode != 'none' else "" # Use empty string for 'none' mode

        encoding = self.tokenizer.encode_plus(
            text=text_comments,
            text_pair=text_content, # Pass empty string if mode is 'none'
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation='longest_first', # Truncate longest sequence first if combined length exceeds max_length
            return_attention_mask=True,
            return_tensors='pt',
        )

        label = int(row['label'])

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def get_dataloader(dataset_path, tokenizer, batch_size=8, comment_mode='all', val_split=0.1):
    """
    Create train and validation dataloaders
    
    Args:
        dataset_path: Path to CSV file
        tokenizer: BERT tokenizer
        batch_size: Batch size
        comment_mode: How to handle comments ('all', 'none', 'limited')
        val_split: Validation split ratio
    """
    # Load data
    data = pd.read_csv(dataset_path)
    
    # Split into train/validation sets
    train_data, val_data = train_test_split(data, test_size=val_split, random_state=42, stratify=data['label'])
    
    # Create datasets
    train_dataset = FakeNewsDataset(train_data, tokenizer, comment_mode=comment_mode)
    val_dataset = FakeNewsDataset(val_data, tokenizer, comment_mode=comment_mode)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
    )
    
    return train_loader, val_loader

def get_test_dataloader(dataset_path, tokenizer, batch_size=8, comment_mode='all'):
    """Create test dataloader"""
    data = pd.read_csv(dataset_path)
    test_dataset = FakeNewsDataset(data, tokenizer, comment_mode=comment_mode)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    return test_loader


# ... Random Style Start ...

# class FakeNewsDataset(Dataset):
#     def __init__(self, data, tokenizer, max_length=512, comment_mode='all', content_token_limit=400, force_none_prob=0.0, is_training=False): # <-- Add force_none_prob and is_training
#         """
#         Args:
#             data: DataFrame containing the dataset
#             tokenizer: BERT tokenizer
#             max_length: Maximum sequence length for the combined input
#             comment_mode: 'all', 'none', 'limited' - the primary mode for this dataset instance
#             content_token_limit: Conceptual limit (truncation handled by encode_plus)
#             force_none_prob: Probability (0.0 to 1.0) of forcing 'none' mode processing for an item during training.
#             is_training: Boolean flag indicating if this dataset is used for training (to enable random forcing).
#         """
#         self.data = data
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.comment_mode = comment_mode
#         self.force_none_prob = force_none_prob # <-- Store the probability
#         self.is_training = is_training         # <-- Store the flag
#         # ... (rest of __init__ remains the same) ...

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]

#         # --- Determine effective comment mode for this item ---
#         current_comment_mode = self.comment_mode
#         # If training and with some probability, force 'none' mode processing
#         if self.is_training and self.comment_mode != 'none' and random.random() < self.force_none_prob:
#             current_comment_mode = 'none'
#             # print(f"Debug: Forcing none mode for item {idx}") # Optional debug print
#         elif not self.is_training:
#             print(f"Debug (Eval): Item {idx}, is_training={self.is_training}, Initial mode={self.comment_mode}, Effective mode={current_comment_mode}")
#         # Parse content (ensure robustness)
#         # ... (content parsing logic remains the same) ...
#         try:
#             content = row['content']
#             if isinstance(content, str):
#                 if content.startswith('[') and content.endswith(']'):
#                     content = ast.literal_eval(content)
#                     content = ' '.join(content)
#                 content = content.strip() if content else ""
#             elif pd.isna(content):
#                  content = ""
#             else:
#                  content = str(content).strip()
#         except Exception:
#             content = ""
#         if content is None: content = ""


#         # Process comments (ensure robustness) - only if effective mode is not 'none'
#         comments = ""
#         if current_comment_mode != 'none': # <-- Use current_comment_mode here
#             try:
#                 comment_data = row['comments']
#                 if isinstance(comment_data, str):
#                     if comment_data.startswith('[') and comment_data.endswith(']'):
#                         comment_list = ast.literal_eval(comment_data)
#                         # Apply 'limited' logic based on the original self.comment_mode
#                         if self.comment_mode == 'limited' and len(comment_list) > 10:
#                             comment_list = comment_list[:10]
#                         comments = ' '.join(comment_list).strip()
#                     else:
#                         comments = comment_data.strip()
#                 elif pd.isna(comment_data):
#                     comments = ""
#                 else:
#                     comments = str(comment_data).strip()
#             except Exception:
#                 comments = ""
#         if comments is None: comments = ""


#         # --- Input Formatting ---
#         # Always use text_pair structure, pass empty string if effective mode is 'none'
#         text_content = content
#         text_comments = comments if current_comment_mode != 'none' else "" # <-- Use current_comment_mode

#         encoding = self.tokenizer.encode_plus(
#             text=text_content,
#             text_pair=text_comments,
#             add_special_tokens=True,
#             max_length=self.max_length,
#             padding='max_length',
#             truncation='longest_first',
#             return_attention_mask=True,
#             return_tensors='pt',
#         )

#         label = int(row['label']) # Use int() for safety if labels are 0/1

#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'labels': torch.tensor(label, dtype=torch.long)
#         }


# # --- Modify get_dataloader ---
# def get_dataloader(dataset_path, tokenizer, batch_size=8, comment_mode='all', val_split=0.1, force_none_prob_train=0.05): # <-- Add force_none_prob_train
#     """
#     Create train and validation dataloaders. Training loader can randomly force 'none' mode.

#     Args:
#         dataset_path: Path to CSV file
#         tokenizer: BERT tokenizer
#         batch_size: Batch size
#         comment_mode: The primary comment mode ('all', 'none', 'limited')
#         val_split: Validation split ratio
#         force_none_prob_train: Probability of forcing 'none' mode during training dataset processing.
#     """
#     # Load data
#     data = pd.read_csv(dataset_path)

#     # Split into train/validation sets
#     train_data, val_data = train_test_split(data, test_size=val_split, random_state=42, stratify=data['label'])

#     # Create datasets
#     # For training dataset, enable forcing 'none' mode with the specified probability
#     train_dataset = FakeNewsDataset(
#         train_data, tokenizer, comment_mode=comment_mode,
#         force_none_prob=force_none_prob_train, is_training=True # <-- Pass probability and flag
#     )
#     # For validation dataset, strictly use the specified comment_mode (no random forcing)
#     val_dataset = FakeNewsDataset(
#         val_data, tokenizer, comment_mode=comment_mode,
#         force_none_prob=0.0, is_training=False # <-- Probability 0, not training
#     )

#     # Create dataloaders
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False, # No need to shuffle validation data
#     )

#     return train_loader, val_loader

# # --- Modify get_test_dataloader ---
# def get_test_dataloader(dataset_path, tokenizer, batch_size=8, comment_mode='all'):
#     """Create test dataloader. Strictly uses the specified comment_mode."""
#     data = pd.read_csv(dataset_path)
#     # For test dataset, strictly use the specified comment_mode (no random forcing)
#     test_dataset = FakeNewsDataset(
#         data, tokenizer, comment_mode=comment_mode,
#         force_none_prob=0.0, is_training=False # <-- Probability 0, not training
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#     )

#     return test_loader


# ... Random Style End ...