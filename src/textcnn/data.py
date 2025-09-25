import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import nltk
import re
# nltk.download("punkt")
from nltk.tokenize import word_tokenize

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

def parse_label(label):
    if isinstance(label, str):
        label = label.strip()
    return int(label)

class FakeNewsDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.samples = []
        for _, row in self.data.iterrows():
            content = str(row["content"])
            comments = str(row["comments"])
            combined_text = content + " " + comments
            label = parse_label(row["label"])
            self.samples.append((combined_text, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def nltk_tokenizer(text: str):
     # Normalize text
    text = text.lower()
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s.,!?]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Optional: lemmatize (requires nltk.download('wordnet'))
    # from nltk.stem import WordNetLemmatizer
    # lemmatizer = WordNetLemmatizer()
    # tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

def build_vocab(all_texts, min_freq=1):
    freq = Counter()
    for tokens in all_texts:
        freq.update(tokens)

    itos = [PAD_TOKEN, UNK_TOKEN]
    for token, count in freq.items():
        if count >= min_freq:
            itos.append(token)

    stoi = {token: idx for idx, token in enumerate(itos)}
    return stoi, itos

def collate_fn(batch, stoi, pad_idx):
    raw_texts, labels = zip(*batch)
    tokenized = [nltk_tokenizer(text) for text in raw_texts]
    numericalized = [[stoi.get(t, stoi[UNK_TOKEN]) for t in tokens] for tokens in tokenized]
    max_len = max(len(seq) for seq in numericalized)
    padded_batch = [(seq + [pad_idx] * (max_len - len(seq))) for seq in numericalized]
    text_tensor = torch.tensor(padded_batch, dtype=torch.long)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return text_tensor, label_tensor

def get_dataloader(csv_file, batch_size, shuffle=False, min_freq=1, stoi=None, itos=None):
    """
    If stoi/itos are None, build the vocab from this csv.
    Otherwise, reuse stoi/itos passed in.
    """
    dataset = FakeNewsDataset(csv_file)

    if stoi is None or itos is None:
        # Build from scratch
        all_token_lists = []
        for text, _ in dataset.samples:
            all_token_lists.append(nltk_tokenizer(text))
        stoi, itos = build_vocab(all_token_lists, min_freq=min_freq)

    pad_idx = stoi[PAD_TOKEN]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, stoi, pad_idx)
    )

    return loader, (stoi, itos)