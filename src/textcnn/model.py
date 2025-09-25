import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    """
    A basic TextCNN model for binary classification.
    """
    def __init__(self, vocab_size, config):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, config.embed_dim, padding_idx=0)
        
        # Initialize with pre-trained embeddings if available
        if hasattr(config, 'pretrained_embeddings'):
            self.embedding.weight.data.copy_(config.pretrained_embeddings)
            print("Initialized with pretrained embeddings")
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, config.kernel_num, (k, config.embed_dim))
            for k in config.kernel_sizes
        ])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.kernel_num * len(config.kernel_sizes), 2)

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        embedded = embedded.unsqueeze(1)  # [batch_size, 1, seq_len, embed_dim]

        # Convolutions
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # Pooling
        pooled = [F.max_pool1d(tensor, tensor.size(2)).squeeze(2) for tensor in conved]
        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        out = self.fc(cat)
        return out