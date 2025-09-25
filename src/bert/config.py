import torch

class Config:
    def __init__(self):
        self.model_name = 'bert-large-uncased'  # BERT model to use
        self.max_length = 512                  # Max sequence length for BERT
        self.batch_size = 8                    # Smaller batch size for BERT
        self.lr = 2e-5                         # Learning rate for fine-tuning
        self.epochs = 10                        # BERT typically needs fewer epochs
        self.weight_decay = 0.01
        self.warmup_ratio = 0.1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def dump(self):
        print(vars(self))