import torch

class Config:
    def __init__(self):
        self.embed_dim = 300                 # Dimension of word embeddings
        self.kernel_sizes = [2, 3, 4, 5]         # Conv filter sizes
        self.kernel_num = 100                 # Number of filters (output channels)
        self.dropout = 0.5
        self.lr = 0.001
        self.batch_size = 64
        self.epochs = 100
        self.shuffle = True
        self.weight_decay = 0.0001


        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def dump(self):
        print(vars(self))