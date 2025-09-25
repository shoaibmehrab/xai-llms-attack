# Base paths for datasets and model checkpoints
DATASET_FOLDER = "/home/shoaib/attack-on-fake-news/fake_news_data"  
MODEL_LOG_FOLDER = "/home/shoaib/attack-on-fake-news/bert-fake-news/logs/saved-models"
MODEL_RESULTS_FOLDER = "/home/shoaib/attack-on-fake-news/bert-fake-news/logs/results"
TOKENIZER_FOLDER = "/home/shoaib/attack-on-fake-news/bert-fake-news/tokenizers"

# Filenames
MODEL_FORMAT = ".pt"
MODEL_FILENAME = f"bert-fake-news-model{MODEL_FORMAT}"
BEST_MODEL_FILENAME = f"{MODEL_FILENAME}_prioritized_comment_first_style-best{MODEL_FORMAT}"
