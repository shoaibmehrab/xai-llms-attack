import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import Config
from data import get_dataloader
from constants import DATASET_FOLDER, MODEL_LOG_FOLDER, BEST_MODEL_FILENAME

# ensure_dir function for creating directories or checking directory existence
def ensure_dir(folder_path: str):
    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"Directory path ensured: {folder_path}")
    except Exception as e:
        print(f"Error creating directory {folder_path}: {e}")
        raise

def train_model(model, train_loader, val_loader, config, dataset_name):
    # Create dataset-specific save directory
    save_dir = os.path.join(MODEL_LOG_FOLDER, dataset_name)
    ensure_dir(save_dir)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )

    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(1, config.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config.epochs}")
        print(f"{'='*50}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()
            
            # Print progress
            if (i + 1) % 20 == 0:
                print(f"Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                labels = batch['labels'].to(config.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds)
        val_rec = recall_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        
        print(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}")
        print(f"Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}")
        
        # Save checkpoint if this is the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(save_dir, BEST_MODEL_FILENAME)
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'val_f1': val_f1,
            }, best_model_path)
            
            print(f"New best model saved with accuracy: {val_acc:.4f}")
            
        # Save model periodically
        # if epoch % 1 == 0:
        #     checkpoint_path = os.path.join(save_dir, f"bert-model-epoch-{epoch}.pt")
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'val_accuracy': val_acc,
        #         'val_f1': val_f1,
        #     }, checkpoint_path)
    
    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BERT model for fake news detection")
    parser.add_argument(
        "--dataset",
        type=str,
        default="gossipcop",
        choices=["gossipcop", "politifact"],
        help="Which dataset to use."
    )
    args = parser.parse_args()

    config = Config()
    
    # Print training configuration
    print("\nTraining configuration:")
    config.dump()
    print(f"Dataset: {args.dataset}")
    print(f"Device: {config.device}")
    print("")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    
    # Define dataset paths
    train_csv_path = os.path.join(DATASET_FOLDER, f"{args.dataset}_train.csv")
    print(f"Loading training data from {train_csv_path}")
    
    # Get data loaders
    train_loader, val_loader = get_dataloader(
        train_csv_path, 
        tokenizer, 
        batch_size=config.batch_size
    )
    
    # Initialize BERT model
    model = BertForSequenceClassification.from_pretrained(
        config.model_name, 
        num_labels=2
    ).to(config.device)
    
    # Ensure ALL necessary directories exist
    ensure_dir(MODEL_LOG_FOLDER)  # Main logs folder
    # dataset_dir = os.path.join(MODEL_LOG_FOLDER, args.dataset)
    # ensure_dir(dataset_dir)  # Dataset-specific directory
    
    # Start training
    train_model(model, train_loader, val_loader, config, args.dataset)