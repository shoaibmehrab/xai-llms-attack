import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from transformers import BertTokenizer, BertForSequenceClassification

from config import Config
from data import get_test_dataloader
from constants import DATASET_FOLDER, MODEL_LOG_FOLDER, MODEL_RESULTS_FOLDER, BEST_MODEL_FILENAME

def ensure_dir(folder_path: str):
    os.makedirs(folder_path, exist_ok=True)

def save_results_to_file(metrics, all_labels, all_preds, all_probs, dataset_name, comment_mode):
    """Save evaluation results to a text file"""
    results_dir = os.path.join(MODEL_RESULTS_FOLDER, dataset_name)
    ensure_dir(results_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{comment_mode}_results_{timestamp}.txt'
    filepath = os.path.join(results_dir, filename)
    
    # Calculate metrics
    acc = metrics['accuracy']
    prec = metrics['precision']
    rec = metrics['recall']
    f1 = metrics['f1']
    auc_roc = metrics['auc']
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Generate report
    class_report = classification_report(all_labels, all_preds, target_names=['Real', 'Fake'])
    
    # Write to file
    with open(filepath, 'w') as f:
        f.write(f"=== Evaluation Results for {dataset_name} ({comment_mode} mode) ===\n\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"AUC-ROC:   {auc_roc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(class_report)
    
    print(f"Results saved to {filepath}")
    
    # Also update a summary file
    summary_file = os.path.join(results_dir, f'summary.csv')
    header = not os.path.exists(summary_file)
    
    with open(summary_file, 'a') as f:
        if header:
            f.write("timestamp,dataset,comment_mode,accuracy,precision,recall,f1,auc\n")
        f.write(f"{timestamp},{dataset_name},{comment_mode},{acc:.4f},{prec:.4f},{rec:.4f},{f1:.4f},{auc_roc:.4f}\n")
    
    print(f"Summary updated in {summary_file}")

def plot_roc_curve(fpr, tpr, roc_auc, dataset_name, comment_mode):
    """Plot ROC curve and save to file"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name} ({comment_mode} mode)')
    plt.legend(loc="lower right")
    
    plots_dir = os.path.join(MODEL_RESULTS_FOLDER, dataset_name, 'plots')
    ensure_dir(plots_dir)
    
    plt.savefig(os.path.join(plots_dir, f'roc_curve_{comment_mode}.png'))
    plt.close()

def plot_confusion_matrix(cm, dataset_name, comment_mode):
    """Plot confusion matrix and save to file"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {dataset_name} ({comment_mode} mode)')
    plt.colorbar()
    
    classes = ['Real', 'Fake']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plots_dir = os.path.join(MODEL_RESULTS_FOLDER, dataset_name, 'plots')
    ensure_dir(plots_dir)
    
    plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{comment_mode}.png'))
    plt.close()

def evaluate_model(model, test_loader, config):
    """
    Evaluate model and return predictions and ground truth for metrics calculation
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # For ROC curve
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (fake)
    
    return all_labels, all_preds, all_probs

def display_metrics(y_true, y_pred, y_probs, dataset_name, comment_mode):
    """Calculate and display comprehensive metrics"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    print("\n" + "="*50)
    print(f"EVALUATION METRICS FOR {dataset_name} ({comment_mode} mode)")
    print("="*50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {roc_auc:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot ROC curve
    plot_roc_curve(fpr, tpr, roc_auc, dataset_name, comment_mode)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, dataset_name, comment_mode)
    
    # Return metrics for saving to file
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': roc_auc
    }
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BERT model for fake news detection")
    parser.add_argument(
        "--dataset",
        type=str,
        default="gossipcop",
        choices=["gossipcop", "politifact"],
        help="Which dataset to use."
    )
    parser.add_argument(
        "--comment-mode",
        type=str,
        default="all",
        choices=["all", "none", "limited"],
        help="How to handle comments: 'all' (content + all comments), 'none' (only content), 'limited' (limited comments)"
    )
    args = parser.parse_args()

    config = Config()
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    
    # Define dataset paths
    test_csv_path = os.path.join(DATASET_FOLDER, f"{args.dataset}_test.csv")
    print(f"Loading test data from {test_csv_path}")
    
    # Get test data loader
    test_loader = get_test_dataloader(
        test_csv_path,
        tokenizer,
        batch_size=config.batch_size,
        comment_mode=args.comment_mode
    )

    print(f"--- Evaluating with comment_mode: {args.comment_mode} ---") # <-- Add this line
    
    # Load the best model for this dataset and comment mode
    model_path = os.path.join(MODEL_LOG_FOLDER, args.dataset, BEST_MODEL_FILENAME)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        exit(1)
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=config.device)
    
    model = BertForSequenceClassification.from_pretrained(
        config.model_name, 
        num_labels=2
    ).to(config.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation accuracy {checkpoint['val_accuracy']:.4f}")
    
    # Evaluate the model
    all_labels, all_preds, all_probs = evaluate_model(model, test_loader, config)
    
    # Display and save metrics
    metrics = display_metrics(all_labels, all_preds, all_probs, args.dataset, args.comment_mode)
    
    # Save results to file
    save_results_to_file(metrics, all_labels, all_preds, all_probs, args.dataset, args.comment_mode)