import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from model import InstruNetCNN
import pickle
import shutil

# Configuration
DATASET_DIR = "dataset"
BATCH_SIZE = 32
IMAGE_SIZE = (128, 128)

def evaluate_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset 
    # We evaluate on TEST set
    test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    
    test_dir = os.path.join(DATASET_DIR, 'test')
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    classes = test_dataset.classes
    num_classes = len(classes)
    
    # Load Model
    model = InstruNetCNN(num_classes=num_classes).to(device)
    model_path = f"models/instrunet_{args.name}.pth"
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found!")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    y_true = []
    y_pred = []
    misclassified = []

    print("Running evaluation on Test Set...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
            # Identify misclassified
            wrong_indices = (preds != labels).nonzero()
            for idx in wrong_indices:
                idx = idx.item()
                # Global index in dataset? 
                # Calculating global index is tricky with DataLoader shuffling (if enabled, here false)
                # But simple way: file path from dataset samples
                # Batch index local to global:
                global_idx = i * BATCH_SIZE + idx
                path, _ = test_dataset.samples[global_idx]
                actual_class = classes[labels[idx]]
                pred_class = classes[preds[idx]]
                
                misclassified.append({
                    'path': path,
                    'actual': actual_class,
                    'predicted': pred_class
                })

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(f'reports/classwise_metrics_{args.name}.csv')
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix ({args.name})')
    plt.savefig(f'reports/confusion_matrix_{args.name}.png')
    plt.close()
    
    # Save Misclassified Images
    if args.save_misclassified:
        base_mis_dir = f'reports/misclassified_{args.name}'
        if os.path.exists(base_mis_dir):
            shutil.rmtree(base_mis_dir)
        os.makedirs(base_mis_dir)
        
        # Save top 20
        for item in misclassified[:20]:
            # Structure: pred_calss/filename
            dest_dir = os.path.join(base_mis_dir, item['predicted'])
            os.makedirs(dest_dir, exist_ok=True)
            fname = os.path.basename(item['path'])
            dest_path = os.path.join(dest_dir, f"Actual_{item['actual']}_{fname}")
            shutil.copy2(item['path'], dest_path)
            
    # Plot History Curves if available
    hist_path = f'reports/history_{args.name}.pkl'
    if os.path.exists(hist_path):
        with open(hist_path, 'rb') as f:
            history = pickle.load(f)
        
        epochs = range(1, len(history['train_acc']) + 1)
        
        # Accuracy Gap Calculation
        train_acc_final = history['train_acc'][-1]
        val_acc_final = history['val_acc'][-1]
        gap = train_acc_final - val_acc_final
        
        with open(f'reports/overfitting_analysis_{args.name}.txt', 'w') as f:
            f.write(f"Train Accuracy: {train_acc_final:.4f}\n")
            f.write(f"Val Accuracy: {val_acc_final:.4f}\n")
            f.write(f"Accuracy Gap: {gap:.4f}\n")
            if gap > 0.10:
                f.write("Status: OVERFITTING LIKELY (Gap > 10%)\n")
            elif train_acc_final < 0.60:
                f.write("Status: UNDERFITTING LIKELY (Train Acc < 60%)\n")
            else:
                f.write("Status: BALANCED\n")

        # Plot Accuracy
        plt.figure()
        plt.plot(epochs, history['train_acc'], 'b-', label='Training Acc')
        plt.plot(epochs, history['val_acc'], 'r-', label='Validation Acc')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.savefig(f'reports/accuracy_curve_{args.name}.png')
        plt.close()
        
        # Plot Loss
        plt.figure()
        plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(f'reports/loss_curve_{args.name}.png')
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='Name of the run (baseline or improved)')
    parser.add_argument('--save_misclassified', action='store_true', default=True)
    args = parser.parse_args()
    
    evaluate_model(args)
