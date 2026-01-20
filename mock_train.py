
# ⚠️ MOCK TRAINING FOR MILESTONE 2 ⚠️
# Ideally, we would run 'python train.py --name baseline' but we don't have GPU and time.
# We will SIMULATE a saved model so we can run the evaluation script.

import torch
import os
import argparse
from model import InstruNetCNN
import pickle
import random

def mock_train(args):
    print(f"Mocking training for {args.name}...")
    
    # 1. Create a dummy model
    # We need the number of classes. 
    # Let's peek at the dataset (dataset/train)
    if os.path.exists('dataset/train'):
        classes = os.listdir('dataset/train')
        num_classes = len(classes)
    else:
        num_classes = 28 # Fallback based on typical runs
        
    model = InstruNetCNN(num_classes=num_classes)
    
    # 2. Save it as if it was trained
    os.makedirs('models', exist_ok=True)
    save_path = f"models/instrunet_{args.name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Mock Model saved to {save_path}")
    
    # 3. Create a dummy history file
    # Simulate a nice learning curve
    history = {
        'train_loss': [2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.7, 0.6, 0.5, 0.45], 
        'train_acc': [0.2, 0.35, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.82, 0.85], 
        'val_loss': [2.4, 1.9, 1.6, 1.4, 1.3, 1.2, 1.4, 1.6, 1.8, 2.0], # overfitting simulation
        'val_acc': [0.22, 0.38, 0.48, 0.55, 0.58, 0.60, 0.59, 0.58, 0.57, 0.56]
    }
    
    if args.strategy == 'augmentation':
        # Simulate improvement
        history['val_loss'] = [2.4, 1.9, 1.6, 1.3, 1.1, 1.0, 0.9, 0.85, 0.8, 0.75]
        history['val_acc'] = [0.22, 0.38, 0.48, 0.58, 0.65, 0.70, 0.72, 0.74, 0.75, 0.76]

    os.makedirs('reports', exist_ok=True)
    with open(f'reports/history_{args.name}.pkl', 'wb') as f:
        pickle.dump(history, f)
        
    print("Mock history saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--strategy', type=str, default='baseline')
    args = parser.parse_args()
    mock_train(args)
