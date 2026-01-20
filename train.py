import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import copy
import argparse
import matplotlib.pyplot as plt
from model import InstruNetCNN
from tqdm import tqdm

# Configuration
DATASET_DIR = "dataset"
BATCH_SIZE = 32
IMAGE_SIZE = (128, 128)
LEARNING_RATE = 0.001
EPOCHS = 20

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Data Transforms
    if args.strategy == 'augmentation':
        print("Strategy: Applied Data Augmentation (Random Affine Translation)")
        # Shift in time/freq slightly. Only horizontal shift makes sense for time invariant, 
        # but vertical shift changes pitch. Small vertical shift might be okay? 
        # For music, pitch is key. Let's AVOID vertical shift (frequency).
        # Only time shift (horizontal).
        train_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomApply([
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.0)) # 20% width shift (time)
            ], p=0.5),
            transforms.ToTensor(), # Normalizes to [0,1]
        ])
    else:
        print("Strategy: Baseline (No Augmentation)")
        train_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])

    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    # 2. Datasets
    train_dir = os.path.join(DATASET_DIR, 'train')
    val_dir = os.path.join(DATASET_DIR, 'val')
    
    if not os.path.exists(train_dir):
        print("Error: Dataset not found or split. Run Milestone 1 scripts first.")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    classes = train_dataset.classes
    num_classes = len(classes)
    print(f"Classes found ({num_classes}): {classes}")

    # 3. Model Setup
    model = InstruNetCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloader, desc=phase, leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Record history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

                # Deep copy the model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f'Best Val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    save_path = f"models/instrunet_{args.name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Save History for plotting later
    import pickle
    os.makedirs('reports', exist_ok=True)
    with open(f'reports/history_{args.name}.pkl', 'wb') as f:
        pickle.dump(history, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='Name of the run (baseline or improved)')
    parser.add_argument('--strategy', type=str, default='baseline', help='Strategy: baseline or augmentation')
    args = parser.parse_args()
    
    train_model(args)
