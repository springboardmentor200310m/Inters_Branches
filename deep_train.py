import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from model import InstruNetCNN
from tqdm import tqdm

# High-Performance Quick Training Configuration
DATASET_DIR = "dataset"
BATCH_SIZE = 32
IMAGE_SIZE = (128, 128)
LEARNING_RATE = 0.0005 # Reduced LR for stability
EPOCHS = 3 
SAMPLES_PER_CLASS = 150 # Increased from 10

def deep_train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Deep Accuracy Optimization on {device}...")
    
    # Advanced Data Augmentation
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(DATASET_DIR, 'train')
    if not os.path.exists(train_dir):
        print("Dataset not found!")
        return

    full_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    
    # Controlled subsetting for speed vs accuracy balance
    indices = []
    class_counts = {}
    
    # Shuffle indices manually for random subsetting
    all_indices = list(range(len(full_dataset)))
    import random
    random.seed(42)
    random.shuffle(all_indices)
    
    for i in all_indices:
        _, label = full_dataset.samples[i]
        if class_counts.get(label, 0) < SAMPLES_PER_CLASS:
            indices.append(i)
            class_counts[label] = class_counts.get(label, 0) + 1
    
    subset = torch.utils.data.Subset(full_dataset, indices)
    train_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
    
    num_classes = len(full_dataset.classes)
    model = InstruNetCNN(num_classes=num_classes).to(device)
    
    # Use Weighted cross entropy or label smoothing if classes are imbalanced? 
    # For now, standard CE with AdamW
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f"{running_loss/len(train_loader):.4f}", 'acc': f"{100.*correct/total:.2f}%"})
            
        scheduler.step()
        
        # Intermediate Save
        torch.save(model.state_dict(), 'models/instrunet_final.pth')
        print(f"Epoch {epoch+1} complete. Model checkpointed.")
        
    print(f"\n✅ Optimization Complete. Final accuracy on subset: {100.*correct/total:.2f}%")
    print("Locked & Saved: models/instrunet_final.pth")

if __name__ == "__main__":
    deep_train()
