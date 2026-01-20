import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from model import InstruNetCNN
from tqdm import tqdm

# Quick Training Configuration
DATASET_DIR = "dataset"
BATCH_SIZE = 16
IMAGE_SIZE = (128, 128)
LEARNING_RATE = 0.001
EPOCHS = 1  # Just to get SOME variance

def quick_train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Quick Training on {device}...")
    
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    train_dir = os.path.join(DATASET_DIR, 'train')
    if not os.path.exists(train_dir):
        print("Dataset not found!")
        return

    # Use a small subset to save time (e.g. first 20 images from each class)
    # We'll create a temporary subset
    full_dataset = datasets.ImageFolder(train_dir, transform=transform)
    
    # Stratified subsetting (approximate)
    indices = []
    class_counts = {}
    for i, (_, label) in enumerate(full_dataset.samples):
        if class_counts.get(label, 0) < 10: # 10 samples per class
            indices.append(i)
            class_counts[label] = class_counts.get(label, 0) + 1
    
    subset = torch.utils.data.Subset(full_dataset, indices)
    train_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
    
    num_classes = len(full_dataset.classes)
    model = InstruNetCNN(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1} loss: {running_loss/len(train_loader)}")

    # Save as FINAL
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/instrunet_final.pth')
    print("Real model weights (small training) saved to models/instrunet_final.pth")

if __name__ == "__main__":
    quick_train()
