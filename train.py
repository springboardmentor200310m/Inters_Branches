from data_Fetcher import SpectrogramDataset,DataPrefetcher
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from data import DatasetLoader
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
import json

def save_history_to_file(history_dict, filename="history_resNet.json"):
    with open(filename, 'w') as f:
        json.dump(history_dict, f, indent=4)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True


    # Replace FC
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 11)
    )
    model = model.to(device)

    dl=DatasetLoader()
    train_df, valid_df,test_df = dl.get_dataframes()

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),

        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomApply([
            transforms.RandomRotation(10)
        ], p=0.3),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),

        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


    train_dataset = SpectrogramDataset(train_df, 'train_images', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,pin_memory=True,num_workers=4)
    valid_dataset = SpectrogramDataset(valid_df, 'valid_images', transform=valid_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False,pin_memory=True,num_workers=4)

    classes = np.unique(train_df["instrument_family"])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["instrument_family"]
    )

    weights = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights,label_smoothing=0.05)

    #criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.layer4.parameters(), "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4
    )




    torch.backends.cudnn.benchmark = True
    scaler = GradScaler()
    patience = 5
    min_delta = 1e-4

    best_val_loss = float('inf')
    epochs_no_improve = 0
    model_save_path = "ResNet.pth"

    history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': []
    }


    for epoch in range(40):
        model.train()
        running_loss = 0.0
        train_loss, train_correct, train_total = 0, 0, 0
        loop_train = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/40")
        train_prefetcher = DataPrefetcher(train_loader, device)
        images, labels = train_prefetcher.next()
    
        while images is not None:
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            predicted = torch.argmax(outputs.data, dim=1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.detach()

            loop_train.set_postfix(loss=loss.item())
            loop_train.update(1)

            images, labels = train_prefetcher.next()
        loop_train.close()
    
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        loop_val = tqdm(total=len(valid_loader), desc=f"Epoch {epoch+1}/40")
        val_prefetcher = DataPrefetcher(valid_loader, device)
        image_val, label_val = val_prefetcher.next()
        with torch.no_grad():
            while image_val is not None:
                with autocast(device_type="cuda"):
                    outputs = model(image_val)
                    loss = criterion(outputs, label_val)
        
                predicted = torch.argmax(outputs.data, dim=1)
                val_total += label_val.size(0)
                val_correct += (predicted == label_val).sum().item()
                val_loss += loss.detach()

                loop_val.set_postfix(loss=f"{loss.item():.4f}")
                loop_val.update(1)
                image_val, label_val = val_prefetcher.next()
        loop_val.close()

        epoch_train_loss = (train_loss / len(train_loader)).item()
        epoch_train_acc = 100 * train_correct/ train_total
        epoch_val_loss = (val_loss / len(valid_loader)).item()
        epoch_val_acc = 100 * val_correct/ val_total

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        save_history_to_file(history)

        if epoch_val_loss < best_val_loss - min_delta:
            best_val_loss=epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "ResNet.pth")
        else:
            epochs_no_improve+=1
            print(f"No improvement in val loss for {epochs_no_improve} epochs")
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

        print(f"Epoch {epoch+1} Summary:")
        print(f"Train Loss: {epoch_train_loss:.4f} | Accuracy: {epoch_train_acc:.2f}%")
        print(f"Valid Loss: {epoch_val_loss:.4f} | Accuracy: {epoch_val_acc:.2f}%")

if __name__ == "__main__":
    main()
