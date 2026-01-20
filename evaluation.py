import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from data import DatasetLoader
from data_Fetcher import SpectrogramDataset
from torch.utils.data import DataLoader

def load_history(path):
    with open(path, "r") as f:
        history = json.load(f)
    return history

def plot_accuracy(history):
    epochs = range(1, len(history["train_acc"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_loss(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()



def plot_confusion_matrix(model, dataloader, device, class_names):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 11)
    )

    model.load_state_dict(torch.load("ResNet.pth", map_location=device))
    model.to(device)
    model.eval()


    dl = DatasetLoader()
    _, valid_df, _ = dl.get_dataframes()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    valid_dataset = SpectrogramDataset(
        valid_df,
        'valid_images',
        transform=transform
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    history = load_history("history_resNet.json")

    plot_accuracy(history)
    plot_loss(history)

    plot_confusion_matrix(
        model,
        valid_loader,
        device,
        class_names=[
            'bass', 'brass', 'flute', 'guitar', 'keyboard',
            'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal'
        ]
    )


if __name__ == "__main__":
    main()
