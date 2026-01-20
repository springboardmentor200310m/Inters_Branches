import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

from data import DatasetLoader
from data_Fetcher import SpectrogramDataset

def load_model(device):
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

    return model

def get_test_loader():
    dl = DatasetLoader()
    _, _, test_df = dl.get_dataframes()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = SpectrogramDataset(
        test_df,
        'test_images',
        transform=transform
    )

    return DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

import json

def load_best_val_accuracy(path="history_resNet.json"):
    with open(path, "r") as f:
        history = json.load(f)

    best_val_acc = max(history["val_acc"])
    return best_val_acc


def run_inference(model, dataloader, device):
    y_true = []
    y_pred = []
    y_prob = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = softmax(outputs)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_prob)

from sklearn.metrics import classification_report

def compute_metrics(y_true, y_pred, class_names):
    print("\n📋 Class-wise Precision / Recall / F1")
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True
    )

    # Pretty print per-class metrics
    for cls in class_names:
        print(
            f"{cls:12s} | "
            f"Precision: {report[cls]['precision']:.4f} | "
            f"Recall: {report[cls]['recall']:.4f} | "
            f"F1: {report[cls]['f1-score']:.4f} | "
            f"Support: {int(report[cls]['support'])}"
        )

    # Macro & weighted summaries
    print("\n📊 Overall Metrics")
    print(f"Macro Precision : {report['macro avg']['precision']:.4f}")
    print(f"Macro Recall    : {report['macro avg']['recall']:.4f}")
    print(f"Macro F1-score  : {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1     : {report['weighted avg']['f1-score']:.4f}")


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
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
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_test.png", dpi=300)
    plt.show()
    plt.close()

from sklearn.preprocessing import label_binarize

def plot_roc_auc(y_true, y_prob, class_names):
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))

    plt.figure(figsize=(8, 6))

    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC–AUC Curve (One-vs-Rest)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("roc_auc_test.png", dpi=300)
    plt.show()
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = [
        'bass', 'brass', 'flute', 'guitar', 'keyboard',
        'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal'
    ]

    model = load_model(device)
    test_loader = get_test_loader()

    y_true, y_pred, y_prob = run_inference(
        model,
        test_loader,
        device
    )

    test_accuracy = (y_true == y_pred).mean() * 100
    print(f"✅ Test Accuracy: {test_accuracy:.2f}%")

    best_val_acc = load_best_val_accuracy()
    print(f"📈 Best Validation Accuracy: {best_val_acc:.2f}%")

    gap = best_val_acc - test_accuracy
    print(f"📉 Validation–Test Accuracy Gap: {gap:.2f}%")




    #compute_metrics(y_true, y_pred,class_names)
    #plot_confusion_matrix(y_true, y_pred, class_names)
    #plot_roc_auc(y_true, y_prob, class_names)
if __name__ == "__main__":
    main()
