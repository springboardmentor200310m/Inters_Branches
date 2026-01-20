import torch
import numpy as np
import os
from model import InstruNetCNN

def diagnose():
    train_dir = "dataset/train"
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    num_classes = len(classes)
    
    model_path = "models/instrunet_final.pth"
    if not os.path.exists(model_path):
        print("Model not found")
        return

    print(f"Loading model with {num_classes} classes from {model_path}")
    model = InstruNetCNN(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Check some weights
    fc_weights = model.fc_layers[-1].weight.detach().numpy()
    print(f"FC layer weight range: {np.min(fc_weights):.5f} to {np.max(fc_weights):.5f}")
    print(f"FC layer weight std: {np.std(fc_weights):.5f}")
    
    # Run on noise
    with torch.no_grad():
        x1 = torch.randn(1, 3, 128, 128)
        x2 = torch.randn(1, 3, 128, 128)
        
        out1 = torch.sigmoid(model(x1)).numpy()[0]
        out2 = torch.sigmoid(model(x2)).numpy()[0]
        
        print("\nNoise Analysis (Random Input 1):")
        print(f"Min: {np.min(out1):.4f}, Max: {np.max(out1):.4f}, Mean: {np.mean(out1):.4f}")
        
        print("\nNoise Analysis (Random Input 2):")
        print(f"Min: {np.min(out2):.4f}, Max: {np.max(out2):.4f}, Mean: {np.mean(out2):.4f}")
        
        diff = np.abs(out1 - out2)
        print(f"\nMax difference between two random inputs: {np.max(diff):.6f}")

if __name__ == "__main__":
    diagnose()
