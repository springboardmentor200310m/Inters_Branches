import torch
import numpy as np
import os
from model import InstruNetCNN

def diagnose_all():
    train_dir = "dataset/train"
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    num_classes = len(classes)
    
    files = ["instrunet_baseline.pth", "instrunet_improved.pth", "instrunet_final.pth"]
    
    x = torch.randn(1, 3, 128, 128)
    
    for f in files:
        path = os.path.join("models", f)
        if not os.path.exists(path): continue
        
        print(f"\n--- Checking {f} ---")
        model = InstruNetCNN(num_classes=num_classes)
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        model.eval()
        
        with torch.no_grad():
            out = torch.sigmoid(model(x)).numpy()[0]
            print(f"Mean: {np.mean(out):.4f}, Std: {np.std(out):.4f}")
            print(f"Top 3 confidence: {sorted(out, reverse=True)[:3]}")
            
            # Check weight statistics for the last layer
            w = model.fc_layers[-1].weight.detach().numpy()
            print(f"Weight Mean: {np.mean(w):.6f}, Std: {np.std(w):.6f}")

if __name__ == "__main__":
    diagnose_all()
