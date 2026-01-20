import torch
import torch.nn as nn
import torchvision.models as models

class MultiLabelInstruNet(nn.Module):
    def __init__(self, num_classes=10, backbone='custom'):
        super(MultiLabelInstruNet, self).__init__()
        
        if backbone == 'resnet18':
            # Use ResNet18
            self.model = models.resnet18(pretrained=True)
            # Replace first conv layer if input is NOT 3 channels? 
            # Our Spectrograms are saved as PNG (RGB/RGBA or Grayscale). 
            # Librosa saves usually as viridis/magma which is RGBA/RGB.
            # ResNet expects 3 channels.
            
            # Replace FC layer
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, num_classes)
            )
        else:
            # Custom CNN (similar to Milestone 2 but optimized)
            self.model = nn.Sequential(
                # Block 1
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                
                # Block 2
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                
                # Block 3
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                
                # Block 4
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                
                nn.Flatten(),
                nn.Linear(256 * 8 * 8, 512), # Assuming 128x128 input -> 8x8 spatial
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        logits = self.model(x)
        return logits # We return logits (BCEWithLogitsLoss handles sigmoid internally during training)
        # During inference, we will look at this and apply torch.sigmoid()

