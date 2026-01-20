import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os

    

class SpectrogramDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['image_id']
        label = self.df.iloc[idx]['instrument_family']
        
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)


class DataPrefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        if device.type == "cuda":
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None

        self.preload()

    def preload(self):
        try:
            self.next_images, self.next_labels = next(self.loader)
        except StopIteration:
            self.next_images = None
            self.next_labels = None
            return

        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.next_images = self.next_images.to(self.device, non_blocking=True)
                self.next_labels = self.next_labels.to(self.device, non_blocking=True)
        else:
            self.next_images = self.next_images.to(self.device)
            self.next_labels = self.next_labels.to(self.device)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        images = self.next_images
        labels = self.next_labels
        self.preload() 
        return images, labels
    




