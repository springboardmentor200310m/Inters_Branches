import torch
import numpy as np
import librosa
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import io

class InferenceEngine:
    def __init__(self, model_path, classes, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = classes
        
        # Load Model
        # We need the architecture definition. For inference, we assume the MultiLabelInstruNet 
        # or the previous InstruNetCNN depending on what was trained. 
        # To make this robust, let's look at the filename or try loading.
        # For Milestone 3, we expect MultiLabelInstruNet.
        from multilabel_model import MultiLabelInstruNet
        self.model = MultiLabelInstruNet(num_classes=len(classes), backbone='custom')
        
        # Handle strict loading if keys match, otherwise ignore (for partial loading or updates)
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Could not load state dict strictly: {e}")
            print("Initializing random weights for demonstration if model not found.")
        
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
    def audio_to_spectrogram_image(self, audio, sr=22050):
        # Generate Mel Spec
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Save to buffer as image to match training pipeline (PNG loading)
        # This handles the colormap mapping
        buf = io.BytesIO()
        plt.imsave(buf, mel_spec_db, cmap='magma', origin='lower')
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img
        
    def predict_segment(self, audio, sr=22050, threshold=0.5):
        # 1. Audio -> Image
        img = self.audio_to_spectrogram_image(audio, sr)
        
        # 2. Image -> Tensor
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # 3. Model Forward
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits)
            
        probs = probs.cpu().numpy()[0]
        
        # 4. Filter by threshold
        predictions = {}
        for i, prob in enumerate(probs):
            if prob >= threshold:
                predictions[self.classes[i]] = float(prob)
                
        # Sort by confidence
        predictions = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))
        
        return predictions

