import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
import io

class AudioProcessor:
    def __init__(self, sample_rate=22050, n_mels=128, n_fft=2048, hop_length=512, image_size=(128, 128)):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_spectrogram_image(self, y):
        """
        Converts audio signal to a Mel-Spectrogram image (PIL) exactly as during training.
        """
        # Generate Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=self.sample_rate, n_mels=self.n_mels, 
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Use matplotlib to apply the 'magma' colormap, matching generate_spectrograms.py
        # We need to save it to a buffer and read it back as PIL to get the RGB channels
        buf = io.BytesIO()
        plt.imsave(buf, mel_spec_db, cmap='magma', origin='lower', format='png')
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img

    def preprocess_chunk(self, y):
        """
        Converts numeric audio chunk to a tensor ready for CNN.
        """
        img = self.get_spectrogram_image(y)
        tensor = self.transform(img)
        return tensor.unsqueeze(0) # Add batch dimension

    def load_audio(self, file_path):
        """
        Loads audio, converts to mono, trims silence, and normalizes.
        """
        y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        # Trim silence to match training
        y, _ = librosa.effects.trim(y, top_db=20)
        # Normalize
        if len(y) > 0:
            y = librosa.util.normalize(y)
        return y, sr

    def split_into_chunks(self, y, duration_sec=3.0, overlap=0.5):
        """
        Splits audio into overlapping chunks.
        """
        chunk_size = int(duration_sec * self.sample_rate)
        hop_length = int(chunk_size * (1 - overlap))
        
        chunks = []
        for i in range(0, len(y) - chunk_size + 1, hop_length):
            chunks.append(y[i:i + chunk_size])
            
        # If the last bit is smaller than chunk_size, we might ignore it or pad.
        # For now, let's keep it simple.
        return chunks
