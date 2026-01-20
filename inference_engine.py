import torch
import numpy as np
from model import InstruNetCNN
from audio_processor import AudioProcessor
import os

class InferenceEngine:
    def __init__(self, model_path, classes, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = classes
        self.num_classes = len(classes)
        
        # Initialize and load model
        self.model = InstruNetCNN(num_classes=self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        
        self.processor = AudioProcessor()

    def moving_average(self, data, window=5):
        """
        Applies a moving average filter to smooth predictions.
        """
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='same')

    def predict_chunks(self, chunks):
        """
        Runs inference on multiple chunks and returns raw probabilities for each.
        """
        all_probs = []
        with torch.no_grad():
            for chunk in chunks:
                tensor = self.processor.preprocess_chunk(chunk).to(self.device)
                logits = self.model(tensor)
                # Apply sigmoid for multi-label output
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                all_probs.append(probs)
        
        return np.array(all_probs)

    def process_full_audio(self, audio_path, chunk_duration=3.0, threshold=0.6, top_k=5):
        """
        Production pipeline: load -> chunk -> predict -> smooth -> aggregate -> filter.
        """
        y, sr = self.processor.load_audio(audio_path)
        chunks = self.processor.split_into_chunks(y, duration_sec=chunk_duration, overlap=0.5)
        
        if not chunks:
            # Handle short audio by padding to at least one chunk duration
            target_samples = int(chunk_duration * sr)
            y_padded = np.pad(y, (0, max(0, target_samples - len(y))))
            chunks = [y_padded]
            
        chunk_probs = self.predict_chunks(chunks)
        
        # 1. Temporal Smoothing per class
        smoothed_probs = np.zeros_like(chunk_probs)
        for i in range(self.num_classes):
            smoothed_probs[:, i] = self.moving_average(chunk_probs[:, i], window=5)
            
        # 2. Aggregate chunk predictions (mean of smoothed)
        overall_probs = np.mean(smoothed_probs, axis=0)
        
        # 3. Apply Confidence Threshold & Top-K Filtering
        # Get indices of top K instruments based on average confidence
        top_indices = np.argsort(overall_probs)[::-1][:top_k]
        top_classes = [self.classes[i] for i in top_indices]
        
        results = {}
        for i, cls_name in enumerate(self.classes):
            # Only mark as 'present' if it's in top K AND above threshold
            is_top = (cls_name in top_classes)
            is_above_thresh = (overall_probs[i] >= threshold)
            
            results[cls_name] = {
                "confidence": float(overall_probs[i]),
                "present": bool(is_top and is_above_thresh),
                "timeline": smoothed_probs[:, i].tolist(),
                "raw_timeline": chunk_probs[:, i].tolist()
            }
            
        return results, y, sr
