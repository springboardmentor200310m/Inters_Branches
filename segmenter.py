import librosa
import numpy as np
import os
import warnings
import soundfile as sf

warnings.filterwarnings('ignore')

class AudioSegmenter:
    def __init__(self, sample_rate=22050, duration=3.0, overlap=0.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.overlap = overlap
        
    def load_and_segment(self, file_path):
        """
        Loads an audio file and segments it.
        Returns:
            list of dicts: {'timestamp': start_time, 'audio': numpy_array}
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
            
        # Load audio
        y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Calculate window size and hop length
        window_size = int(self.duration * self.sample_rate)
        # overlap is percentage. If 0.5, hop is half window.
        hop_length = int(window_size * (1 - self.overlap))
        
        segments = []
        total_samples = len(y)
        
        for start_sample in range(0, total_samples, hop_length):
            end_sample = start_sample + window_size
            
            # If valid segment
            if end_sample <= total_samples:
                segment_audio = y[start_sample:end_sample]
                timestamp = start_sample / self.sample_rate
                
                segments.append({
                    'timestamp': timestamp,
                    'audio': segment_audio
                })
            else:
                # Handle last segment? Pad?
                pass
                
        return segments

if __name__ == "__main__":
    # Test
    pass
