import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from parse_json import parse_json_file
import os
from tqdm import tqdm

def save_spectrogram_as_image(data_dir, output_dir):
    audio_dir = os.path.join(data_dir, 'audio')
    os.makedirs(output_dir, exist_ok=True)
    plt.switch_backend('agg') 
    sample_ids, metadata = parse_json_file(data_dir)
    for sample_id in tqdm(sample_ids):
        input_path = os.path.join(audio_dir, f"{sample_id}.wav")
        output_path = os.path.join(output_dir, f"{sample_id}.png")
        
        if os.path.exists(output_path): continue
            
        try:
            y, sr = librosa.load(input_path, sr=16000, mono=True)
            y = librosa.util.normalize(y) 
            
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            fig = plt.figure(figsize=(2.56, 2.56), dpi=100) 
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
            librosa.display.specshow(S_dB, sr=sr, cmap='magma')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
        except Exception as e:
            print(f"Error at {sample_id}: {e}")

save_spectrogram_as_image('nsynth-train', 'train_images')
save_spectrogram_as_image('nsynth-valid', 'valid_images')
save_spectrogram_as_image('nsynth-test', 'test_images')