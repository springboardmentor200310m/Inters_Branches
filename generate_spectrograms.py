import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Configuration
SOURCE_DIR = 'processed_audio'
OUTPUT_DIR = 'spectrograms'
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_spectrogram(file_path, instrument, output_dir):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Generate Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Save as image
        filename = os.path.splitext(os.path.basename(file_path))[0]
        save_name = f"{filename}.png"
        save_path = os.path.join(output_dir, instrument, save_name)
        
        create_dir(os.path.dirname(save_path))
        
        # Save the pure image without axes/margins using pyplot
        # Alternatively, use matplotlib.image.imsave to save the raw array mapped to colors
        plt.imsave(save_path, mel_spec_db, cmap='magma', origin='lower')
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    # Dynamically find all subdirectories in processed_audio
    if not os.path.exists(SOURCE_DIR):
        print(f"Source directory '{SOURCE_DIR}' not found. Run preprocess.py first.")
        return

    create_dir(OUTPUT_DIR)
    
    # Get all potential instrument directories
    instruments = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    instruments.sort()
    
    print(f"Found {len(instruments)} instrument classes in '{SOURCE_DIR}':")
    print(instruments)
    
    if not instruments:
        print("No instrument directories found in processed_audio.")
        return
    
    for instrument in instruments:
        instrument_dir = os.path.join(SOURCE_DIR, instrument)
        output_inst_dir = os.path.join(OUTPUT_DIR, instrument)
        
        # Ensure output dir exists
        create_dir(output_inst_dir)

        # Collect all wav files for this instrument
        wav_files = []
        for root, dirs, files in os.walk(instrument_dir):
            for file in files:
                if file.lower().endswith('.wav'):
                    wav_files.append(os.path.join(root, file))
        
        print(f"Generating spectrograms for '{instrument}': {len(wav_files)} files found.")
        
        if not wav_files:
            continue

        for f in tqdm(wav_files, desc=f"Spectrograms: {instrument}", unit="img"):
            generate_spectrogram(f, instrument, OUTPUT_DIR)
            
    print("\nSpectrogram generation complete.")

if __name__ == "__main__":
    main()
