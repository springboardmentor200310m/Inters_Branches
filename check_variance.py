import torch
from inference_engine import InferenceEngine
import os
import glob
import numpy as np

def check_variance():
    train_dir = "dataset/train"
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    model_path = "models/instrunet_final.pth"
    
    engine = InferenceEngine(model_path, classes)
    
    # Get two different audio files
    wav_files = glob.glob(os.path.join(train_dir, "*", "*.wav"))
    if len(wav_files) < 2:
        # Check processed_audio
        wav_files = glob.glob(os.path.join("processed_audio", "*", "*.wav"))
        
    if len(wav_files) < 2:
        print("Not enough audio files found to test.")
        return

    f1 = wav_files[0]
    f2 = wav_files[len(wav_files)//2]
    
    print(f"Testing File 1: {f1}")
    res1, _, _ = engine.process_full_audio(f1)
    
    print(f"Testing File 2: {f2}")
    res2, _, _ = engine.process_full_audio(f2)
    
    # Extract confidence scores
    conf1 = np.array([res1[c]['confidence'] for c in classes])
    conf2 = np.array([res2[c]['confidence'] for c in classes])
    
    print("\nResults Analysis:")
    print(f"File 1 Conf Mean: {np.mean(conf1):.4f}, Std: {np.std(conf1):.4f}")
    print(f"File 2 Conf Mean: {np.mean(conf2):.4f}, Std: {np.std(conf2):.4f}")
    
    diff = np.abs(conf1 - conf2)
    print(f"Max difference between files: {np.max(diff):.6f}")
    
    if np.max(diff) < 1e-4:
        print("\nCRITICAL: Model produces NEARLY IDENTICAL results for different real files.")
        print("This confirms either a model loading issue or a model that hasn't learned any features.")

if __name__ == "__main__":
    check_variance()
