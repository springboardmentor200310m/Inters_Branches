import os
import shutil
import random
import pandas as pd
from tqdm import tqdm

# Configuration
SOURCE_DIR = 'spectrograms'
DATASET_DIR = 'dataset'
SPLIT_RATIOS = {'train': 0.7, 'val': 0.15, 'test': 0.15}

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Source directory '{SOURCE_DIR}' not found. Run generate_spectrograms.py first.")
        return

    # Clear existing dataset dir if exists to avoid mixing
    if os.path.exists(DATASET_DIR):
        print(f"Removing existing '{DATASET_DIR}'...")
        shutil.rmtree(DATASET_DIR)

    create_dir(DATASET_DIR)
    for split in ['train', 'val', 'test']:
        create_dir(os.path.join(DATASET_DIR, split))

    instruments = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    instruments.sort()
    
    csv_data = []

    print(f"Beginning dataset split: {SPLIT_RATIOS}")
    print(f"Found {len(instruments)} classes to split: {instruments}")

    for instrument in instruments:
        instrument_dir = os.path.join(SOURCE_DIR, instrument)
        
        # Collect all spectrogram images for this class
        files = []
        for root, dirs, filenames in os.walk(instrument_dir):
            for f in filenames:
                if f.lower().endswith('.png'):
                    files.append(f)
        
        if not files:
            print(f"Warning: No images found for class '{instrument}'. Skipping.")
            continue

        # Shuffle files
        random.seed(42)
        random.shuffle(files)
        
        total_files = len(files)
        train_count = int(total_files * SPLIT_RATIOS['train'])
        val_count = int(total_files * SPLIT_RATIOS['val'])
        # test takes the rest to ensure no loss due to rounding
        
        splits = {
            'train': files[:train_count],
            'val': files[train_count:train_count+val_count],
            'test': files[train_count+val_count:]
        }
        
        print(f"Splitting '{instrument}': Total={total_files} -> Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
        
        for split, split_files in splits.items():
            split_dir = os.path.join(DATASET_DIR, split, instrument)
            create_dir(split_dir)
            
            for f in split_files:
                # We need the full path to copy
                # Since we collected just filenames, we assume they are in instrument_dir.
                # BUT if os.walk found them in subdirs, we have a problem.
                # Let's simple assume flat structure in spectrograms/Instrument/
                # If generate_spectrograms produced subdirs (it didn't), we'd need full paths.
                # generate_spectrograms uses a flat structure inside each instrument folder.
                src = os.path.join(instrument_dir, f)
                dst = os.path.join(split_dir, f)
                shutil.copy2(src, dst)
                
                csv_data.append({
                    'filename': f,
                    'file_path': os.path.join(split, instrument, f).replace('\\', '/'), # Relative path for easy loading
                    'instrument': instrument,
                    'split': split
                })

    # Save CSV
    df = pd.DataFrame(csv_data)
    df.to_csv('labels.csv', index=False)
    print(f"Dataset split complete. 'labels.csv' generated with {len(df)} entries.")

if __name__ == "__main__":
    main()
