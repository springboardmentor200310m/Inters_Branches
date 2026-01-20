# InstaNet AI - CNN-Based Music Instrument Recognition System

## Milestone 1: Data Collection & Preprocessing Pipeline

This project builds a clean, labeled, preprocessed dataset for training a CNN to recognize musical instruments from audio spectrograms.

### Directory Structure
- `music_dataset/`: Raw audio files (wav, mp3) organized by instrument folders.
- `processed_audio/`: Intermediate audio segments (mono, 22050Hz, trimmed, 3s chunks).
- `spectrograms/`: Generated Mel-Spectrogram images (PNG).
- `dataset/`: Final dataset split into Train/Val/Test.
- `labels.csv`: Metadata mapping filenames to instruments and splits.

### Instruments
Supported instruments (folder structure based):
- Accordion
- Acoustic Guitar
- Banjo
- Bass Guitar
- Clarinet
- Cymbals
- Drum Set
- Electric Guitar
- Piano
- Saxophone
- Trumpet
- Ukulele
- Violin
- Flute
- Vibraphone
- etc.

### Usage

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place Raw Data**
   Ensure your raw audio files are in `music_dataset/{InstrumentName}/`.

3. **Run Preprocessing**
   Convert to mono, normalize, trim silence, and split into 3-second segments.
   ```bash
   python preprocess.py
   ```

4. **Generate Spectrograms**
   Create Mel-Spectrogram images from the processed audio.
   ```bash
   python generate_spectrograms.py
   ```

5. **Split Dataset**
   Organize into Train (70%), Validation (15%), and Test (15%) sets and generate `labels.csv`.
   ```bash
   python dataset_split.py
   ```

### Output
- **processed_audio/**: Contains `.wav` segments.
- **spectrograms/**: Contains `.png` spectrograms.
- **dataset/**: Contains the final folder structure for CNN training.
- **labels.csv**: CSV file with columns `filename`, `instrument`, `split`.
