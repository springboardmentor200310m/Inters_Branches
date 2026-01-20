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


## Milestone 2: CNN Model Development

This milestone focuses on building and training a Convolutional Neural Network (CNN) for multi-label instrument classification from audio spectrograms.

### Architecture
- **Input**: Mel-Spectrogram images (128x128 or 224x224 pixels)
- **Layers**: 
  - Multiple Conv2D layers with ReLU activation
  - MaxPooling for spatial downsampling
  - Batch Normalization for training stability
  - Dropout layers for regularization
  - Fully connected layers for classification
- **Output**: Multi-label predictions (sigmoid activation for each instrument class)

### Training Pipeline

1. **Prepare Training Data**
   ```bash
   python train.py
   ```
   - Loads spectrograms from `dataset/train/`, `dataset/val/`, `dataset/test/`
   - Applies data augmentation (optional: rotation, noise, time-shift)
   - Uses PyTorch DataLoader for batch processing

2. **Model Training**
   - **Loss Function**: Binary Cross-Entropy (multi-label) or Cross-Entropy (single-label)
   - **Optimizer**: Adam with learning rate scheduling
   - **Epochs**: 20-50 epochs with early stopping
   - **Checkpointing**: Saves best model based on validation accuracy

3. **Quick Training (for testing)**
   ```bash
   python quick_train.py
   ```

### Milestone Achievement
✅ **Initial CNN model achieves baseline accuracy** (saved as `models/instrunet_baseline.pth`)

---

## Milestone 3: Model Evaluation & Tuning

This milestone involves rigorous evaluation, hyperparameter optimization, and handling complex multi-instrument scenarios.

### Evaluation Metrics

1. **Run Model Evaluation**
   ```bash
   python evaluate.py
   ```
   - Generates confusion matrix
   - Computes precision, recall, F1-score per class
   - Calculates overall accuracy on test set
   - Saves evaluation reports to `reports/`

2. **Diagnostic Tools**
   ```bash
   python diagnose_model.py
   python diagnose_all.py
   python check_variance.py
   ```
   - Identifies overfitting/underfitting
   - Analyzes class-wise performance
   - Checks data distribution and variance

### Hyperparameter Tuning

Key parameters optimized:
- **Learning Rate**: 0.001 → 0.0001 (with decay)
- **Batch Size**: 16, 32, 64
- **Filter Sizes**: 32, 64, 128, 256
- **Kernel Sizes**: 3x3, 5x5
- **Dropout Rate**: 0.3, 0.5
- **Regularization**: L2 weight decay

### Advanced Training

```bash
python deep_train.py
```
- Extended training with augmentation
- Learning rate scheduling
- Multi-GPU support (if available)

### Multi-Instrument Handling
- **Multi-label classification**: Detects multiple instruments in a single track
- **Threshold tuning**: Adjusts confidence threshold for presence detection
- **Timeline analysis**: Tracks instrument activity over time

### Milestone Achievement
✅ **Optimized model with satisfactory accuracy** (saved as `models/instrunet_final.pth`)

---

## Milestone 4: Deployment & Visualization

This milestone delivers a production-ready system with interactive visualization and comprehensive reporting capabilities.

### Streamlit Web Dashboard

**Launch the application:**
```bash
streamlit run app.py
```

**Features:**
- 🎵 **Audio Upload**: Supports MP3/WAV files
- 🎼 **Real-time Analysis**: CNN-based instrument detection
- 📈 **Waveform Visualization**: Interactive audio waveform display
- 🌈 **Mel-Spectrogram**: Visual frequency representation
- 📊 **Confidence Scores**: Per-instrument detection confidence
- ⏱ **Activity Timeline**: Temporal instrument presence tracking
- 🎨 **Premium UI**: Dark mode with glassmorphism design

### Inference Engine

```bash
python inference.py
```
- Loads trained model (`instrunet_final.pth`)
- Processes audio files in batch
- Outputs JSON predictions

### Report Generation

The system automatically generates:

1. **JSON Reports** (`reports/*.json`)
   - Structured prediction data
   - Confidence scores per instrument
   - Timeline data for visualization

2. **PDF Reports** (`reports/*.pdf`)
   - Professional formatted reports
   - Embedded visualizations (waveform, spectrogram, timeline)
   - Summary statistics and metadata

### Production Pipeline

**End-to-end workflow:**
```
Audio Input → Preprocessing → Spectrogram Generation → 
CNN Inference → Multi-label Detection → Visualization → 
Report Export (JSON/PDF)
```

### Key Components

- **`app.py`**: Streamlit dashboard (main entry point)
- **`inference_engine.py`**: Production inference pipeline
- **`audio_processor.py`**: Audio preprocessing utilities
- **`report_generator.py`**: JSON/PDF export functionality
- **`model.py`**: CNN architecture definition
- **`multilabel_model.py`**: Multi-label classification variant

### Milestone Achievement
✅ **End-to-end pipeline from audio input to instrument recognition and report generation**

---

## Running the Complete System

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the dashboard**
   ```bash
   streamlit run app.py
   ```

3. **Access the web interface**
   - Open browser to `http://localhost:8501`
   - Upload an audio file
   - Click "Analyze Track"
   - View results and download reports

---

## Project Status

| Milestone | Status | Deliverable |
|-----------|--------|-------------|
| 1. Data Collection & Preprocessing | ✅ Complete | Clean labeled dataset with spectrograms |
| 2. CNN Model Development | ✅ Complete | Baseline model (`instrunet_baseline.pth`) |
| 3. Model Evaluation & Tuning | ✅ Complete | Optimized model (`instrunet_final.pth`) |
| 4. Deployment & Visualization | ✅ Complete | Production Streamlit dashboard |

---

## Technologies Used

- **Deep Learning**: PyTorch, torchvision
- **Audio Processing**: librosa, soundfile, pydub
- **Visualization**: matplotlib, seaborn, streamlit
- **Data Science**: numpy, pandas, scikit-learn
- **Reporting**: fpdf2, JSON

---
