import os
import librosa
import soundfile as sf
from tqdm import tqdm

INPUT_DIR = "music_dataset"
OUTPUT_DIR = "processed_audio"
SAMPLE_RATE = 22050
SEGMENT_DURATION = 3  # seconds

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_audio(file_path, instrument):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        y, _ = librosa.effects.trim(y, top_db=20)

        segment_length = SEGMENT_DURATION * sr
        total_segments = len(y) // segment_length

        if total_segments == 0:
            return

        out_dir = os.path.join(OUTPUT_DIR, instrument)
        os.makedirs(out_dir, exist_ok=True)

        for i in range(total_segments):
            start = i * segment_length
            end = start + segment_length
            segment = y[start:end]

            out_file = os.path.join(out_dir, f"{instrument}_{os.path.basename(file_path)}_{i}.wav")
            sf.write(out_file, segment, sr)

    except Exception as e:
        print(f"❌ Failed file: {file_path}")
        print("   Error:", e)

def main():
    instruments = sorted(os.listdir(INPUT_DIR))

    print(f"\n🎵 Found {len(instruments)} instrument folders:\n")
    for inst in instruments:
        print(" -", inst)

    for instrument in instruments:
        instrument_path = os.path.join(INPUT_DIR, instrument)

        if not os.path.isdir(instrument_path):
            continue

        print(f"\n▶ Processing instrument: {instrument}")

        files = os.listdir(instrument_path)

        for file in tqdm(files):
            if file.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                file_path = os.path.join(instrument_path, file)
                process_audio(file_path, instrument)

    print("\n✅ Preprocessing completed for all instruments.")

if __name__ == "__main__":
    main()
