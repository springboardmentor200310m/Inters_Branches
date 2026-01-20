import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from segmenter import AudioSegmenter
from inference import InferenceEngine

# Mock classes if we don't have a dataset loaded
DEFAULT_CLASSES = ['Accordion', 'Acoustic_Guitar', 'Banjo', 'Bass_Guitar', 'Clarinet', 'Cymbals', 
                   'Drum_set', 'Flute', 'Piano', 'Saxophone', 'Trumpet', 'Violin']

def main():
    parser = argparse.ArgumentParser(description="InstruNet AI - Music Instrument Detection")
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--model', type=str, default='models/instrunet_final.pth', help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--save_report', action='store_true', default=True, help='Generate PDF/JSON reports')
    args = parser.parse_args()
    
    # 1. Setup
    if not os.path.exists(args.audio):
        print(f"Error: Audio file '{args.audio}' not found.")
        return

    # Try to detect classes from dataset folder if possible, else use default
    classes = DEFAULT_CLASSES
    if os.path.exists('dataset/train'):
        classes = sorted(os.listdir('dataset/train'))
        
    print(f"Initializing Inference Engine with {len(classes)} classes...")
    engine = InferenceEngine(args.model, classes)
    segmenter = AudioSegmenter()
    
    # 2. Segment Audio
    print(f"Processing audio: {args.audio}")
    try:
        segments = segmenter.load_and_segment(args.audio)
    except Exception as e:
        print(f"Error reading audio: {e}")
        return
        
    print(f"Generated {len(segments)} segments (3s windows). Running inference...")
    
    # 3. Run Inference
    timeline_data = [] # List of {'timestamp': t, 'predictions': {inst: prob}}
    
    for seg in segments:
        preds = engine.predict_segment(seg['audio'], threshold=args.threshold)
        timeline_data.append({
            'timestamp': seg['timestamp'],
            'predictions': preds
        })
        
        # Live log
        if preds:
            top_inst = list(preds.keys())[0] if preds else "Silence"
            print(f"[{seg['timestamp']:.1f}s] Detected: {list(preds.keys())}")
            
    # 4. Generate Reports
    if args.save_report:
        generate_reports(args.audio, timeline_data, classes)

def generate_reports(audio_path, timeline_data, classes):
    os.makedirs('reports', exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # JSON
    json_path = f"reports/{base_name}_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(timeline_data, f, indent=4)
    print(f"Saved analysis to {json_path}")
    
    # Timeline Plot
    plot_timeline(base_name, timeline_data, classes)
    
    # Text Summary
    txt_path = f"reports/{base_name}_report.txt"
    with open(txt_path, 'w') as f:
        f.write(f"Inference Report for: {audio_path}\n")
        f.write("="*40 + "\n\n")
        
        # Aggregate counts
        counts = {c: 0 for c in classes}
        for item in timeline_data:
            for p in item['predictions']:
                counts[p] += 1
        
        sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
        
        f.write("Detected Instruments Frequency (Segments):\n")
        for inst, count in sorted_counts.items():
            if count > 0:
                f.write(f"- {inst}: {count}\n")
                
    print(f"Saved summary to {txt_path}")

def plot_timeline(name, data, classes):
    # Prepare data matrix: (num_classes, num_segments)
    times = [d['timestamp'] for d in data]
    matrix = np.zeros((len(classes), len(data)))
    
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    for t_idx, item in enumerate(data):
        for inst, prob in item['predictions'].items():
            if inst in class_to_idx:
                matrix[class_to_idx[inst], t_idx] = prob
                
    plt.figure(figsize=(15, 8))
    import seaborn as sns
    sns.heatmap(matrix, yticklabels=classes, cmap='viridis', vmin=0, vmax=1)
    plt.xlabel('Time Segment Index (3s window)')
    plt.ylabel('Instrument')
    plt.title(f'Instrument Activity Timeline - {name}')
    plt.tight_layout()
    plt.savefig(f"reports/{name}_timeline.png")
    plt.close()
    print(f"Saved timeline to reports/{name}_timeline.png")

if __name__ == "__main__":
    main()
