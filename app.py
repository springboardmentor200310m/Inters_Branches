import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import io
import tempfile
from inference_engine import InferenceEngine
from report_generator import ReportGenerator
from audio_processor import AudioProcessor

# Page Config
st.set_page_config(page_title="InstruNet AI - Music Intelligence", layout="wide", page_icon="🎼")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(19, 20, 30) 0%, rgb(10, 10, 15) 100%);
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .status-box {
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.8em;
    }
    .status-present { background-color: #27ae60; color: white; }
    .status-not { background-color: #c0392b; color: white; }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #0e1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #444;
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/isometric-folders/100/music.png", width=80)
    st.title("InstruNet AI")
    st.info("Interactive Audio Intelligence System for Instrument Detection & Visualization.")
    
    threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.3, 0.05)
    st.divider()
    st.caption("v1.3.0 (Accuracy Optimized) | Developed by AI Architect")

# Initialize Engine
@st.cache_resource
def get_engine(version="1.3.0"): # Accuracy Optimized
    # Get classes dynamically
    train_dir = "dataset/train"
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    model_path = "models/instrunet_final.pth"
    if not os.path.exists(model_path):
        # Fallback to baseline if final doesn't exist
        model_path = "models/instrunet_baseline.pth"
    
    return InferenceEngine(model_path, classes)

engine = get_engine(version="1.3.0")
report_gen = ReportGenerator()
processor = AudioProcessor()

# Header
st.title("🎼 Music Instrument Recognition Dashboard")
st.markdown("---")

# Layout
top_col1, top_col2 = st.columns([1, 2])

with top_col1:
    st.subheader("🎵 Audio Input")
    uploaded_file = st.file_uploader("Upload Music Track (MP3/WAV)", type=["wav", "mp3"])
    
    st.markdown("OR")
    record = st.button("🎤 Record from Microphone (Simulated)")
    
    if uploaded_file:
        st.audio(uploaded_file)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # Run Analysis
        if st.button("🚀 Analyze Track", type="primary"):
            with st.spinner("Analyzing audio with CNN..."):
                # Optimized production pipeline call
                results, y, sr = engine.process_full_audio(tmp_path, threshold=threshold, top_k=5)
                st.session_state['results'] = results
                st.session_state['y'] = y
                st.session_state['sr'] = sr
                st.session_state['track_name'] = uploaded_file.name
                
        # Clean up
        # os.remove(tmp_path)

# Results Section
if 'results' in st.session_state:
    results = st.session_state['results']
    y = st.session_state['y']
    sr = st.session_state['sr']
    track_name = st.session_state['track_name']
    
    # Visualization Row 1: Waveform & Spectrogram
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("📈 Waveform")
        fig_wave, ax_wave = plt.subplots(figsize=(10, 4), facecolor='#0e1117')
        librosa.display.waveshow(y, sr=sr, ax=ax_wave, color='#3498db')
        ax_wave.set_axis_off()
        st.pyplot(fig_wave)
        
    with c2:
        st.subheader("🌈 Mel-Spectrogram")
        img_spec = processor.get_spectrogram_image(y)
        st.image(img_spec, use_container_width=True)

    st.markdown("---")
    
    # Results Panel
    st.subheader("🎼 Prediction Results")
    
    present_instruments = [k for k, v in results.items() if v['present']]
    
    col_list, col_bars = st.columns([1, 2])
    
    with col_list:
        st.write("### Detected Instruments")
        if not present_instruments:
            st.warning("No instruments detected above threshold.")
        else:
            for instr in present_instruments:
                conf = results[instr]['confidence']
                st.markdown(f"**{instr}**: <span class='status-box status-present'>Present ({conf*100:.1f}%)</span>", unsafe_allow_html=True)
                st.progress(conf)
            
            st.write("#### 📊 Instrument Intensity (Avg)")
            for instr in present_instruments:
                intensity = int(results[instr]['confidence'] * 20)
                st.text(f"{instr:12}: {'|' * intensity}")
    
    with col_bars:
        st.write("### Confidence Distribution (Top 8)")
        # Plot top classes confidence
        sorted_res = sorted(results.items(), key=lambda x: x[1]['confidence'], reverse=True)[:8]
        names = [x[0] for x in sorted_res]
        confs = [x[1]['confidence'] for x in sorted_res]
        
        fig_bar, ax_bar = plt.subplots(figsize=(10, 5), facecolor='none')
        bars = ax_bar.barh(names[::-1], confs[::-1], color='#9b59b6')
        ax_bar.set_xlim(0, 1.0)
        ax_bar.set_xlabel("Confidence", color='white')
        ax_bar.tick_params(colors='white')
        st.pyplot(fig_bar)

    st.markdown("---")
    
    # Timeline
    st.subheader("⏱ Instrument Activity Timeline")
    
    # Prepare timeline data
    # Only show top 5 active instruments for space
    top_5 = sorted(results.items(), key=lambda x: max(x[1].get('timeline', [0])), reverse=True)[:5]
    
    fig_time, ax_time = plt.subplots(figsize=(12, 6), facecolor='#0e1117')
    for name, data in top_5:
        times = np.linspace(0, len(y)/sr, len(data['timeline']))
        ax_time.plot(times, data['timeline'], label=name, alpha=0.8, linewidth=2)
    
    ax_time.set_xlabel("Time (s)", color='white')
    ax_time.set_ylabel("Intensity / Confidence", color='white')
    ax_time.legend()
    ax_time.tick_params(colors='white')
    ax_time.grid(alpha=0.1)
    st.pyplot(fig_time)

    st.markdown("---")
    
    # Exports
    st.subheader("📄 Report Export Panel")
    ex1, ex2 = st.columns(2)
    
    with ex1:
        if st.button("📥 Download JSON Report"):
            json_path = report_gen.generate_json(track_name, results)
            with open(json_path, 'r') as f:
                st.download_button("Click here to download JSON", f, file_name=os.path.basename(json_path))

    with ex2:
        if st.button("📄 Generate PDF Report"):
            # We need to save the figures as temp images for PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f1, \
                 tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f2, \
                 tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f3, \
                 tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f4:
                
                fig_wave.savefig(f1.name, format='png', bbox_inches='tight')
                img_spec.save(f2.name)
                fig_time.savefig(f3.name, format='png', bbox_inches='tight')
                fig_bar.savefig(f4.name, format='png', bbox_inches='tight')
                
                pdf_path = report_gen.generate_pdf(
                    track_name, results, list(results.keys()), 
                    f1.name, f2.name, f3.name, f4.name
                )
                
                with open(pdf_path, 'rb') as f:
                    st.download_button("Download PDF", f, file_name=os.path.basename(pdf_path))
else:
    st.info("Please upload an audio track and click 'Analyze' to begin.")
