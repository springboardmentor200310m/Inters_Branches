"""
VANIDYA AI - Streamlit Web Interface
Music Instrument Recognition System
"""

import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from datetime import datetime
import tempfile
import pandas as pd  # Needed for easier plotly handling

# Page configuration
st.set_page_config(
    page_title="VANIDYA AI - Instrument Recognition",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS - Cyber/Dark Theme
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    /* Global settings */
    /* Global settings */
    html, body, .stApp {
        font-family: 'Outfit', sans-serif;
    }

    /* Main App Background */
    .stApp {
        background-color: #0f172a;
        background-image: 
            radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.1) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.15) 0px, transparent 50%);
    }
    
    /* Content Container */
    .block-container {
        padding-top: 2rem;
        max-width: 95rem;
    }
    
    /* Animated Header */
    .main-header {
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(to right, #22d3ee, #e879f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding: 1rem;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
        text-shadow: 0 0 40px rgba(34, 211, 238, 0.3);
        animation: pulseHeader 3s infinite alternate;
    }
    
    @keyframes pulseHeader {
        0% { filter: drop-shadow(0 0 15px rgba(34, 211, 238, 0.3)); }
        100% { filter: drop-shadow(0 0 25px rgba(232, 121, 249, 0.4)); }
    }
    
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #94a3b8;
        margin-bottom: 3rem;
        font-weight: 400;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    /* Instrument Cards */
    .instrument-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .instrument-card:hover {
        transform: translateY(-5px);
        border-color: rgba(34, 211, 238, 0.3);
        box-shadow: 0 10px 40px -10px rgba(0,0,0,0.5);
    }
    
    .detected {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(6, 78, 59, 0.2) 100%);
        border-left: 4px solid #10b981;
    }
    
    .marginal {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(120, 53, 15, 0.2) 100%);
        border-left: 4px solid #f59e0b;
    }
    
    .not-detected {
        background: rgba(30, 41, 59, 0.4);
        border-left: 4px solid #334155;
        opacity: 0.7;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        color: #94a3b8;
        border: 1px solid transparent;
        transition: all 0.2s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(30, 41, 59, 0.8);
        color: #fff;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1e293b;
        border: 1px solid #38bdf8;
        color: #38bdf8 !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.15);
    }
    
    /* File Uploader */
    .stFileUploader section {
        background-color: rgba(30, 41, 59, 0.5);
        border: 2px dashed #334155;
        border-radius: 16px;
        padding: 2rem;
    }
    
    .stFileUploader section:hover {
        border-color: #38bdf8;
        background-color: rgba(30, 41, 59, 0.8);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0b1120;
        border-right: 1px solid #1e293b;
    }
    
    h1, h2, h3 {
        color: #f8fafc;
    }
    
    p, li {
        color: #94a3b8;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(to right, #06b6d4, #3b82f6);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s;
        box-shadow: 0 4px 12px rgba(6, 182, 212, 0.25);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(6, 182, 212, 0.4);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(to right, #22d3ee, #a855f7);
    }
</style>
    """,
    unsafe_allow_html=True,
)

# Configuration
CONFIG = {
    "sample_rate": 22050,
    "mel_bands": [64, 96, 128],
    "n_fft": 2048,
    "hop_length": 512,
    "detection_threshold": 0.50,  # Real-world standard: 0.5 for balanced precision/recall
    "use_spectral_contrast": False,  # Set to False for old model, True for new model with spectral contrast
    # CRITICAL: Instruments MUST be in alphabetical order (lowercase, underscores)
    # This matches the MultiLabelBinarizer(classes=sorted(unique_instruments)) from training
    "instruments": [
        "accordion",
        "acoustic_guitar",
        "banjo",
        "bass_guitar",
        "clarinet",
        "cowbell",
        "cymbals",
        "dobro",
        "drum_set",
        "electric_guitar",
        "floor_tom",
        "flute",
        "harmonica",
        "harmonium",
        "hi_hats",
        "horn",
        "keyboard",
        "mandolin",
        "organ",
        "piano",
        "saxophone",
        "shakers",
        "tambourine",
        "trombone",
        "trumpet",
        "ukulele",
        "vibraphone",
        "violin",
    ],
}


@st.cache_resource
def load_model(model_path):
    """Load the trained model and auto-detect configuration"""
    try:
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)

            # Auto-detect if model uses spectral contrast by inspecting input shapes
            expected_shapes = [inp.shape for inp in model.inputs]
            mel_bands = CONFIG["mel_bands"]

            # Check first input shape: if frequency dimension > mel_bands[0], spectral contrast is used
            first_freq_dim = expected_shapes[0][1]  # (None, freq_dim, time_dim, 1)
            uses_spectral_contrast = first_freq_dim > mel_bands[0]

            # Update CONFIG automatically - REMOVED side effect, returned instead
            # CONFIG["use_spectral_contrast"] = uses_spectral_contrast

            contrast_status = (
                "WITH spectral contrast"
                if uses_spectral_contrast
                else "WITHOUT spectral contrast"
            )
            # st.success(f"✅ Loaded model: {os.path.basename(model_path)} ({contrast_status})")
            # Commented out success message to avoid cluttering UI on every re-run if cached,
            # or usage inside sidebar/main area needs care.
            # But let's keep the logging/info.
            print(
                f"Loaded model: {model_path} ({contrast_status})"
            )  # Log to console instead

            return model, uses_spectral_contrast

        st.error(f"Model file not found: {model_path}")
        return None, False
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, False


def extract_features_from_audio(audio, target_time_dim=259):
    """Extract multi-resolution mel spectrograms with spectral contrast from audio array"""
    try:
        features = []

        for n_mels in CONFIG["mel_bands"]:
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=CONFIG["sample_rate"],
                n_fft=CONFIG["n_fft"],
                hop_length=CONFIG["hop_length"],
                n_mels=n_mels,
                power=2.0,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # CRITICAL FIX: Normalize BEFORE padding/cropping to match training code
            # This ensures the normalization statistics are computed on real audio data only,
            # not including zero-padding which would shift the distribution
            mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)
            mel_db = mel_db.astype(np.float32)

            # Pad or crop mel_db to target_time_dim AFTER normalization
            if mel_db.shape[1] < target_time_dim:
                pad_width = target_time_dim - mel_db.shape[1]
                mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
            elif mel_db.shape[1] > target_time_dim:
                mel_db = mel_db[:, :target_time_dim]

            # Add spectral contrast if enabled (for better guitar distinction)
            if CONFIG.get("use_spectral_contrast", False):
                contrast = librosa.feature.spectral_contrast(
                    y=audio,
                    sr=CONFIG["sample_rate"],
                    n_fft=CONFIG["n_fft"],
                    hop_length=CONFIG["hop_length"],
                    n_bands=6,
                )
                # Normalize contrast BEFORE padding
                contrast = (contrast - np.mean(contrast)) / (np.std(contrast) + 1e-8)

                # Pad or crop contrast to match mel time dimension
                if contrast.shape[1] < target_time_dim:
                    pad_width = target_time_dim - contrast.shape[1]
                    contrast = np.pad(
                        contrast, ((0, 0), (0, pad_width)), mode="constant"
                    )
                elif contrast.shape[1] > target_time_dim:
                    contrast = contrast[:, :target_time_dim]

                # Combine mel spectrogram with spectral contrast AFTER both are normalized
                mel_db = np.vstack([mel_db, contrast])

            features.append(np.expand_dims(mel_db, axis=-1))

        return features
    except Exception as e:
        st.error(f"Error extracting features from audio: {e}")
        return None


def extract_features(audio_file, target_time_dim=259):
    """Extract multi-resolution mel spectrograms from audio file"""
    try:
        audio, _ = librosa.load(audio_file, sr=CONFIG["sample_rate"], mono=True)
        features = extract_features_from_audio(audio, target_time_dim)
        return features, audio
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None, None


def predict_instruments(model, features):
    """Predict instruments from features"""
    try:
        # Add batch dimension
        X = [np.expand_dims(f, axis=0) for f in features]
        predictions = model.predict(X, verbose=0)[0]
        return predictions
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None


def sliding_window_predict(model, audio_file, window_size=1.0, hop_size=0.5):
    """Predict instruments over time using sliding window"""
    try:
        audio, sr = librosa.load(audio_file, sr=CONFIG["sample_rate"])
        duration = librosa.get_duration(y=audio, sr=sr)
        times = np.arange(0, duration - window_size, hop_size)
        all_preds = []

        progress_bar = st.progress(0)
        for idx, t in enumerate(times):
            start = int(t * sr)
            end = int((t + window_size) * sr)
            segment = audio[start:end]

            if len(segment) < int(window_size * sr):
                pad = np.zeros(int(window_size * sr) - len(segment))
                segment = np.concatenate([segment, pad])

            features = extract_features_from_audio(segment)
            if features:
                X = [np.expand_dims(f, axis=0) for f in features]
                pred = model.predict(X, verbose=0)
                all_preds.append(pred[0])

            progress_bar.progress((idx + 1) / len(times))

        progress_bar.empty()
        return np.array(all_preds), times
    except Exception as e:
        st.error(f"Error in sliding window prediction: {e}")
        return None, None


# Header
st.markdown(
    """
    <div class="main-header">
        VANIDYA AI
    </div>
    <div class="sub-header">
        Next-Gen Instrument Recognition Engine
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem;">
            <h1 style="font-size: 3rem; margin: 0; filter: drop-shadow(0 0 10px rgba(56, 189, 248, 0.5));">🎻</h1>
            <h2 style="color: #f8fafc; font-size: 1.5rem; margin: 0.5rem 0; font-weight: 800;">VANIDYA</h2>
            <p style="color: #38bdf8; font-size: 0.9rem; letter-spacing: 2px; text-transform: uppercase;">Neural Edition</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    with st.container():
        st.markdown("### 🧠 Neural Core")

        # Model Selection
        model_files = [f for f in os.listdir() if f.endswith(".keras")]
        # Prioritize keeping the default ones at the top if they exist
        priority = ["best_model.keras", "instrument_classifier_v2.keras"]
        model_files.sort(key=lambda x: priority.index(x) if x in priority else 999)

        selected_model_name = st.selectbox("Select Model", model_files)

        if selected_model_name:
            st.info(f"Active: {selected_model_name}")

        st.markdown("### 🎛️ System Status")
        st.success("✔ TPU/GPU Optimized")

    st.markdown("---")

    st.markdown("### 🎯 Target Classes")
    st.markdown(
        """
        <div style="display: flex; flex-wrap: wrap; gap: 5px;">
        """,
        unsafe_allow_html=True,
    )
    for inst in CONFIG["instruments"]:
        st.markdown(
            f"""<span style="background: rgba(255,255,255,0.05); padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; color: #94a3b8;">{inst}</span>""",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("v2.5.0 | Build 2025.12.31")

# Main content
tab1, tab2, tab3 = st.tabs(["💾 UPLOAD & SCAN", "📊 NEURAL ANALYSIS", "� DATA EXPORT"])

with tab1:
    st.markdown(
        """
        <div style="text-align: center; padding: 3rem 0 2rem 0;">
            <h2 style="color: #e2e8f0; margin-bottom: 0.5rem;">Initialize Audio Sequence</h2>
            <p style="color: #94a3b8; font-size: 1.1rem;">Upload WAV/MP3 for Spectral Decomposition</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Drop Audio File Here",
        type=["wav", "mp3", "ogg", "flac"],
        help="Compatible with 44.1kHz / 16-bit PCM",
    )

    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### 🔊 Monitor")
            st.audio(uploaded_file, format="audio/wav")

        with col2:
            audio, sr = librosa.load(tmp_path, sr=CONFIG["sample_rate"])
            duration = librosa.get_duration(y=audio, sr=sr)

            # Metric row
            m1, m2, m3 = st.columns(3)
            m1.metric("Duration", f"{duration:.2f}s")
            m2.metric("Sample Rate", f"{sr}Hz")
            m3.metric("Tensor Size", f"{audio.shape[0]}")

        # Plotly Waveform
        st.markdown("### 🌊 Signal Visualization")

        # Downsample for performance if too long
        if len(audio) > 100000:
            audio_display = audio[:: int(len(audio) / 100000)]
        else:
            audio_display = audio

        fig_wave = go.Figure()
        fig_wave.add_trace(
            go.Scatter(
                y=audio_display,
                mode="lines",
                name="Amplitude",
                line=dict(color="#22d3ee", width=1),
                fill="tozeroy",
                fillcolor="rgba(34, 211, 238, 0.1)",
            )
        )
        fig_wave.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            hovermode="x",
        )
        st.plotly_chart(fig_wave, use_container_width=True)

        # Plotly Spectrogram
        st.markdown("### 🎨 Spectral Density (Mel)")

        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig_spec = px.imshow(
            mel_db,
            origin="lower",
            aspect="auto",
            color_continuous_scale="Magma",
            labels=dict(x="Time Frame", y="Frequency Band", color="dB"),
        )
        fig_spec.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            coloraxis_colorbar=dict(title="Amplitude (dB)"),
        )
        st.plotly_chart(fig_spec, use_container_width=True)

        # Analysis button
        st.markdown("---")
        if st.button(
            "� INITIATE NEURAL ANALYSIS", type="primary", use_container_width=True
        ):
            with st.spinner("Processing Tensors..."):
                if selected_model_name:
                    model, use_spectral_contrast = load_model(selected_model_name)
                    # Update global config with the setting from the loaded model
                    CONFIG["use_spectral_contrast"] = use_spectral_contrast
                else:
                    st.error("No model selected or found.")
                    model = None

            if model is not None:
                progress_text = "Running Inference Modules..."
                my_bar = st.progress(0, text=progress_text)

                with st.spinner("Analyzing audio..."):
                    # Quick prediction
                    my_bar.progress(30, text="Extracting Spectral Features...")
                    features, _ = extract_features(tmp_path)

                    if features:
                        my_bar.progress(60, text="Forward Propagation...")
                        predictions = predict_instruments(model, features)

                        if predictions is not None:
                            # Store results
                            st.session_state["predictions"] = predictions
                            st.session_state["audio_path"] = tmp_path
                            st.session_state["filename"] = uploaded_file.name
                            st.session_state["duration"] = duration

                            # Timeline analysis
                            my_bar.progress(80, text="Temporal Localization...")
                            preds_timeline, times_timeline = sliding_window_predict(
                                model, tmp_path
                            )
                            if preds_timeline is not None:
                                st.session_state["timeline_preds"] = preds_timeline
                                st.session_state["timeline_times"] = times_timeline

                            my_bar.progress(100, text="Analysis Complete!")
                            st.success("✅ Sequence Decoded Successfully")
                            st.toast(
                                "Analysis Complete! Switching to Results Tab...",
                                icon="🎉",
                            )


with tab2:
    st.markdown(
        """
        <div style="text-align: center; padding: 1.5rem 0;">
            <h2 style="color: #e2e8f0; margin-bottom: 0.5rem;">📊 Neural Decoding Results</h2>
            <p style="color: #94a3b8; font-size: 1.1rem;">Confidence Scores & Temporal Presence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "predictions" in st.session_state:
        predictions = st.session_state["predictions"]

        # Use fixed threshold from CONFIG
        threshold = CONFIG["detection_threshold"]

        st.markdown("---")
        st.markdown("### 🎯 Classification Matrix")

        # Create columns for instrument cards
        cols = st.columns(3)
        for idx, inst in enumerate(CONFIG["instruments"]):
            with cols[idx % 3]:
                confidence = predictions[idx]
                is_detected = confidence > threshold
                is_marginal = 0.30 < confidence <= threshold

                if is_detected:
                    card_class = "detected"
                    status = "ACTIVE"
                    emoji = "🔊"
                    text_color = "#10b981"
                elif is_marginal:
                    card_class = "marginal"
                    status = "POSSIBLE"
                    emoji = "❔"
                    text_color = "#f59e0b"
                else:
                    card_class = "not-detected"
                    status = "SILENT"
                    emoji = "🔇"
                    text_color = "#64748b"

                # Format instrument name for display (capitalize and replace underscores)
                display_name = inst.replace("_", " ").title()

                st.markdown(
                    f"""
                <div class="instrument-card {card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <h3 style="margin: 0; font-size: 1.1rem; color: #e2e8f0;">{display_name}</h3>
                        <span style="font-size: 1.5rem;">{emoji}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                        <span style="color: {text_color}; font-weight: 800; font-size: 0.9rem; letter-spacing: 1px;">{status}</span>
                        <span style="font-size: 1.8rem; font-weight: 900; color: #f8fafc;">{confidence:.1%}</span>
                    </div>
                    <div style="width: 100%; height: 4px; background: rgba(255,255,255,0.1); margin-top: 10px; border-radius: 2px;">
                        <div style="width: {confidence*100}%; height: 100%; background: {text_color}; border-radius: 2px; transition: width 0.5s;"></div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Summary statistics
        st.markdown("---")
        st.markdown("### 📈 Inference Stats")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Active Classes",
                sum(1 for p in predictions if p > threshold),
                delta="Above Threshold",
            )
        with col2:
            st.metric(
                "Marginal Classes",
                sum(1 for p in predictions if 0.30 < p <= threshold),
                delta_color="off",
            )
        with col3:
            st.metric(
                "Mean Confidence",
                f"{np.mean(predictions):.1%}",
            )
        with col4:
            st.metric(
                "Peak Confidence",
                f"{np.max(predictions):.1%}",
            )

        # Confidence bar chart
        st.markdown("---")
        st.markdown("### 📊 Distribution Analysis")

        # Validate predictions length matches instruments
        if len(predictions) != len(CONFIG["instruments"]):
            st.error(
                f"⚠️ Model output mismatch: Expected {len(CONFIG['instruments'])} predictions, got {len(predictions)}"
            )
            st.info(
                f"Model predictions shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}"
            )
            st.info(f"Number of instruments in config: {len(CONFIG['instruments'])}")
        else:
            # Prepare data for Plotly
            colors = [
                "#10b981" if p > threshold else "#f59e0b" if p > 0.3 else "#ef4444"
                for p in predictions
            ]

            df_chart = pd.DataFrame(
                {
                    "Instrument": [
                        inst.replace("_", " ").title() for inst in CONFIG["instruments"]
                    ],
                    "Confidence": predictions,
                    "Color": colors,
                }
            )

            df_chart = df_chart.sort_values("Confidence", ascending=True)

            fig_bar = px.bar(
                df_chart,
                x="Confidence",
                y="Instrument",
                orientation="h",
                text=[f"{c:.1%}" for c in df_chart["Confidence"]],
                color="Instrument",  # Dummy for color update
            )

            # Custom color update
            fig_bar.update_traces(
                marker_color=df_chart["Color"],
                textposition="outside",
                textfont_color="white",
            )

            fig_bar.update_layout(
                height=500,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                xaxis=dict(
                    showgrid=True, gridcolor="rgba(255,255,255,0.05)", range=[0, 1.1]
                ),
                yaxis=dict(showgrid=False),
                showlegend=False,
            )

            # Add threshold line
            fig_bar.add_vline(
                x=threshold,
                line_width=2,
                line_dash="dash",
                line_color="#38bdf8",
                annotation_text=f"Threshold {threshold:.0%}",
                annotation_position="top right",
            )

            st.plotly_chart(fig_bar, use_container_width=True)

        # Timeline analysis
        if "timeline_preds" in st.session_state:
            st.markdown("---")
            st.markdown("### ⏱️ Temporal Localization")
            st.markdown(
                '<p style="color: #94a3b8; font-size: 1rem; margin-bottom: 1rem;">Dynamic instrument activity over audio duration</p>',
                unsafe_allow_html=True,
            )

            preds_timeline = st.session_state["timeline_preds"]
            times_timeline = st.session_state["timeline_times"]

            # Create interactive line chart
            fig_time = go.Figure()

            # Add trace for each instrument that has at least some significant presence
            max_conf_per_inst = np.max(preds_timeline, axis=0)

            for i, inst in enumerate(CONFIG["instruments"]):
                # Only show instruments that cross the marginal threshold at least once to reduce clutter
                if max_conf_per_inst[i] > 0.2:
                    display_name = inst.replace("_", " ").title()
                    fig_time.add_trace(
                        go.Scatter(
                            x=times_timeline,
                            y=preds_timeline[:, i],
                            mode="lines",
                            name=display_name,
                            line=dict(width=2),
                            hovertemplate="<b>%{text}</b><br>Time: %{x:.2f}s<br>Conf: %{y:.1%}<extra></extra>",
                            text=[display_name] * len(times_timeline),
                        )
                    )

            fig_time.update_layout(
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                xaxis=dict(
                    title="Time (seconds)",
                    showgrid=False,
                    gridcolor="rgba(255,255,255,0.05)",
                ),
                yaxis=dict(
                    title="Probability",
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.05)",
                ),
                hovermode="x unified",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )

            fig_time.add_hline(
                y=threshold, line_width=1, line_dash="dash", line_color="#38bdf8"
            )

            st.plotly_chart(fig_time, use_container_width=True)

    else:
        st.markdown(
            """
            <div style="text-align: center; padding: 4rem 2rem; 
            background: rgba(30, 41, 59, 0.5);
            border-radius: 20px; margin: 2rem 0; border: 1px dashed #334155;">
                <h2 style="color: #64748b; font-size: 2rem; margin-bottom: 1rem;">
                    Waiting for Input
                </h2>
                <p style="color: #94a3b8; font-size: 1.2rem; margin-bottom: 1rem;">
                    Upload an audio file in the <b>"UPLOAD & SCAN"</b> tab to activate the neural engine.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab3:
    st.markdown(
        """
        <div style="text-align: center; padding: 1.5rem 0;">
            <h2 style="color: #e2e8f0; margin-bottom: 0.5rem;">📥 Export Data</h2>
            <p style="color: #94a3b8; font-size: 1.1rem;">Download Analysis Artifacts</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "predictions" in st.session_state:
        predictions = st.session_state["predictions"]
        filename = st.session_state["filename"]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                <div style="background: rgba(30, 41, 59, 0.7); 
                padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem; border: 1px solid rgba(255,255,255,0.05);">
                    <h3 style="margin: 0 0 0.5rem 0; color: #38bdf8;">📄 JSON Structure</h3>
                    <p style="margin: 0; opacity: 0.8; font-size: 0.9rem; color: #94a3b8;">
                        Raw data dump including temporal sequences and probability vectors.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Create JSON report
            report = {
                "filename": filename,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": st.session_state.get("duration", 0),
                "detected_instruments": {
                    inst: {
                        "confidence": float(predictions[idx]),
                        "detected": bool(predictions[idx] > 0.5),
                    }
                    for idx, inst in enumerate(CONFIG["instruments"])
                },
                "summary": {
                    "total_detected": int(sum(1 for p in predictions if p > 0.5)),
                    "average_confidence": float(np.mean(predictions)),
                    "max_confidence": float(np.max(predictions)),
                },
            }

            if "timeline_preds" in st.session_state:
                report["timeline"] = [
                    {
                        "time": float(t),
                        **{
                            inst: float(st.session_state["timeline_preds"][j, i])
                            for i, inst in enumerate(CONFIG["instruments"])
                        },
                    }
                    for j, t in enumerate(st.session_state["timeline_times"])
                ]

            json_str = json.dumps(report, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"{filename}_analysis.json",
                mime="application/json",
                use_container_width=True,
            )

        with col2:
            st.markdown(
                """
                <div style="background: rgba(30, 41, 59, 0.7); 
                padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem; border: 1px solid rgba(255,255,255,0.05);">
                    <h3 style="margin: 0 0 0.5rem 0; color: #e879f9;">📝 Summary Report</h3>
                    <p style="margin: 0; opacity: 0.8; font-size: 0.9rem; color: #94a3b8;">
                        Human-readable text abstract of the detected instrument classes.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Create text report
            text_report = f"""
VANIDYA AI - NEURAL ANALYSIS REPORT
{'='*50}

Target File: {filename}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Length: {st.session_state.get('duration', 0):.2f}s

CLASSIFICATION PROBABILITIES:
{'='*50}
"""
            for idx, inst in enumerate(CONFIG["instruments"]):
                display_name = inst.replace("_", " ").title()
                if predictions[idx] > CONFIG["detection_threshold"]:
                    status = "[ACTIVE]"
                elif predictions[idx] > 0.3:
                    status = "[POSSIBLE]"
                else:
                    status = "[SILENT]"
                text_report += f"\n{display_name:20s} : {status:12s} | {predictions[idx]:.2%} Conf."

            text_report += f"""

METRICS:
{'='*50}
Threshold Setting: {CONFIG["detection_threshold"]:.0%}
Active Classes: {sum(1 for p in predictions if p > CONFIG["detection_threshold"])}
Mean Confidence: {np.mean(predictions):.2%}

Generated by VANIDYA AI v2.5
"""

            st.download_button(
                label="📥 Download Text Report",
                data=text_report,
                file_name=f"{filename}_analysis.txt",
                mime="text/plain",
                use_container_width=True,
            )

        # Display preview
        st.markdown(
            '<h3 style="color: #e2e8f0; margin: 2rem 0 1rem 0;">👀 Data Preview</h3>',
            unsafe_allow_html=True,
        )
        with st.expander("Expand JSON Data Structure", expanded=False):
            st.json(report)

        with st.expander("Expand Text Report", expanded=False):
            st.code(text_report)

    else:
        st.markdown(
            """
            <div style="text-align: center; padding: 4rem 2rem; 
            background: rgba(30, 41, 59, 0.5);
            border-radius: 20px; margin: 2rem 0; border: 1px dashed #334155;">
                <h2 style="color: #64748b; font-size: 2rem; margin-bottom: 1rem;">
                    No Data Available
                </h2>
                <p style="color: #94a3b8; font-size: 1.2rem; margin-bottom: 1rem;">
                    Processing is required before export.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; padding: 3rem 0; opacity: 0.6;'>
    <p style="margin: 0; font-size: 0.9rem; color: #64748b;">
        VANIDYA AI • Powered by TensorFlow & Streamlit • 2025
    </p>
</div>
""",
    unsafe_allow_html=True,
)
