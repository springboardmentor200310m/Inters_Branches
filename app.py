import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from fpdf import FPDF

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Instrument Recognition",
    page_icon="🎵",
    layout="centered"
)

# -----------------------------
# YOUR CSS (UNCHANGED)
# -----------------------------
st.markdown(
    """
    <style>
    html, body, [data-testid="stApp"] {
        background-color: #0b0f19;
        color: #ffffff;
    }

    section[data-testid="stSidebar"], .block-container {
        background-color: #0b0f19;
    }

    .navbar {
        background: linear-gradient(90deg, #312e81, #3730a3);
        padding: 18px;
        border-radius: 14px;
        text-align: center;
        color: #ffffff;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 25px;
        box-shadow: 0px 6px 20px rgba(0,0,0,0.6);
    }

    .stFileUploader {
        background: linear-gradient(90deg, #312e81, #3730a3);
        border-radius: 12px;
        padding: 10px;
    }

    .stFileUploader label {
        color: #ffffff !important;
        font-weight: 600;
    }

    .section {
        font-size: 22px;
        font-weight: 600;
        margin-top: 25px;
        color: #e8eaf6;
    }

    .card {
        background-color: #151a2c;
        padding: 16px 20px;
        border-radius: 14px;
        margin-bottom: 14px;
        border-left: 5px solid #593AAB;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.5);
    }

    .instrument {
        font-size: 18px;
        font-weight: 600;
        color: #ffffff;
    }

    .confidence {
        font-size: 14px;
        color: #9fa8da;
    }

    button {
        background-color: #2C67C5 !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "finetuned_cnn_model.h5"
LABEL_MAP_PATH = "label_mapping.json"
TARGET_SR = 22050
IMG_SIZE = (224, 224)

# -----------------------------
# LOAD MODEL & LABELS
# -----------------------------
@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
    return model, label_map

model, label_map = load_resources()

# CRITICAL: correct order
classes = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

# -----------------------------
# AUDIO FUNCTIONS
# -----------------------------
def load_audio(path):
    audio, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio, sr

def preprocess_audio(audio, sr):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    mel_img = tf.image.resize(mel_norm[..., np.newaxis], IMG_SIZE)
    mel_img = tf.repeat(mel_img, 3, axis=-1)
    mel_img = np.expand_dims(mel_img.numpy(), axis=0)

    return mel_img, mel_db

def compute_rms(audio):
    return librosa.feature.rms(y=audio)[0]

# -----------------------------
# PREDICTION (FIXED)
# -----------------------------
def predict_all_nonzero(mel_input, min_confidence=0.5):
    probs = model.predict(mel_input, verbose=0)[0]

    results = []
    for i, p in enumerate(probs):
        conf = p * 100
        if conf >= min_confidence:  # FILTER ZERO CONFIDENCE
            results.append({
                "instrument": classes[i],
                "confidence": float(conf)
            })

    return sorted(results, key=lambda x: x["confidence"], reverse=True)

def compute_intensity(audio):
    """
    Computes RMS (Root Mean Square) energy of the audio signal.
    Used for audio intensity visualization.
    """
    rms = librosa.feature.rms(y=audio)[0]
    return rms

# -----------------------------
# PDF EXPORT
# -----------------------------
def export_pdf(audio_name, predictions, mel_db, audio, sr, output_path="track_report.pdf"):
    pdf = FPDF()
    pdf.add_page()

    # -------- Title --------
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Instrument Recognition Report", ln=True)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Audio File: {audio_name}", ln=True)
    pdf.ln(5)

    # -------- Predictions --------
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Predicted Instruments:", ln=True)

    pdf.set_font("Arial", size=11)
    valid_preds = []
    for p in predictions:
        if p["confidence"] > 0:
            valid_preds.append(p)
            pdf.cell(0, 8, f"{p['instrument']} : {p['confidence']:.2f}%", ln=True)

    # -------- Mel Spectrogram --------
    fig, ax = plt.subplots(figsize=(6, 4))
    img = librosa.display.specshow(
        mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram")

    mel_path = "mel_spec.png"
    plt.savefig(mel_path, bbox_inches="tight")
    plt.close(fig)

    pdf.add_page()
    pdf.image(mel_path, x=10, y=20, w=180)

    # -------- Audio Intensity (RMS × Confidence) --------
    rms = librosa.feature.rms(y=audio)[0]
    times = librosa.times_like(rms, sr=sr)

    fig, ax = plt.subplots(figsize=(7, 4))

    for p in valid_preds:
        weight = p["confidence"] / 100.0
        instrument_rms = rms * weight

        ax.plot(
            times,
            instrument_rms,
            label=f"{p['instrument']} ({p['confidence']:.1f}%)"
        )

    ax.set_title("Audio Intensity Across Predicted Instruments")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS Intensity")
    ax.legend()
    ax.grid(alpha=0.3)

    rms_path = "intensity_plot.png"
    plt.savefig(rms_path, bbox_inches="tight")
    plt.close(fig)

    pdf.add_page()
    pdf.image(rms_path, x=10, y=20, w=180)

    pdf.output(output_path)

    os.remove(mel_path)
    os.remove(rms_path)

    return output_path

# -----------------------------
# UI
# -----------------------------
st.markdown('<div class="navbar">🎵 Musical Instrument Recognition</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload WAV audio file", type=["wav"])

if uploaded_file:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio("temp.wav")

    audio, sr = load_audio("temp.wav")
    mel_input, mel_db = preprocess_audio(audio, sr)
    rms = compute_rms(audio)

    results = predict_all_nonzero(mel_input)

    st.markdown('<div class="section">🎯 Predictions</div>', unsafe_allow_html=True)

    for r in results:
        st.markdown(
            f"""
            <div class="card">
                <div class="instrument">{r['instrument']}</div>
                <div class="confidence">{r['confidence']:.2f}% confidence</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="section">📊 Mel Spectrogram</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    fig.colorbar(img, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown('<div class="section">📈 Audio Intensity</div>', unsafe_allow_html=True)
    rms = compute_intensity(audio)
    times = librosa.times_like(rms, sr=sr)

    fig, ax = plt.subplots(figsize=(7, 4))

    for r in results:
        if r["confidence"] <= 0:
            continue  # ignore 0 confidence instruments

        weight = r["confidence"] / 100.0  # scale 0–1
        instrument_rms = rms * weight     # 🔑 KEY LINE

        ax.plot(
            times,
            instrument_rms,
            label=f"{r['instrument']} ({r['confidence']:.1f}%)"
        )

    ax.set_title("Audio Intensity Across Predicted Instruments")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS Intensity")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)




    pdf_path = export_pdf(
    uploaded_file.name,
    results,
    mel_db,
    audio,
    sr,
    output_path="track_report.pdf"
)

    st.download_button("⬇ Download PDF Report", data=open(pdf_path, "rb"), file_name="track_report.pdf")

    os.remove("temp.wav")
