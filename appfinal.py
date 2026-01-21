import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import pickle
import tempfile
import os
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Instrument Recognition System",
    page_icon="🎵",
    layout="centered"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#667eea,#764ba2);
}
.main {
    background: rgba(255,255,255,0.95);
    border-radius: 16px;
    padding: 25px;
}
.title {
    text-align:center;
    font-size:36px;
    font-weight:bold;
    color:#4b0082;
}
.card {
    background:#ffffff;
    padding:14px;
    border-radius:10px;
    margin-bottom:10px;
    box-shadow:0 4px 10px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

# ---------------- CONFIG ----------------
SAMPLE_RATE = 22050
SINGLE_MODEL_PATH = "instrument_cnn_model.h5"
MULTI_MODEL_PATH = "multilabel_model.h5"
ENCODER_PATH = "label_encoder.pkl"

# ---------------- LOAD LABEL ENCODER ----------------
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# ---------------- AUDIO → SPECTROGRAM ----------------
def audio_to_spec(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Save spectrogram image
    fig, ax = plt.subplots(figsize=(6, 4))
    img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    plt.title("Mel-Spectrogram (Intensity shown by color)")
    spec_path = audio_path.replace(".wav", "_spec.png")
    plt.savefig(spec_path, bbox_inches="tight")
    plt.close()

    # Normalize for model
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
    S_db = tf.image.resize(S_db[..., np.newaxis], (128, 256))
    return tf.expand_dims(S_db, axis=0), spec_path

# ---------------- INTENSITY VS TIME ----------------
def plot_intensity_time(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(times, rms, color="purple", linewidth=2)
    ax.set_title("Audio Intensity (Energy) vs Time")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Intensity (RMS Energy)")
    ax.grid(alpha=0.3)
    return fig

# ---------------- PREDICTION ----------------
def predict_single(model, audio_path):
    spec, spec_img = audio_to_spec(audio_path)
    probs = model.predict(spec)[0]
    idx = np.argmax(probs)
    return [(label_encoder.classes_[idx], probs[idx] * 100)], spec_img

def predict_multi(model, audio_path, threshold):
    spec, spec_img = audio_to_spec(audio_path)
    probs = model.predict(spec)[0]
    results = [
        (label_encoder.classes_[i], probs[i] * 100)
        for i in range(len(probs)) if probs[i] >= threshold
    ]
    return results, spec_img

# ---------------- PDF REPORT ----------------
def generate_pdf(results, spec_img, model_type):
    pdf_path = "Final_Prediction_Report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=A4)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "Music Instrument Recognition Report")

    c.setFont("Helvetica", 11)
    c.drawString(50, 770, f"Model Type: {model_type}")
    c.drawString(50, 750, f"Generated On: {datetime.now()}")

    c.drawImage(spec_img, 50, 470, width=300, height=200)

    y = 440
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Predicted Instruments:")
    y -= 20

    c.setFont("Helvetica", 11)
    for inst, conf in results:
        c.drawString(60, y, f"- {inst} : {conf:.2f}%")
        y -= 15

    c.save()
    return pdf_path

# ---------------- UI ----------------
st.markdown('<div class="title">🎶 Instrument Recognition System</div>', unsafe_allow_html=True)

model_choice = st.radio(
    "Select Model Type",
    ["Single-Label Model", "Multi-Label Model"]
)

uploaded_file = st.file_uploader("Upload Audio File (.wav / .mp3)", type=["wav", "mp3"])

threshold = st.slider("Multi-label Confidence Threshold", 0.1, 0.9, 0.4, 0.05)

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("🔍 Predict"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            audio_path = tmp.name

        if model_choice == "Single-Label Model":
            model = tf.keras.models.load_model(SINGLE_MODEL_PATH)
            results, spec_img = predict_single(model, audio_path)
        else:
            model = tf.keras.models.load_model(MULTI_MODEL_PATH)
            results, spec_img = predict_multi(model, audio_path, threshold)

        st.subheader("🎼 Mel-Spectrogram")
        st.image(spec_img)

        st.subheader("🔊 Intensity (Energy) vs Time")
        st.pyplot(plot_intensity_time(audio_path))

        st.subheader("🎵 Prediction Results")
        for inst, conf in results:
            st.markdown(
                f"<div class='card'><b>{inst}</b><br>Confidence: {conf:.2f}%</div>",
                unsafe_allow_html=True
            )

        pdf_path = generate_pdf(results, spec_img, model_choice)

        with open(pdf_path, "rb") as f:
            st.download_button(
                "📄 Download Final Report",
                f,
                file_name="Instrument_Prediction_Report.pdf",
                mime="application/pdf"
            )

        os.remove(audio_path)
