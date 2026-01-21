import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tempfile
import os

from backend_predict import predict_audio, index_to_label

# -------------------------------
# STREAMLIT CONFIG
# -------------------------------
st.set_page_config(page_title="Musical Instrument Recognition", layout="wide")
st.title("🎵 Musical Instrument Recognition System")

st.markdown("""
Upload **one or more audio files (.wav)**  
The system will:
- Convert audio → spectrogram
- Predict instrument
- Show confidence & probabilities
- Show intensity over time
""")

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload Audio Files",
    type=["wav"],
    accept_multiple_files=True
)

# -------------------------------
# PROCESS FILES
# -------------------------------
if uploaded_files:
    for file in uploaded_files:
        st.divider()
        st.subheader(f"🎧 {file.name}")

        # Save temp file safely
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(file.read())
            temp_path = tmp.name

        try:
            label, confidence, probs, intensity = predict_audio(temp_path)

            # -------------------------------
            # RESULTS
            # -------------------------------
            st.success(f"🎼 Predicted Instrument: **{label}**")
            st.metric("Confidence", f"{confidence:.2f}")

            # -------------------------------
            # PROBABILITY BAR CHART
            # -------------------------------
            st.subheader("📊 Class Probabilities")

            labels = list(index_to_label.values())
            fig1, ax1 = plt.subplots()
            ax1.bar(labels, probs)
            ax1.set_ylabel("Probability")
            ax1.set_xticklabels(labels, rotation=45, ha="right")
            st.pyplot(fig1)

            # -------------------------------
            # INTENSITY OVER TIME
            # -------------------------------
            st.subheader("📈 Intensity Over Time")

            fig2, ax2 = plt.subplots()
            librosa_display = intensity.mean(axis=0)
            ax2.plot(librosa_display)
            ax2.set_xlabel("Time Frames")
            ax2.set_ylabel("Intensity")
            st.pyplot(fig2)

            # -------------------------------
            # SPECTROGRAM IMAGE
            # -------------------------------
            st.subheader("🖼 Spectrogram")
            fig3, ax3 = plt.subplots()
            ax3.imshow(intensity, aspect="auto", origin="lower", cmap="magma")
            ax3.set_title("Mel Spectrogram")
            st.pyplot(fig3)

        except Exception as e:
            st.error(f"❌ Error processing {file.name}: {e}")

        finally:
            os.remove(temp_path)
# ---------------- CONFIG ----------------
MODEL_PATH = "instrument_cnn_model.h5"
ENCODER_PATH = "label_encoder.pkl"
SAMPLE_RATE = 22050

st.set_page_config(
    page_title="Music Instrument Recognition",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

model, label_encoder = load_model_and_encoder()

# ---------------- AUDIO → SPECTROGRAM ----------------
def audio_to_melspectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )

    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)

    S_db = tf.image.resize(S_db[..., np.newaxis], (128, 256))
    S_db = tf.expand_dims(S_db, axis=0)

    return S_db, y, sr

# ---------------- UI ----------------
st.title("🎶 Music Instrument Recognition System")
st.markdown("Upload an audio file and detect the musical instrument using CNN")

uploaded_file = st.file_uploader(
    "Upload Audio File (.wav / .mp3)",
    type=["wav", "mp3"]
)

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(audio_path)

    spec, y, sr = audio_to_melspectrogram(audio_path)

    # Prediction
    predictions = model.predict(spec)[0]
    pred_index = np.argmax(predictions)
    pred_label = label_encoder.inverse_transform([pred_index])[0]
    confidence = predictions[pred_index]

    # Display result
    st.success(f"🎵 **Predicted Instrument:** {pred_label}")
    st.info(f"🔍 **Confidence:** {confidence:.2f}")

    # Confidence bar chart
    st.subheader("Prediction Confidence")
    fig, ax = plt.subplots()
    ax.barh(label_encoder.classes_, predictions)
    ax.set_xlabel("Confidence")
    ax.set_xlim(0, 1)
    st.pyplot(fig)

    # Spectrogram display

    # ---------------- Spectrogram Display ----------------
    st.subheader("Mel-Spectrogram")

    import librosa.display
    import matplotlib.pyplot as plt

    # Create figure and axis
    fig2, ax2 = plt.subplots(figsize=(8, 4))

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )

    S_db = librosa.power_to_db(S, ref=np.max)

    # Plot spectrogram
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        ax=ax2
    )

    # Add colorbar correctly
    fig2.colorbar(img, ax=ax2, format="%+2.0f dB")

    ax2.set_title("Mel-Spectrogram")

    st.pyplot(fig2)


    os.remove(audio_path)
