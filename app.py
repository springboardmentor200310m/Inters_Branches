import streamlit as st
import numpy as np
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
