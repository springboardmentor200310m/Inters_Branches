# app1.py
import os
import tempfile
import numpy as np
import torch
import soundfile as sf
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import onnxruntime as ort
import streamlit as st
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt

# ==== Constants ====
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
SPEC_LEN = 128

INSTR_FAMILY_MAP = {
    0: 'Accordion', 1: 'Acoustic_Guitar', 2: 'Banjo', 3: 'Bass_Guitar', 4: 'Clarinet',
    5: 'Cymbals', 6: 'Dobro', 7: 'Drum_set', 8: 'Electro_Guitar', 9: 'Floor_Tom',
    10: 'Harmonica', 11: 'Harmonium', 12: 'Hi_Hats', 13: 'Horn', 14: 'Keyboard',
    15: 'Mandolin', 16: 'Organ', 17: 'Piano', 18: 'Saxophone', 19: 'Shakers',
    20: 'Tambourine', 21: 'Trombone', 22: 'Trumpet', 23: 'Ukulele', 24: 'Violin',
    25: 'Cowbell', 26: 'Flute', 27: 'Vibraphone'
}

# ==== ONNX Wrapper ====
class NsynthOnnxWrapper:
    def __init__(self, onnx_model_path: str):
        self.session = ort.InferenceSession(
            onnx_model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.mel = MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        )
        self.to_db = AmplitudeToDB(stype="power")

    def load_audio(self, wav_path):
        waveform, sr = sf.read(wav_path)
        if waveform.ndim == 2:
            waveform = np.mean(waveform, axis=1)
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

        return waveform

    def preprocess_wav(self, wav_path: str):
        waveform = self.load_audio(wav_path)

        spec = self.to_db(self.mel(waveform)).squeeze(0)

        if spec.size(1) < SPEC_LEN:
            spec = torch.nn.functional.pad(spec, (0, SPEC_LEN - spec.size(1)))
        else:
            spec = spec[:, :SPEC_LEN]

        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        model_input = spec.unsqueeze(0).unsqueeze(0)

        return model_input.numpy().astype("float32"), spec

    def predict(self, wav_path: str):
        x, spec = self.preprocess_wav(wav_path)
        logits = self.session.run([self.output_name], {self.input_name: x})[0]
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[0]
        return probs, spec

# ==== Spectrogram Image Generator ====
def save_spectrogram_image(spec, file_path):
    plt.figure(figsize=(6, 4))
    plt.imshow(spec.numpy(), origin="lower", aspect="auto", cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")
    plt.tight_layout()
    plt.savefig(file_path, dpi=200)
    plt.close()

# ==== PDF Generator ====
def generate_pdf(results: dict, file_path: str):
    c = canvas.Canvas(file_path, pagesize=LETTER)
    width, height = LETTER

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Instrument Classification Report")

    c.setFont("Helvetica", 12)
    y = height - 100

    c.drawString(50, y, f"Predicted Instrument: {results['predicted_instrument']}")
    y -= 30

    c.drawString(50, y, "Top Predictions:")
    y -= 20

    for item in results["top_predictions"]:
        c.drawString(
            70, y,
            f"{item['instrument']} — Probability: {item['probability']:.3f}"
        )
        y -= 18

    c.save()

# ==== Streamlit UI ====
st.set_page_config(page_title="Instrument Identifier", layout="centered")
st.title("🎵 Musical Instrument Identifier")
st.write("Upload a WAV file to get JSON, PDF, and Spectrogram image.")

MODEL_PATH = "nsynth_instrument_family_cnn.onnx"

@st.cache_resource
def load_model():
    return NsynthOnnxWrapper(MODEL_PATH)

wrapper = load_model()

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        wav_path = tmp.name

    st.audio(uploaded_file)

    if st.button("Predict Instrument"):
        probs, spec = wrapper.predict(wav_path)

        top_k = 5
        top_indices = np.argsort(probs)[::-1][:top_k]

        results = {
            "predicted_class_index": int(top_indices[0]),
            "predicted_instrument": INSTR_FAMILY_MAP[int(top_indices[0])],
            "top_predictions": [
                {
                    "class_index": int(i),
                    "instrument": INSTR_FAMILY_MAP[int(i)],
                    "probability": float(probs[i])
                }
                for i in top_indices
            ]
        }

        # JSON Output
        st.subheader("📊 Prediction Output (JSON)")
        st.json(results)

        # Spectrogram Image
        img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        save_spectrogram_image(spec, img_path)

        st.subheader("🎼 Mel Spectrogram")
        st.image(img_path, caption="Generated Mel Spectrogram", use_container_width=True)

        with open(img_path, "rb") as f:
            st.download_button(
                "🖼 Download Spectrogram Image",
                f,
                file_name="mel_spectrogram.png",
                mime="image/png"
            )

        # PDF
        pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        generate_pdf(results, pdf_path)

        with open(pdf_path, "rb") as f:
            st.download_button(
                "📄 Download PDF Report",
                f,
                file_name="instrument_prediction.pdf",
                mime="application/pdf"
            )
else:
    st.info("Please upload a WAV file to begin.")
#streamlit run "C:/Users/siri reddy/OneDrive/Documents/Desktop/Label/app1_image.py"