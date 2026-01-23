import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import io
from fpdf import FPDF
import tempfile

# --- 1. Page Configuration ---
st.set_page_config(page_title="InstruNet AI", layout="wide")
st.title("🎵 InstruNet AI")

# --- 2. Model Loading (Cached) ---
@st.cache_resource
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 11)
    )
    model.load_state_dict(torch.load("ResNet.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

class_names = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']

# --- 3. Optimized Audio Processing ---
def process_audio_pipeline(audio_file):
    y, sr = librosa.load(audio_file, sr=None, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Normalize for image conversion
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
    img = Image.fromarray((S_norm * 255).astype(np.uint8)).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0), S_db, sr, y

# --- 4. PDF Generation Helper ---
def generate_pdf_report(results, plot_buf):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(40, 10, "InstruNet AI: Track Analysis Report")
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(40, 10, "Detected Instruments:")
    pdf.ln(8)
    
    for instr, conf in results.items():
        pdf.cell(40, 10, f"- {instr.capitalize()}: {conf:.2%}")
        pdf.ln(6)
    
    # Save the plot buffer to a temp file to add to PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(plot_buf.getvalue())
        pdf.image(tmp.name, x=10, y=pdf.get_y() + 10, w=180)
    
    return pdf.output(dest='S').encode('latin-1')

# --- 5. Sidebar & Main UI ---
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Sensitivity Threshold", 0.0, 1.0, 0.5)
uploaded_file = st.sidebar.file_uploader("Upload Audio", type=["wav", "mp3"])

if uploaded_file:
    model, device = get_model()
    
    with st.spinner("Processing..."):
        input_tensor, S_db, sr, y = process_audio_pipeline(uploaded_file)
        
        # Inference (Multi-label)
        with torch.no_grad():
            outputs = model(input_tensor.to(device))
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
        
    col1, col2 = st.columns(2)
    
    with col1:
        # --- Corrected Spectrogram Block ---
        st.subheader("Spectrogram Analysis")
        fig, ax = plt.subplots(figsize=(10, 4))

        # Explicitly capture the 'mappable' object (img_plot)
        img_plot = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)

        # Attach the colorbar to that specific mappable and axis
        fig.colorbar(img_plot, ax=ax, format='%+2.0f dB') 

        st.pyplot(fig)

        # Save plot to buffer for your PDF export
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.audio(uploaded_file)

    with col2:
        st.subheader("Instrument Presence")
        detected = {}
        for i, name in enumerate(class_names):
            p = float(probs[i])
            if p >= threshold:
                st.write(f"**{name.capitalize()}**")
                st.progress(p)
                detected[name] = p
        
        if not detected:
            st.warning("No instruments detected above threshold.")

        # Export Actions
        st.divider()
        c1, c2 = st.columns(2)
        c1.download_button("Export JSON", data=json.dumps(detected), file_name="report.json")
        
        if detected:
            pdf_data = generate_pdf_report(detected, buf)
            c2.download_button("Export PDF Report", data=pdf_data, file_name="report.pdf", mime="application/pdf")
else:
    st.info("Upload an audio file to start.")

