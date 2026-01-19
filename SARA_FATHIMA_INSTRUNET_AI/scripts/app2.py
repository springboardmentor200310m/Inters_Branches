# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from inference_multi import predict_audio
import json
from fpdf import FPDF
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="InstruNetAI",
    page_icon="🎵",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
    font-family: 'Sans-serif';
}

/* Glass container */
.glass-container {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 20px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
    color: white;
}

/* Hero header animation */
@keyframes slideIn {
    from {opacity:0; transform: translateY(-30px);}
    to {opacity:1; transform: translateY(0);}
}
.header-title {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    color: #f9f9f9;
    animation: slideIn 1s ease-out;
}
.header-subtitle {
    text-align: center;
    font-size: 1.2rem;
    color: #d1d1d1;
    margin-bottom: 40px;
    animation: slideIn 1.5s ease-out;
}

/* Predict button */
.stButton>button {
    background-color: #ff9800 !important;
    color: black !important;
    font-weight: 700;
    font-size: 16px;
    padding: 0.6em 1.2em;
    border-radius: 12px;
    border: none;
}
.stButton>button:hover {
    background-color: #ffc107 !important;
    color: black !important;
}

/* Result Banner */
.result-banner {
    background: #0f8a3c;
    color: white;
    font-size: 2rem;
    text-align: center;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 0 20px #0f8a3c;
    margin-bottom: 30px;
}

/* Export buttons */
.export-btn {
    background-color: rgba(255,255,255,0.1) !important;
    color: white !important;
    font-weight: 600;
    font-size: 16px;
    padding: 0.6em;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.3);
    width: 100%;
    margin-bottom: 10px;
}
.export-btn:hover {
    background-color: rgba(255,255,255,0.2) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO HEADER ----------------
st.markdown('<div class="header-title">🎵 InstruNetAI</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">AI-powered Musical Instrument Recognition from Audio</div>', unsafe_allow_html=True)

# ---------------- TOP SECTION ----------------
left_col, right_col = st.columns(2)

# Upload Column
with left_col:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.subheader("🎧 Upload Audio")
    uploaded_file = st.file_uploader("Supported formats: WAV, MP3", type=["wav", "mp3"])
    st.markdown('</div>', unsafe_allow_html=True)

# How it Works Column
with right_col:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.subheader("📝 How it works")
    st.markdown("""
    - 🎶 Audio is converted into a spectrogram  
    - 🤖 A **CNN-based multi-label model** analyzes the sound  
    - 🎷 Multiple instruments can be detected simultaneously
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if uploaded_file:
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Predict Instrument(s)"):
        with st.spinner("Running inference..."):
            try:
                predicted, probs = predict_audio(temp_path)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                predicted, probs = [], {}

        # Result Banner
        main_instrument = predicted[0] if predicted else "None Detected"
        st.markdown(f'<div class="result-banner">Detected: {main_instrument}</div>', unsafe_allow_html=True)

        # Analysis Section
        chart_col, export_col = st.columns([2, 1])

        # Confidence Chart
        with chart_col:
            st.subheader("📊 Confidence Scores")
            if probs:
                instruments = list(probs.keys())
                confidences = [float(v) for v in probs.values()]
            else:
                instruments = ["Piano", "Flute", "Saxophone", "Clarinet"]
                confidences = [0.85, 0.1, 0.1, 0.05]

            fig = go.Figure(go.Bar(
                x=instruments,
                y=confidences,
                marker_color='mediumpurple',
                marker_line_color='rgb(255,255,255)',
                marker_line_width=1.5
            ))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                yaxis=dict(title='Probability'),
                xaxis=dict(title='Instrument')
            )
            st.plotly_chart(fig, use_container_width=True)

        # Export Options
        with export_col:
            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
            st.subheader("💾 Export Options")

            # JSON Download
            json_data = json.dumps({"predicted": predicted, "probabilities": probs}, indent=4)
            json_bytes = io.BytesIO(json_data.encode('utf-8'))
            st.download_button(
                label="Export as JSON",
                data=json_bytes,
                file_name="InstruNetAI_prediction.json",
                mime="application/json"
            )

            # PDF Download
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, txt="InstruNetAI Prediction Report", ln=True, align='C')
            pdf.ln(10)
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 10, txt=f"Predicted Instruments: {', '.join(predicted)}")
            pdf.ln(5)
            pdf.multi_cell(0, 10, txt="Confidence Scores:\n" + "\n".join([f"{k}: {v:.2f}" for k,v in probs.items()]))
            pdf_buffer = io.BytesIO()
            pdf.output(pdf_buffer)
            pdf_buffer.seek(0)
            st.download_button(
                label="Export as PDF",
                data=pdf_buffer,
                file_name="InstruNetAI_prediction.pdf",
                mime="application/pdf"
            )

            st.markdown('</div>', unsafe_allow_html=True)
