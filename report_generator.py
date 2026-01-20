from fpdf import FPDF
import os
import json
import matplotlib.pyplot as plt

class ReportGenerator:
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_pdf(self, track_name, results, classes, waveform_img, spectrogram_img, timeline_img, bar_img):
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font("Arial", 'B', 24)
        pdf.set_text_color(40, 40, 120)
        pdf.cell(0, 20, "InstruNet AI Analysis Report", ln=True, align='C')
        pdf.ln(5)
        
        # Track Info
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, f"Track Name: {track_name}", ln=True)
        pdf.ln(5)
        
        # Detected Instruments
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "1. Detected Instruments Summary", ln=True)
        pdf.set_font("Arial", '', 12)
        
        # Filter present instruments
        detected = [c for c in classes if results[c]['present']]
        if not detected:
            pdf.cell(0, 8, "- No instruments detected above threshold.", ln=True)
        for instr in detected:
            conf = results[instr]['confidence'] * 100
            pdf.cell(0, 8, f"- {instr}: Present ({conf:.1f}%)", ln=True)
        pdf.ln(10)

        # Confidence Chart
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "2. Confidence Distribution", ln=True)
        pdf.image(bar_img, x=10, y=pdf.get_y(), w=190)
        pdf.ln(100) # Jump down after image
        
        # Visuals (Waveform/Spectrogram)
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "3. Audio Visualizations", ln=True)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Waveform", ln=True)
        pdf.image(waveform_img, x=10, y=pdf.get_y(), w=190)
        pdf.ln(60)
        
        pdf.cell(0, 8, "Mel-Spectrogram", ln=True)
        pdf.image(spectrogram_img, x=10, y=pdf.get_y(), w=190)
        pdf.ln(70)
        
        # Timeline
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "4. Instrument Activity Timeline", ln=True)
        pdf.image(timeline_img, x=10, y=pdf.get_y(), w=190)
        
        # Final Export
        filename = f"{os.path.splitext(track_name)[0]}_report.pdf"
        output_path = os.path.join(self.output_dir, filename)
        pdf.output(output_path)
        return output_path

    def generate_json(self, track_name, results):
        report = {
            "track_name": track_name,
            "detected_instruments": [c for c in results if results[c]['present']],
            "confidence_scores": {c: results[c]['confidence'] for c in results},
            "instrument_timeline": {c: results[c]['timeline'] for c in results}
        }
        filename = f"{os.path.splitext(track_name)[0]}_instruments.json"
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        return output_path
