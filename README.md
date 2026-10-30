# 👁️ IcterusLens AI — Jaundice (Scleral Icterus) Detection System

An AI-powered clinical decision-support and screening web dashboard designed to detect jaundice from sclera (eye) photography using computer vision and machine learning ensemble colorimetry.

---

## 🌟 Key Features

1. **Multi-Input Support**:
   - **File Upload**: Upload high-resolution eye or face photographs (JPG, PNG, WEBP).
   - **Live Webcam Capture**: Directly take a photo using your laptop or mobile camera.
   - **Pre-loaded Test Cases**: Instant 1-click test samples (Confirmed Healthy vs Confirmed Jaundice cases).

2. **Computer Vision & Scleral Colorimetry**:
   - Isolates scleral tissue and brightness regions.
   - Calculates the **Yellow Chromaticity Index**: `(R + G) - 2B`.
   - Multi-space color extraction across RGB, HSV, and YCbCr spectra.

3. **High-Accuracy ML Ensemble**:
   - Trained on the 328 annotated images from `Jaundice_Dataset.zip`.
   - Soft-voting ensemble combining **Random Forest** and **Extra Trees** classifiers.
   - **Benchmark Test Accuracy: 84.62%** with high sensitivity on positive jaundice cases.

4. **Clinical Decision Support**:
   - Probability & risk score gauge.
   - Sclera Region of Interest (ROI) visual segmentation overlay.
   - Patient symptom and risk factor checklist (dark urine, abdominal pain, pruritus, pale stool).
   - Clinical recommendations for laboratory confirmation (Serum Bilirubin / LFT).

---

## 🚀 How to Run

### Option 1: Double-click
Double-click `run_app.bat` inside `C:\Users\pc\jaundice_app\`.

### Option 2: Command Line
```powershell
python -m streamlit run "C:\Users\pc\jaundice_app\app.py"
```

The application will be accessible at:
👉 **http://localhost:8501**
