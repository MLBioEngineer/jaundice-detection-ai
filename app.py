import os
import io
import joblib
import cv2
import requests
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="IcterusLens AI - Jaundice Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling to Hide "Hosted with Streamlit", Menus, and Branding
st.markdown("""
<style>
    /* Completely Hide Streamlit Menus, Headers, and Footers */
    #MainMenu {visibility: hidden !important; display: none !important;}
    header {visibility: hidden !important; display: none !important;}
    footer {visibility: hidden !important; display: none !important;}
    footer:after {content: "" !important; display: none !important;}
    
    /* Ensure the sidebar toggle remains accessible */
    [data-testid="collapsedControl"] {
        display: block !important;
        visibility: visible !important;
        top: 10px !important;
        left: 10px !important;
        z-index: 999999 !important;
    }
    
    /* Hide Streamlit Cloud "Hosted with Streamlit" Badge & Viewer Badges */
    .viewerBadge_container__1QSob {display: none !important; visibility: hidden !important;}
    .viewerBadge_link__146K3 {display: none !important; visibility: hidden !important;}
    a[class*="viewerBadge"] {display: none !important; visibility: hidden !important;}
    div[class*="viewerBadge"] {display: none !important; visibility: hidden !important;}
    span[class*="viewerBadge"] {display: none !important; visibility: hidden !important;}
    div[class*="StyledFooter"] {display: none !important; visibility: hidden !important;}
    div[class*="StreamlitBranding"] {display: none !important; visibility: hidden !important;}
    a[href*="streamlit.io"] {display: none !important; visibility: hidden !important;}
    a[href*="streamlit.app"] {display: none !important; visibility: hidden !important;}
    
    /* Hide Cloud Toolbar, Deploy Buttons & GitHub profile elements */
    .stDeployButton {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    [data-testid="stStatusWidget"] {display: none !important;}
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stToolbarActions"] {display: none !important;}
    [data-testid="manage-app-button"] {display: none !important; visibility: hidden !important;}
    div[class*="ProfileBadge"] {display: none !important; visibility: hidden !important;}
    div[class*="manageApp"] {display: none !important; visibility: hidden !important;}
    a[href*="github.com"] {display: none !important;}
    img[src*="avatars.githubusercontent.com"] {display: none !important;}
    svg[data-icon="github"] {display: none !important;}
    
    .main-header {
        background: linear-gradient(135deg, #0d3b66 0%, #001e3d 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
    }
    .main-header h1 {
        color: #f4d35e;
        margin: 0;
        font-size: 2.2rem;
    }
    .main-header p {
        color: #e0e1dd;
        margin-top: 6px;
        font-size: 1rem;
    }
    .badge-jaundice {
        background-color: #feeaa7;
        color: #d63031;
        padding: 10px 18px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.15rem;
        display: inline-block;
        border: 2px solid #e17055;
    }
    .badge-healthy {
        background-color: #d4edda;
        color: #155724;
        padding: 10px 18px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.15rem;
        display: inline-block;
        border: 2px solid #28a745;
    }
    .badge-eye-verified {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 6px 14px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #81c784;
        margin-bottom: 12px;
    }
    .rejection-box {
        background-color: #ffebee;
        border-left: 6px solid #d32f2f;
        padding: 20px 24px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(211, 47, 47, 0.12);
    }
    .zoom-box {
        background-color: #e3f2fd;
        border-left: 6px solid #1976d2;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Dynamic Cloud Badge & "Hosted with Streamlit" Remover
components.html("""
<script>
    function removeStreamlitBranding() {
        try {
            const targets = [
                'footer',
                '.viewerBadge_container__1QSob',
                'a[class*="viewerBadge"]',
                'div[class*="viewerBadge"]',
                'span[class*="viewerBadge"]',
                '[data-testid="stStatusWidget"]',
                '[data-testid="manage-app-button"]',
                'div[class*="manageApp"]',
                'a[href*="streamlit.io"]'
            ];
            // Clean inside current frame
            targets.forEach(sel => {
                document.querySelectorAll(sel).forEach(el => el.remove());
            });
            // Clean inside parent window (if running on Streamlit Cloud)
            if (window.parent && window.parent.document) {
                targets.forEach(sel => {
                    window.parent.document.querySelectorAll(sel).forEach(el => el.remove());
                });
            }
        } catch(e) {}
    }
    removeStreamlitBranding();
    setInterval(removeStreamlitBranding, 400);
</script>
""", height=0, width=0)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "jaundice_ensemble_model.joblib")
SAMPLES_DIR = os.path.join(APP_DIR, "sample_images")

@st.cache_resource
def load_classifier():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}")
        return None
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_cascades():
    face_c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eye_tree_c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
    return face_c, eye_c, eye_tree_c

model = load_classifier()
face_cascade, eye_cascade, eye_tree_cascade = load_cascades()

def process_and_extract_eye(im_pil):
    """
    Intelligent Eye Localizer & Clinical Gatekeeper:
    1. Full face: Requires adequate close-up resolution (face >= 260px or >= 38% width),
       frontal gaze with open eye, visible iris/pupil and sclera. Distant sitting or side-profile photos are rejected.
    2. Close-up eye macro: Verifies human eye geometry and biological tones.
    3. Strict non-eye rejection: Rejects shoes, clothes, cars, landscapes, sunglasses, or occluded eyes.
    """
    im_rgb = im_pil.convert("RGB")
    arr = np.array(im_rgb)
    h_orig, w_orig, _ = arr.shape
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    
    # Check for faces (minSize 140x140 prevents false face detection inside close-up eyes)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(140, 140))
    
    if len(faces) > 0:
        # Sort by largest face
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        fx, fy, fw, fh = faces[0]
        
        # Face must be large enough: close-up portrait (not far sitting/standing body shot)
        if fw < 260 and (fw / w_orig) < 0.38:
            return None, False, "ছবিটি অনেক দূর থেকে তোলা হয়েছে (Face is too far away). দূর থেকে তোলা ছবিতে চোখের রেজোলিউশন অত্যন্ত কম থাকায় জন্ডিস পরীক্ষা করা যায় না। অনুগ্রহ করে ক্যামেরার একদম কাছে এসে চোখের স্পষ্ট ছবি তুলুন।"
            
        # Upper face zone where eyes sit (18% to 58% down the face)
        eye_zone_y1 = max(0, fy + int(fh * 0.18))
        eye_zone_y2 = min(h_orig, fy + int(fh * 0.58))
        eye_zone_x1 = max(0, fx + int(fw * 0.08))
        eye_zone_x2 = min(w_orig, fx + int(fw * 0.92))
        
        eye_zone_gray = gray[eye_zone_y1:eye_zone_y2, eye_zone_x1:eye_zone_x2]
        
        # Detect individual eye within face eye zone
        eyes = eye_cascade.detectMultiScale(eye_zone_gray, scaleFactor=1.1, minNeighbors=3)
        if len(eyes) == 0:
            eyes = eye_tree_cascade.detectMultiScale(eye_zone_gray, scaleFactor=1.1, minNeighbors=2)
            
        if len(eyes) == 0:
            return None, False, "মুখমণ্ডলে কোনো খোলা চোখ শনাক্ত করা যায়নি (No open eye detected). সরাসরি ক্যামেরার দিকে তাকিয়ে দুই চোখ খোলা রেখে ছবি তুলুন।"
            
        ex, ey, ew, eh = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[0]
        
        # Eye must have sufficient resolution (at least 50x35 px)
        if ew < 50 or eh < 35:
            return None, False, f"চোখের রেজোলিউশন খুব কম ({ew}x{eh}px). চোখের স্ক্লেরা (সাদা অংশ) স্পষ্ট দেখতে কমপক্ষে ৫০ পিক্সেল রেজোলিউশন প্রয়োজন। ক্যামেরার আরো কাছে আসুন।"
            
        pad_x = int(ew * 0.25)
        pad_y = int(eh * 0.25)
        crop_y1 = max(0, eye_zone_y1 + ey - pad_y)
        crop_y2 = min(h_orig, eye_zone_y1 + ey + eh + pad_y)
        crop_x1 = max(0, eye_zone_x1 + ex - pad_x)
        crop_x2 = min(w_orig, eye_zone_x1 + ex + ew + pad_x)
        cropped_eye = im_rgb.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        
        # Check sclera and iris visibility in cropped eye
        arr_eye = np.array(cropped_eye.resize((128, 128)), dtype=np.float32) / 255.0
        r, g, b = arr_eye[:,:,0], arr_eye[:,:,1], arr_eye[:,:,2]
        lum = 0.299*r + 0.587*g + 0.114*b
        
        # Dark iris/pupil must be present
        if np.min(lum) > 0.35:
            return cropped_eye, False, "চোখের মনি বা পিউপিল দৃশ্যমান নয়। সানগ্লাস পরা থাকলে তা খুলে সরাসরি ক্যামেরার দিকে তাকিয়ে ছবি তুলুন।"
            
        # Sclera must be visible (bright pixels must be present)
        bright_mask = lum >= 0.45
        if np.sum(bright_mask) < 25:
            return cropped_eye, False, "চোখের স্ক্লেরা (সাদা অংশ) স্পষ্ট দেখা যাচ্ছে না। চোখ পুরোপুরি উন্মুক্ত রেখে ছবি তুলুন।"
            
        return cropped_eye, True, "Full face verified. Automatically zoomed and cropped into the eye!"
        
    # Check if already a close-up eye macro
    eyes1 = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    eyes2 = eye_tree_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    
    # Check warm biological colors
    arr_f = arr.astype(np.float32) / 255.0
    r, g, b = arr_f[:,:,0], arr_f[:,:,1], arr_f[:,:,2]
    is_warm = (np.mean(r) >= np.mean(b) - 0.05) and (np.mean(r) >= np.mean(g) - 0.10)
    
    if (len(eyes1) > 0 or len(eyes2) > 0) and is_warm:
        return im_rgb, True, "Close-up human eye verified."
        
    return None, False, "কোনো মানুষের চোখ বা মুখমণ্ডল পাওয়া যায়নি। জুতা, কাপড়, গাছপালা, পশু বা দূরবর্তী প্রাকৃতিক ছবি বাতিল করা হয়েছে।"

def extract_features_from_pil(im):
    im_rgb = im.convert("RGB")
    im_resized = im_rgb.resize((128, 128))
    arr = np.array(im_resized, dtype=np.float32) / 255.0
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    
    yellow_index = (r + g) - 2.0 * b
    b_diff = (r - b) + (g - b)
    
    lum = 0.299*r + 0.587*g + 0.114*b
    bright_mask = lum > np.percentile(lum, 35)
    if np.sum(bright_mask) > 10:
        sclera_r, sclera_g, sclera_b = np.mean(r[bright_mask]), np.mean(g[bright_mask]), np.mean(b[bright_mask])
        sclera_yellow = np.mean(yellow_index[bright_mask])
        sclera_bdiff = np.mean(b_diff[bright_mask])
    else:
        sclera_r, sclera_g, sclera_b, sclera_yellow, sclera_bdiff = np.mean(r), np.mean(g), np.mean(b), np.mean(yellow_index), np.mean(b_diff)
        
    hsv = im_resized.convert("HSV")
    hsv_arr = np.array(hsv, dtype=np.float32)
    h, s, v = hsv_arr[:,:,0]/255.0, hsv_arr[:,:,1]/255.0, hsv_arr[:,:,2]/255.0
    
    ycbcr = im_resized.convert("YCbCr")
    y_arr = np.array(ycbcr, dtype=np.float32)
    cb, cr = y_arr[:,:,1]/255.0, y_arr[:,:,2]/255.0
    
    feats = [
        np.mean(r), np.std(r), np.mean(g), np.std(g), np.mean(b), np.std(b),
        np.mean(yellow_index), np.std(yellow_index),
        sclera_r, sclera_g, sclera_b, sclera_yellow, sclera_bdiff,
        np.mean(h), np.mean(s), np.mean(v),
        np.mean(cb), np.std(cb), np.mean(cr), np.std(cr)
    ]
    for channel in [r, g, b, yellow_index]:
        hist, _ = np.histogram(channel, bins=12, range=(float(np.min(channel)), float(np.max(channel))))
        feats.extend(hist / 16384.0)
        
    diagnostics = {
        "sclera_yellow": float(sclera_yellow),
        "sclera_r": float(sclera_r),
        "sclera_g": float(sclera_g),
        "sclera_b": float(sclera_b),
        "bright_mask": bright_mask,
        "r_mean": float(np.mean(r)),
        "g_mean": float(np.mean(g)),
        "b_mean": float(np.mean(b)),
        "resized_img": im_resized
    }
    return np.array(feats, dtype=np.float32).reshape(1, -1), diagnostics

# Header
st.markdown("""
<div class='main-header'>
    <h1>👁️ IcterusLens AI — Jaundice Sclera Screening System</h1>
    <p>Non-invasive screening for Scleral Icterus (Jaundice) using Computer Vision & AI Ensemble Colorimetry</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Clinical Information")
st.sidebar.info("""
ℹ️ **Scleral Icterus** is the yellowing of the sclera (white of the eye).
It is one of the earliest clinical indicators of jaundice, appearing when serum total bilirubin exceeds **2.0–2.5 mg/dL**.

🛡️ **Strict Eye & Face Gatekeeper**:
- Non-eye images (shoes, clothing, cars, animals) are **strictly rejected**.
- Full face photos are **automatically zoomed** into the eye region.
- Photos with dark sunglasses or closed eyes are **rejected** because visible sclera is required.
""")

# Top Input Section directly on MAIN PAGE
st.markdown("### 📤 Select or Upload Image (ছবি আপলোড বা নির্বাচন করুন)")

tab_upload, tab_camera, tab_esp32, tab_samples = st.tabs([
    "📁 Upload Image (ছবি আপলোড)",
    "📷 Live WebCam (লাইভ ক্যামেরা)",
    "📡 ESP32-CAM Hardware (আইওটি ডিভাইস)",
    "🧪 Pre-loaded Samples (পরীক্ষামূলক নমুনা)"
])

selected_image = None
image_label = ""

with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose an eye or face photo from your phone or PC (JPG, JPEG, PNG, WEBP, BMP):",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="main_file_uploader",
        help="Upload either a close-up eye photo or a face portrait looking at the camera."
    )
    if uploaded_file is not None:
        try:
            im = Image.open(uploaded_file)
            selected_image = ImageOps.exif_transpose(im)
            image_label = uploaded_file.name
            st.success(f"✅ Image loaded: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Could not read the uploaded image: {e}")

with tab_camera:
    camera_photo = st.camera_input("Take a photo of the face or eye", key="main_camera_input")
    if camera_photo is not None:
        try:
            im = Image.open(camera_photo)
            selected_image = ImageOps.exif_transpose(im)
            image_label = "Live Camera Capture"
            st.success("✅ Camera photo captured successfully!")
        except Exception as e:
            st.error(f"Could not process camera image: {e}")

with tab_esp32:
    st.markdown("##### 🔬 ESP32-CAM Macro Scanner Integration")
    st.caption("Capture real-time eye sclera photos directly from your ESP32-CAM hardware over Wi-Fi.")
    
    col_ip, col_btn = st.columns([3, 1])
    with col_ip:
        esp32_ip = st.text_input(
            "ESP32-CAM URL / IP Endpoint:",
            value=st.session_state.get("saved_esp32_ip", "http://192.168.4.1/capture"),
            key="esp32_ip_input",
            help="Enter the IP address shown in your Arduino Serial Monitor (e.g. http://192.168.4.1/capture or http://192.168.1.105/capture)"
        )
    with col_btn:
        st.write("")
        st.write("")
        btn_capture_esp32 = st.button("📸 Capture Eye Image", key="btn_capture_esp32", use_container_width=True)
        
    if btn_capture_esp32:
        if not esp32_ip:
            st.error("Please enter the ESP32-CAM IP address!")
        else:
            clean_url = esp32_ip.strip()
            if not clean_url.startswith("http://") and not clean_url.startswith("https://"):
                clean_url = "http://" + clean_url
            if not clean_url.endswith("/capture") and not clean_url.endswith(".jpg"):
                clean_url = clean_url.rstrip("/") + "/capture"
                
            with st.spinner("📡 Connecting to ESP32-CAM hardware and capturing eye photo..."):
                try:
                    resp = requests.get(clean_url, timeout=5)
                    if resp.status_code == 200:
                        im = Image.open(io.BytesIO(resp.content))
                        selected_image = ImageOps.exif_transpose(im)
                        image_label = f"ESP32-CAM Hardware ({clean_url})"
                        st.session_state["saved_esp32_ip"] = esp32_ip
                        st.success("✅ Photo captured successfully from ESP32-CAM!")
                    else:
                        st.error(f"ESP32-CAM responded with HTTP {resp.status_code}")
                except Exception as ex:
                    st.error(f"❌ Failed to reach ESP32-CAM at `{clean_url}`. Please verify your laptop is connected to the ESP32 Wi-Fi / Hotspot. Details: {ex}")

with tab_samples:
    sample_options = {
        "Normal / Healthy Eye (Sample 1)": "test_Healthy_0.jpg",
        "Normal / Healthy Eye (Sample 2)": "test_Healthy_1.jpg",
        "Normal / Healthy Eye (Sample 3)": "test_Healthy_2.jpg",
        "Jaundice / Scleral Icterus (Sample 1)": "test_Jaundice_1.jpg",
        "Jaundice / Scleral Icterus (Sample 2)": "test_Jaundice_2.jpg",
        "Jaundice / Scleral Icterus (Sample 3)": "test_Jaundice_3.jpg",
    }
    choice = st.selectbox("Select clinical sample case:", list(sample_options.keys()), key="sample_selector")
    sample_path = os.path.join(SAMPLES_DIR, sample_options[choice])
    if os.path.exists(sample_path):
        if st.button("Load This Sample", key="btn_load_sample") or selected_image is None:
            try:
                selected_image = Image.open(sample_path)
                image_label = choice
            except Exception as e:
                st.error(f"Error loading sample: {e}")

st.divider()

# Main Display & Intelligent Eye Processing
if selected_image is not None:
    # RUN INTELLIGENT EYE LOCALIZER & GATEKEEPER
    eye_img, is_valid_eye, gate_message = process_and_extract_eye(selected_image)
    
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        st.subheader("📸 Uploaded Subject Image")
        st.image(selected_image, caption=f"Original Input: {image_label}", use_container_width=True)
        
        if eye_img is not None and eye_img != selected_image:
            st.markdown('<div class="zoom-box">🔍 <strong>Auto-Zoomed Eye Region (স্বয়ংক্রিয়ভাবে জুম করা চোখের অংশ)</strong></div>', unsafe_allow_html=True)
            st.image(eye_img, caption="Cropped Eye Analyzed by AI", use_container_width=True)
            
    if not is_valid_eye:
        # STRICT REJECTION BRANCH - ZERO PREDICTIONS
        with col_right:
            st.markdown(f"""
            <div class="rejection-box">
                <h3 style="color: #c62828; margin: 0 0 10px 0;">❌ Prediction Rejected: Not a Valid Eye Image</h3>
                <h4 style="color: #b71c1c; margin: 0 0 12px 0;">(কোনো স্পষ্ট চোখ পাওয়া যায়নি — পরীক্ষা বাতিল)</h4>
                <p style="color: #212121; font-size: 1.05rem; margin-bottom: 8px;">
                    <strong>Detailed Reason:</strong> {gate_message}
                </p>
                <hr style="border: 0; border-top: 1px solid #ffcdd2; margin: 12px 0;">
                <p style="color: #424242; font-size: 0.95rem; margin: 0;">
                    🛡️ <strong>Safety Guarantee:</strong> This application is trained <strong>exclusively on human eye sclera data</strong>.
                    Images of <strong>shoes, clothing, animals, landscapes, or faces with dark sunglasses</strong> are strictly blocked to prevent false diagnoses.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.warning("""
            **📋 Guidelines for a valid test (সঠিকভাবে পরীক্ষা করার নিয়ম):**
            1. **Take a closer photo**: The face or eye must be close enough so the eye details are clear.
            2. **No sunglasses**: Do not wear dark glasses or sunglasses.
            3. **Keep eyes open**: Sclera (white of the eye) must be visible.
            4. **Good lighting**: Use natural, even lighting.
            """)
    else:
        # VALID EYE ANALYSIS BRANCH
        target_eye = eye_img if eye_img is not None else selected_image
        feats, diag = extract_features_from_pil(target_eye)
        
        with col_left:
            st.markdown("#### 🔍 Sclera Detection Region of Interest (ROI)")
            mask_overlay = np.array(diag["resized_img"]).copy()
            mask = diag["bright_mask"]
            mask_overlay[~mask] = (mask_overlay[~mask] * 0.45).astype(np.uint8)
            st.image(mask_overlay, caption="Detected Scleral Area Analyzed for Bilirubin Pigmentation", use_container_width=True)

        with col_right:
            st.subheader("🔬 AI Diagnostic Assessment")
            st.markdown(f'<div class="badge-eye-verified">👁️ {gate_message}</div>', unsafe_allow_html=True)
            
            if model is not None:
                probabilities = model.predict_proba(feats)[0]
                healthy_prob = float(probabilities[0])
                jaundice_prob = float(probabilities[1])
                
                # Clinical screening thresholding:
                # >= 55%: Positive for Jaundice (Scleral Icterus)
                # 45% - 55%: Borderline / Inconclusive due to ambient lighting
                # < 45%: Negative (Normal Healthy Sclera)
                if jaundice_prob >= 0.55:
                    predicted_status = "POSITIVE"
                    st.markdown(
                        """<div style="margin: 15px 0;"><span class="badge-jaundice">⚠️ POSITIVE: Scleral Icterus (Jaundice) Detected</span></div>""",
                        unsafe_allow_html=True
                    )
                elif jaundice_prob >= 0.45:
                    predicted_status = "BORDERLINE"
                    st.markdown(
                        """<div style="margin: 15px 0;"><span style="background-color: #fff3cd; color: #856404; padding: 10px 18px; border-radius: 20px; font-weight: bold; font-size: 1.15rem; display: inline-block; border: 2px solid #ffeeba;">⚡ BORDERLINE / INCONCLUSIVE (অস্পষ্ট / পুনরায় পরীক্ষা আবশ্যক)</span></div>""",
                        unsafe_allow_html=True
                    )
                    st.info("💡 পরিবেষ্টিত আলোকস্বল্পতা বা ঘরের হলুদাভ আলোর কারণে ফলাফল স্পষ্ট নয়। নিশ্চিত হওয়ার জন্য পর্যাপ্ত প্রাকৃতিক আলোতে ক্যামেরার কাছে এসে পুনরায় ছবি তুলুন।")
                else:
                    predicted_status = "NEGATIVE"
                    st.markdown(
                        """<div style="margin: 15px 0;"><span class="badge-healthy">✅ NEGATIVE: Normal Sclera (Healthy)</span></div>""",
                        unsafe_allow_html=True
                    )
                    
                st.markdown("##### Model Confidence Breakdown")
                c1, c2 = st.columns(2)
                c1.metric("Jaundice Probability", f"{jaundice_prob * 100:.1f}%")
                c2.metric("Healthy Baseline Probability", f"{healthy_prob * 100:.1f}%")
                
                st.progress(float(jaundice_prob), text=f"Jaundice Risk Index: {jaundice_prob*100:.1f}%")
                
                # Check for warm ambient lighting cast
                if diag["b_mean"] < diag["r_mean"] - 0.22:
                    st.warning("⚠️ **Lighting Alert (আলোক সতর্কবার্তা)**: ছবিতে ঘরের মৃদু বা হলুদাভ আলোর প্রভাব রয়েছে। জন্ডিসের সঠিক ফলাফলের জন্য নিরপেক্ষ দিনের আলোতে ছবি তোলার পরামর্শ দেওয়া হচ্ছে।")
                
                st.markdown("##### 📊 Colorimetric Chromaticity Indices")
                m1, m2, m3 = st.columns(3)
                m1.metric("Yellow Index", f"{diag['sclera_yellow']:.3f}")
                m2.metric("Sclera Red/Blue", f"{diag['sclera_r'] / (diag['sclera_b'] + 1e-4):.2f}")
                m3.metric("Sclera Green/Blue", f"{diag['sclera_g'] / (diag['sclera_b'] + 1e-4):.2f}")
                
                color_df = pd.DataFrame({
                    "Channel": ["Red", "Green", "Blue"],
                    "Sclera Intensity": [diag["sclera_r"]*255, diag["sclera_g"]*255, diag["sclera_b"]*255]
                })
                st.bar_chart(color_df.set_index("Channel"), color="#f4d35e")
                
                st.markdown("##### 📋 Recommended Clinical Next Steps")
                if predicted_status == "POSITIVE":
                    st.warning("""
                    - **Serum Bilirubin Test (TSB)**: Prompt laboratory blood test (Total, Direct, and Indirect Bilirubin).
                    - **Liver Function Panel (LFT)**: ALT, AST, ALP, and Albumin check.
                    - **Consultation**: Please consult a Physician or Hepatologist for confirmation and medical intervention.
                    """)
                elif predicted_status == "BORDERLINE":
                    st.warning("""
                    - **Retake Photo**: Capture a macro photo of the eye under neutral daylight or with camera flash.
                    - **Observe Symptoms**: Check if urine is dark or if skin looks yellow.
                    - If symptoms persist or liver disorder is suspected, consult a healthcare provider.
                    """)
                else:
                    st.success("""
                    - No significant scleral yellowing detected.
                    - Maintain healthy liver function with proper hydration and balanced nutrition.
                    - If you experience other symptoms (dark urine, fatigue, abdominal pain), consult a medical professional.
                    """)
else:
    col_left, col_right = st.columns([1, 1], gap="large")
    with col_left:
        st.info("👆 Please upload an image above, take a photo with your camera, or select a pre-loaded sample.")
    with col_right:
        st.markdown("""
        ### How the IcterusLens AI System Works
        
        1. **Intelligent Eye & Face Localizer**:
           - **Close-up Eye Photos**: Verifies human eye geometry and biological tones.
           - **Full Face Photos**: Automatically detects the face, zooms into the eye band, and crops the eye.
           - **Non-Eye Rejection**: Rejects shoes, clothes, cars, distant landscapes, or sunglasses.
           
        2. **Sclera Segmentation & Multi-Space Colorimetry**:
           Isolates the scleral region of interest and computes the **Yellow Chromaticity Index**:
           $$\\text{Yellow Index} = (R + G) - 2B$$
           along with chrominance shifts in $RGB$, $HSV$, and $YCbCr$ color spaces.
        
        3. **Machine Learning Ensemble**:
           Leverages Random Forest + Extra Trees trained directly on the annotated clinical sclera dataset in `Jaundice_Dataset.zip`.
        """)

st.divider()
st.subheader("🩺 Patient Symptom & Risk Factor Calculator")
with st.expander("Click to expand Clinical Symptom Assessment"):
    sc1, sc2, sc3 = st.columns(3)
    s1 = sc1.checkbox("Yellowing of the skin (Icterus)")
    s2 = sc1.checkbox("Dark or tea-colored urine")
    s3 = sc2.checkbox("Pale, clay-colored stool")
    s4 = sc2.checkbox("Persistent fatigue or lethargy")
    s5 = sc3.checkbox("Right upper quadrant abdominal pain")
    s6 = sc3.checkbox("Pruritus (Severe generalized itching)")
    
    symptom_count = sum([s1, s2, s3, s4, s5, s6])
    if symptom_count >= 3:
        st.error(f"⚠️ High Symptom Load ({symptom_count}/6 clinical signs checked). Immediate medical evaluation is strongly recommended.")
    elif symptom_count >= 1:
        st.warning(f"⚡ Moderate Symptom Load ({symptom_count}/6 clinical signs checked). Monitor symptoms and schedule a routine physician check.")
    else:
        st.success("Zero additional symptoms reported.")

st.caption("Disclaimer: This tool is intended for research, screening, and educational decision-support purposes only. It does not replace a clinical examination or serum total bilirubin laboratory tests.")
