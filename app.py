import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
from datetime import datetime
import gdown

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Lung & Pancreatic Tumor Characterization",
    layout="wide"
)

# ================== CLEAN MEDICAL UI THEME ==================
st.markdown("""
<style>
html, body, [data-testid="stApp"] {
    background-color: #f8fafc;
    color: #0f172a;
}
h1 { color: #0f766e; font-weight: 700; }
h2, h3 { color: #155e75; }
.card {
    background-color: #ffffff;
    padding: 24px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    box-shadow: 0px 6px 18px rgba(15,23,42,0.08);
    margin-bottom: 24px;
}
.stTabs [aria-selected="true"] {
    color: #0f766e;
    font-weight: 600;
    border-bottom: 2px solid #0f766e;
}
</style>
""", unsafe_allow_html=True)

# ================== CONSTANTS ==================
IMG_SIZE = 224

MODEL_CONFIG = {
    "Lung": {
        "model_path": "lung_cancer_resnet50.h5",
        "classes": ["Normal", "Benign", "Malignant"],
        "last_conv": "conv5_block3_out",
        "file_id": "1UkTbZ_QzH6XEkcU_4bkVicG2BJQPbZVM"
    },
    "Pancreas": {
        "model_path": "pancreas_cancer_resnet50.h5",
        "classes": ["Normal", "Tumor"],
        "last_conv": "conv5_block3_out",
        "file_id": "17W3lAsJtVD6d7SoXMgT_7vjxBSsF4P50"
    }
}

# ================== MODEL DOWNLOAD ==================
def load_or_download_model(model_path, file_id):
    if not os.path.exists(model_path):
        with st.spinner("Downloading model file..."):
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                model_path,
                quiet=False
            )
    return tf.keras.models.load_model(model_path)

# ================== SESSION STATE ==================
if "history" not in st.session_state:
    st.session_state.history = []
if "gradcam" not in st.session_state:
    st.session_state.gradcam = None

# ================== IMAGE FUNCTIONS ==================
def preprocess_image(img):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def make_gradcam(img_array, model, last_conv):
    grad_model = tf.keras.models.Model(
        model.input,
        [model.get_layer(last_conv).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = tf.reduce_max(preds)

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap

def overlay_gradcam(image, heatmap):
    img = np.array(image)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# ================== SIDEBAR ==================
st.sidebar.markdown("## 🧾 Patient Information")
patient_name = st.sidebar.text_input("Patient Name")
age = st.sidebar.number_input("Age", 1, 120)
gender = st.sidebar.radio("Gender", ["Male", "Female", "Other"], horizontal=True)

# ================== HEADER ==================
st.title("Lung & Pancreatic Tumor Characterization")
st.markdown("**AI-assisted CT image analysis** · Academic & research use only")

organ = st.selectbox("Select Organ", ["Lung", "Pancreas"])

cfg = MODEL_CONFIG[organ]
model = load_or_download_model(cfg["model_path"], cfg["file_id"])

CLASS_NAMES = cfg["classes"]
LAST_CONV = cfg["last_conv"]

# ================== TABS ==================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧪 Diagnosis", "📊 Confidence", "🧠 Grad-CAM", "📜 History", "🧑‍⚕️ Report"
])

# ================== DIAGNOSIS ==================
with tab1:
    uploaded = st.file_uploader("Upload CT Images", type=["png","jpg","jpeg"], accept_multiple_files=True)
    if uploaded:
        preds_list = []
        for img_file in uploaded:
            image = Image.open(img_file)
            arr = preprocess_image(image)
            preds = model.predict(arr, verbose=0)[0]
            preds_list.append(preds)

        final_preds = np.mean(preds_list, axis=0)
        idx = np.argmax(final_preds)
        result = CLASS_NAMES[idx]
        conf = final_preds[idx] * 100

        heatmap = make_gradcam(arr, model, LAST_CONV)
        st.session_state.gradcam = overlay_gradcam(image, heatmap)

        st.markdown(f"**Diagnosis:** {result}  \n**Confidence:** {conf:.2f}%")
        st.session_state.history.append({
            "Patient": patient_name,
            "Organ": organ,
            "Result": result,
            "Confidence": round(conf,2)
        })

# ================== CONFIDENCE ==================
with tab2:
    if uploaded:
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, final_preds * 100)
        ax.set_ylim(0,100)
        st.pyplot(fig)

# ================== GRAD-CAM ==================
with tab3:
    if st.session_state.gradcam is not None:
        st.image(st.session_state.gradcam, use_container_width=True)

# ================== HISTORY ==================
with tab4:
    st.table(st.session_state.history)

# ================== REPORT ==================
with tab5:
    if uploaded:
        report = f"""
Patient: {patient_name}
Age/Gender: {age}/{gender}
Organ: {organ}
Diagnosis: {result}
Confidence: {conf:.2f}%
Generated: {datetime.now()}
"""
        st.download_button("Download Report", report, "report.txt")
