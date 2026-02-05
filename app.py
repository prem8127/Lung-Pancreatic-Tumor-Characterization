import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
from datetime import datetime

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
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# ================== CONSTANTS ==================
IMG_SIZE = 224

MODEL_CONFIG = {
    "Lung": {
        "model_path": "lung_cancer_resnet50.h5",
        "classes": ["Normal", "Benign", "Malignant"],
        "last_conv": "conv5_block3_out"
    },
    "Pancreas": {
        "model_path": "pancreas_cancer_resnet50.h5",
        "classes": ["Normal", "Tumor"],
        "last_conv": "conv5_block3_out"
    }
}

# ================== SESSION STATE ==================
if "history" not in st.session_state:
    st.session_state.history = []
if "gradcam" not in st.session_state:
    st.session_state.gradcam = None

# ================== FUNCTIONS ==================
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
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
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# ================== SIDEBAR ==================
st.sidebar.markdown("## 🧾 Patient Information")
patient_name = st.sidebar.text_input("Patient Full Name", placeholder="e.g., Ramesh Kumar")
age = st.sidebar.number_input("Age", 1, 120)
gender = st.sidebar.radio("Gender", ["Male", "Female", "Other"], horizontal=True)
st.sidebar.caption("Academic & research use only")

# ================== HEADER ==================
st.title("Lung & Pancreatic Tumor Characterization")
st.markdown("**AI-assisted CT image analysis** · Academic & research use only")

organ = st.selectbox("Select Organ", ["Lung", "Pancreas"])

MODEL_PATH = MODEL_CONFIG[organ]["model_path"]
CLASS_NAMES = MODEL_CONFIG[organ]["classes"]
LAST_CONV = MODEL_CONFIG[organ]["last_conv"]

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

model = load_model(MODEL_PATH)

# ================== SAFE VARIABLES ==================
final_preds = None
result = None
conf = None

# ================== TABS ==================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🧪 Diagnostic Assessment",
    "📊 Confidence Chart",
    "🧠 Grad-CAM",
    "📜 Case History",
    "🧑‍⚕️ Doctor Summary",
    "⚠ Precautions"
])

# ================== DIAGNOSTIC ==================
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload CT Scan Image(s)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded:
        preds_list = []

        for img_file in uploaded:
            image = Image.open(img_file)
            img_array = preprocess_image(image)
            preds = model.predict(img_array, verbose=0)[0]
            preds_list.append(preds)

        final_preds = np.mean(preds_list, axis=0)
        idx = int(np.argmax(final_preds))
        conf = final_preds[idx] * 100
        result = CLASS_NAMES[idx]

        # ---- GRAD-CAM ----
        heatmap = make_gradcam(img_array, model, LAST_CONV)
        st.session_state.gradcam = overlay_gradcam(image, heatmap)

        st.markdown(f"""
### Diagnostic Summary
- **Organ:** {organ}  
- **Finding:** **{result}**  
- **Confidence:** **{conf:.2f}%**
""")

        st.session_state.history.append({
            "Patient": patient_name if patient_name else "Not Provided",
            "Organ": organ,
            "Finding": result,
            "Confidence (%)": round(conf, 2)
        })

    st.markdown('</div>', unsafe_allow_html=True)

# ================== CONFIDENCE CHART ==================
with tab2:
    if final_preds is not None:
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, final_preds * 100)
        ax.set_ylabel("Confidence (%)")
        ax.set_ylim(0, 100)
        st.pyplot(fig)
    else:
        st.info("Upload CT images to view confidence chart.")

# ================== GRAD-CAM VIEW ==================
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if st.session_state.gradcam is not None:
        st.subheader("Grad-CAM Visualization")
        st.image(
            st.session_state.gradcam,
            caption="Highlighted regions influencing prediction",
            use_container_width=True
        )
    else:
        st.info("Run diagnostic assessment to view Grad-CAM.")
    st.markdown('</div>', unsafe_allow_html=True)

# ================== HISTORY ==================
with tab4:
    if st.session_state.history:
        st.table(st.session_state.history)
    else:
        st.info("No cases recorded yet.")

# ================== DOCTOR SUMMARY + REPORT DOWNLOAD ==================
with tab5:
    if result is not None:
        report = f"""
AI-Assisted Tumor Characterization Report
----------------------------------------
Patient Name : {patient_name if patient_name else "Not Provided"}
Age / Gender : {age} / {gender}
Organ        : {organ}
Diagnosis    : {result}
Confidence   : {conf:.2f}%

Generated On : {datetime.now().strftime('%Y-%m-%d %H:%M')}

⚠ This report is for academic and research purposes only.
"""

        st.markdown(f"""
### Clinical Summary

**Patient:** {patient_name if patient_name else "Not Provided"}  
**Age / Gender:** {age} / {gender}  
**Organ:** {organ}  
**Diagnosis:** **{result}**  
**Confidence:** {conf:.2f}%
""")

        st.download_button(
            "📄 Download Report (TXT)",
            report,
            file_name="Tumor_Analysis_Report.txt",
            mime="text/plain"
        )
    else:
        st.info("Upload CT images to generate summary and report.")

# ================== PRECAUTIONS ==================
with tab6:
    st.markdown("""
• Not for real clinical diagnosis  
• Depends on dataset and image quality  
• For academic & research use only  
""")
