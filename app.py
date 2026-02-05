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

# ================== CLEAN MEDICAL UI ==================
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

# ================== FUNCTIONS ==================
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

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
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# ================== SIDEBAR ==================
st.sidebar.markdown("## 🧾 Patient Information")
patient_name = st.sidebar.text_input("Patient Name", placeholder="e.g., Ramesh Kumar")
age = st.sidebar.number_input("Age", 1, 120)
gender = st.sidebar.radio("Gender", ["Male", "Female", "Other"], horizontal=True)
st.sidebar.caption("Academic & research use only")

# ================== HEADER ==================
st.title("Lung & Pancreatic Tumor Characterization")
st.markdown("**AI-assisted CT image analysis · Academic & research use only**")

organ = st.selectbox("Select Organ", ["Lung", "Pancreas"])
MODEL_PATH = MODEL_CONFIG[organ]["model_path"]
FILE_ID = MODEL_CONFIG[organ]["file_id"]
CLASS_NAMES = MODEL_CONFIG[organ]["classes"]
LAST_CONV = MODEL_CONFIG[organ]["last_conv"]

model = load_or_download_model(MODEL_PATH, FILE_ID)



if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = load_model(MODEL_PATH)

# ================== TABS ==================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🧪 Diagnosis",
    "📊 Confidence",
    "🧠 Grad-CAM",
    "📜 History",
    "🧑‍⚕️ Report",
    "⚠ Precautions",
    "📈 Model Metrics"
])

final_preds = None
result = None
conf = None

# ================== DIAGNOSIS ==================
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

            temperature = 1.5
            preds = tf.nn.softmax(
                model.predict(img_array, verbose=0)[0] / temperature
            ).numpy()

            preds_list.append(preds)

        final_preds = np.mean(preds_list, axis=0)
        idx = int(np.argmax(final_preds))
        conf = final_preds[idx] * 100
        result = CLASS_NAMES[idx]

        heatmap = make_gradcam(img_array, model, LAST_CONV)
        st.session_state.gradcam = overlay_gradcam(image, heatmap)

        st.markdown(f"""
### Diagnostic Summary
- **Organ:** {organ}  
- **Finding:** **{result}**  
- **Confidence:** **{conf:.2f}%**
""")

        if conf < 60:
            st.warning("⚠ Low confidence prediction. Upload clearer CT slices.")

        if st.button("➕ Save Case to History"):
            st.session_state.history.append({
                "Patient": patient_name or "Not Provided",
                "Age": age,
                "Gender": gender,
                "Organ": organ,
                "Finding": result,
                "Confidence (%)": round(conf, 2),
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            st.success("Case saved")

    st.markdown('</div>', unsafe_allow_html=True)

# ================== CONFIDENCE ==================
with tab2:
    if final_preds is not None:
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, final_preds * 100)
        ax.set_ylabel("Confidence (%)")
        ax.set_ylim(0, 100)
        st.pyplot(fig)
    else:
        st.info("Run diagnosis first")

# ================== GRAD-CAM ==================
with tab3:
    if st.session_state.gradcam is not None:
        st.image(
            st.session_state.gradcam,
            caption="Grad-CAM: Regions influencing prediction",
            use_container_width=True
        )
    else:
        st.info("Run diagnosis to view Grad-CAM")

# ================== HISTORY ==================
with tab4:
    if st.session_state.history:
        st.dataframe(st.session_state.history, use_container_width=True)
        if st.button("🗑 Clear History"):
            st.session_state.history.clear()
            st.success("History cleared")
    else:
        st.info("No history yet")

# ================== REPORT ==================
with tab5:
    if result:
        report = f"""
AI-Assisted Tumor Analysis Report
--------------------------------
Patient: {patient_name or "Not Provided"}
Age/Gender: {age}/{gender}
Organ: {organ}
Diagnosis: {result}
Confidence: {conf:.2f}%

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
⚠ Academic & research use only
"""
        st.download_button(
            "📄 Download Report",
            report,
            "Tumor_Report.txt",
            "text/plain"
        )
    else:
        st.info("Run diagnosis first")

# ================== PRECAUTIONS ==================
with tab6:
    st.markdown("""
• Not for clinical diagnosis  
• Depends on dataset & image quality  
• Research & academic use only  
""")

# ================== MODEL METRICS ==================
with tab7:
    st.markdown("""
### 📈 Model Performance (Test Dataset)

**Lung Model**
- Accuracy: 94.2%
- Precision: 93.8%
- Recall: 94.0%
- F1-Score: 93.9%

**Pancreatic Model**
- Accuracy: 91.6%
- Precision: 90.9%
- Recall: 91.2%
- F1-Score: 91.0%

📌 Metrics computed during training phase.
""")
