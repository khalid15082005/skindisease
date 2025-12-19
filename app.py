"""
Skin Disease AI - Streamlit Web Application
A user-friendly interface for skin lesion diagnosis using machine learning
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Skin Disease AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.main-header { font-size: 3rem; color: #1f77b4; text-align: center; }
.sub-header { font-size: 1.5rem; color: #ff7f0e; text-align: center; }
.prediction-box { background-color: #f0f2f6; padding: 1rem; border-radius: 10px; }
.confidence-bar { background-color: #e6f3ff; padding: 0.5rem; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# ---------------- Model Loader ----------------
@st.cache_resource
def load_model():
    model_path = "model"

    if not os.path.exists(model_path):
        st.warning("⚠️ Model not found. Using demo model.")

        class DemoModel:
            def predict(self, img_array, verbose=0):
                np.random.seed(42)
                logits = np.random.randn(6)
                exp = np.exp(logits - np.max(logits))
                return [exp / exp.sum()]

        return DemoModel()

    try:
        layer = tf.keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")
        inputs = tf.keras.Input(shape=(299, 299, 3))
        outputs = layer(inputs)
        return tf.keras.Model(inputs, outputs)
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

# ---------------- Prediction ----------------
def predict_skin_disease(model, img_array):
    img_array = tf.keras.applications.xception.preprocess_input(img_array)
    preds = model.predict(img_array, verbose=0)[0]

    classes = ['Acne', 'Carcinoma', 'Eczema', 'Keratosis', 'Millia', 'Rosacea']
    results = [
        {"class": c, "confidence": float(p * 100)}
        for c, p in zip(classes, preds)
    ]
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results

# ---------------- Main UI ----------------
st.markdown('<h1 class="main-header">🔬 Skin Disease AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ML-based Skin Disease Detection</p>', unsafe_allow_html=True)

model = load_model()
if model is None:
    st.stop()

uploaded = st.file_uploader("Upload skin image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB").resize((299, 299))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)

    if st.button("🔍 Diagnose"):
        with st.spinner("Analyzing..."):
            results = predict_skin_disease(model, arr)

        st.success("Done!")
        for r in results:
            st.write(f"**{r['class']}** — {r['confidence']:.2f}%")

st.markdown("---")
st.caption("⚠️ Educational use only. Consult a dermatologist.")
