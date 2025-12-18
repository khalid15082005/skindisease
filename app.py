"""
Skin Disease AI - Streamlit Web Application
A user-friendly interface for skin lesion diagnosis using machine learning
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
import os

# Page configuration
st.set_page_config(
    page_title="Skin Disease AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e6f3ff;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_model():
    """Load the trained model with compatibility handling for Keras 3"""
    try:
        model_path = "model"
        if os.path.exists(model_path):
            # Check if the SavedModel has all required files
            model_files = os.listdir(model_path)
            if 'variables' not in str(model_files) or 'variables' not in os.listdir(model_path):
                st.warning("⚠️ SavedModel appears to be incomplete (missing variables directory).")
                st.info("💡 Creating a demo model for demonstration purposes...")
                
                # Create a demo model for demonstration
                class DemoModel:
                    def predict(self, img_array, verbose=0):
                        # Generate random predictions for demonstration
                        np.random.seed(42)  # For consistent results
                        # Generate random logits and apply softmax manually
                        logits = np.random.randn(6)
                        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
                        prediction = exp_logits / np.sum(exp_logits)
                        return [prediction]
                
                model = DemoModel()
                st.success("✅ Demo model loaded successfully")
                st.warning("⚠️ This is a demonstration model with random predictions. For actual predictions, please provide a complete model.")
                return model
            
            # Try using TFSMLayer approach
            try:
                tf_sm_layer = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
                
                # Create a functional model with the TFSMLayer
                inputs = tf.keras.Input(shape=(299, 299, 3), name='input_image')
                outputs = tf_sm_layer(inputs)
                model = tf.keras.Model(inputs=inputs, outputs=outputs)
                
                st.success("✅ Model loaded successfully using TFSMLayer")
                return model
            except Exception as e:
                st.error(f"❌ TFSMLayer loading failed: {str(e)}")
                return None
        else:
            st.error(f"Model not found at {model_path}. Please ensure the model files are in the correct location.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Prediction function
def predict_skin_disease(model, img_array):
    """Make prediction on the uploaded image"""
    try:
        # Preprocess image for Xception model
        img_array = tf.keras.applications.xception.preprocess_input(img_array)
        
        # Make prediction based on model type
        if hasattr(model, 'predict'):
            # Standard Keras model or demo model
            prediction = model.predict(img_array, verbose=0)[0]
        elif hasattr(model, 'signatures'):
            # TensorFlow SavedModel
            serving_fn = model.signatures['serving_default']
            prediction = serving_fn(tf.constant(img_array))['output_0'].numpy()[0]
        else:
            # Try direct call
            prediction = model(img_array).numpy()[0]
        
        # Define class names
        class_names = ['Acne', 'Carcinoma', 'Eczema', 'Keratosis', 'Millia', 'Rosacea']
        
        # Get top prediction
        top_prediction_idx = np.argmax(prediction)
        top_confidence = prediction[top_prediction_idx] * 100
        
        # Create results dictionary
        results = []
        for i, (class_name, confidence) in enumerate(zip(class_names, prediction)):
            results.append({
                'class': class_name,
                'confidence': confidence * 100,
                'is_top': i == top_prediction_idx
            })
        
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return results, top_prediction_idx, top_confidence
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 Skin Disease AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Machine Learning for Skin Lesion Diagnosis</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.info("""
        This AI system can diagnose 6 types of skin conditions:
        - **Acne** - Common skin condition with pimples
        - **Carcinoma** - Skin cancer
        - **Eczema** - Inflammatory skin condition
        - **Keratosis** - Rough, scaly patches
        - **Millia** - Small white bumps
        - **Rosacea** - Facial redness and bumps
        """)
        
        st.header("⚠️ Disclaimer")
        st.warning("""
        This tool is for educational purposes only. 
        Always consult a qualified dermatologist for 
        medical diagnosis and treatment.
        """)
        
        st.header("📊 Model Info")
        st.success("""
        - **Architecture**: Xception
        - **Accuracy**: 92%
        - **Training**: 1,657 images
        - **Classes**: 6 skin conditions
        """)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a skin lesion image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of a skin lesion for AI diagnosis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image_pil = Image.open(uploaded_file)
            st.image(image_pil, caption="Uploaded Image", use_container_width=True)
            
            # Convert to array and resize
            img_array = img_to_array(image_pil.convert('RGB').resize((299, 299)))
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction button
            if st.button("🔍 Diagnose", type="primary"):
                with st.spinner("Analyzing image..."):
                    results, top_idx, top_conf = predict_skin_disease(model, img_array)
                
                if results is not None:
                    st.success("Analysis complete!")
    
    with col2:
        st.header("📊 Diagnosis Results")
        
        if uploaded_file is not None and 'results' in locals():
            # Top prediction
            if results is not None:
                top_result = results[0]
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>🎯 Most Likely Diagnosis</h3>
                    <h2 style="color: #1f77b4;">{top_result['class']}</h2>
                    <h3 style="color: #ff7f0e;">{top_result['confidence']:.1f}% Confidence</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # All predictions
                st.subheader("📈 All Predictions")
                for i, result in enumerate(results):
                    color = "#1f77b4" if result['is_top'] else "#666666"
                    st.markdown(f"""
                    <div class="confidence-bar">
                        <strong style="color: {color};">{result['class']}</strong>
                        <div style="background-color: #ddd; height: 20px; border-radius: 10px; margin-top: 5px;">
                            <div style="background-color: {color}; height: 100%; width: {result['confidence']}%; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                                {result['confidence']:.1f}%
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Interpretation
                st.subheader("💡 Interpretation")
                if top_result['confidence'] > 70:
                    st.success("High confidence diagnosis. Consider consulting a dermatologist for confirmation.")
                elif top_result['confidence'] > 50:
                    st.warning("Moderate confidence. Further evaluation recommended.")
                else:
                    st.info("Low confidence. Please consult a dermatologist for proper diagnosis.")
        
        else:
            st.info("👆 Upload an image to get started with the diagnosis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🔬 Skin Disease AI | Built with Streamlit & TensorFlow</p>
        <p>⚠️ For educational purposes only | Consult a dermatologist for medical advice</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
