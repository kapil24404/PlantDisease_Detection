import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import json

# App configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# Constants
MODELS_DIR = 'saved_models'
IMG_SIZE = (224, 224)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    h1 {
        color: #2e7d32;
        text-align: center;
        margin-bottom: 30px;
        font-weight: 700;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-top: 20px;
        text-align: center;
    }
    .confidence-bar {
        height: 10px;
        border-radius: 5px;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_class_indices():
    try:
        class_indices = np.load(os.path.join(MODELS_DIR, 'class_indices.npy'), allow_pickle=True).item()
        # Invert the dictionary to map index to class name
        index_to_class = {v: k for k, v in class_indices.items()}
        return index_to_class
    except FileNotFoundError:
        st.warning("Class indices not found. Using dummy classes for demonstration.")
        return {0: "Healthy", 1: "Diseased_Type_A", 2: "Diseased_Type_B"}

@st.cache_resource
def load_selected_model(model_name):
    model_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error(f"Model file {model_path} not found. Please train the model first.")
        return None

def preprocess_image(image):
    # Resize and convert to RGB
    img = image.convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    # Normalize
    img_array = img_array / 255.0
    # Expand dimensions for batch
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    st.title("🌿 Plant Disease Detection System")
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.1em;'>A Transfer Learning approach using advanced CNNs and Transformers</p>", unsafe_allow_html=True)
    
    # Sidebar for model selection
    st.sidebar.header("⚙️ Model Configuration")
    available_models = [
        'Hybrid CNN-Transformer',
        'EfficientNet-B4',
        'ResNet50',
        'MobileNetV2',
        'DenseNet121',
        'VGG16',
        'Vision Transformer'
    ]
    selected_model_name = st.sidebar.selectbox("Select Model", available_models)
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Hybrid CNN-Transformer** is expected to yield the highest accuracy, "
        "while **MobileNetV2** is the fastest for edge deployments."
    )
    
    # Main UI
    st.write("### Upload a Leaf Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
        with col2:
            st.write("### Analysis Results")
            if st.button("Predict Disease"):
                with st.spinner(f"Analyzing with {selected_model_name}..."):
                    model = load_selected_model(selected_model_name)
                    index_to_class = load_class_indices()
                    
                    if model is not None:
                        processed_img = preprocess_image(image)
                        predictions = model.predict(processed_img)[0]
                        predicted_class_index = np.argmax(predictions)
                        confidence = predictions[predicted_class_index] * 100
                        
                        predicted_class_name = index_to_class.get(predicted_class_index, "Unknown")
                        
                        # Format the class name (replace underscores with spaces)
                        formatted_class_name = predicted_class_name.replace("_", " ").title()
                        
                        # Display Results
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2 style="color: {'#e53935' if 'healthy' not in predicted_class_name.lower() else '#43a047'}; margin-bottom: 5px;">
                                {formatted_class_name}
                            </h2>
                            <p style="font-size: 1.2em; color: #555;">Confidence: <strong>{confidence:.2f}%</strong></p>
                            <div style="width: 100%; background-color: #e0e0e0; border-radius: 5px; margin-top: 10px;">
                                <div class="confidence-bar" style="width: {confidence}%;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.success("Analysis Complete!")

if __name__ == "__main__":
    main()
