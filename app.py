import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
import gemini  # Import the gemini.py file
from dotenv import load_dotenv # Import dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set the page configuration
st.set_page_config(
    page_title="Tree Species Identification",
    page_icon="🌳",
    layout="wide",
)

# --- Model Loading ---
@st.cache_resource
def load_all_models():
    """Load all the h5 models from the project directory."""
    models = {}
    models["Basic CNN"] = load_model('basic_cnn_tree_species.h5')
    models["Main Model (EfficientNet)"] = load_model('tree_species_model.h5')
    models["Improved CNN"] = load_model('improved_cnn_model.h5')
    return models

models = load_all_models()

# --- Class Names ---
class_names = [
    'other', 'pipal', 'bamboo', 'khajur', 'motichanoti', 'mango', 'pilikaren',
    'jamun', 'simlo', 'cactus', 'saptaparni', 'kanchan', 'bili', 'neem',
    'vad', 'sitafal', 'champa', 'asopalav', 'sonmahor', 'gulmohor',
    'shirish', 'garmalo', 'amla', 'kesudo', 'sugarcane', 'coconut',
    'banyan', 'gunda', 'babul', 'nilgiri'
]

# --- Image Preprocessing ---
def preprocess_image(image):
    """Preprocesses the uploaded image to be model-ready."""
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Streamlit App UI ---
st.title("🌳 Tree Species Identification with AI Insights")
st.write("Upload an image of a tree, and the model will predict its species and provide details.")

# --- Model Prediction Section ---
st.header("Model Prediction")
model_choice = st.selectbox("Choose a model:", list(models.keys()))
model = models[model_choice]

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)

    with st.spinner(f"Predicting using {model_choice}..."):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        confidence = np.max(prediction)

    st.success(f"**Prediction:** {predicted_class_name}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # --- AI Insights Section (now below the prediction) ---
    st.header("AI-Powered Insights")
    # Check if the API key exists in the environment
    if not os.getenv("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY not found. Please create a .env file with your key.")
    # Check if a prediction was made and it's not 'other'
    elif predicted_class_name and predicted_class_name != 'other':
        with st.spinner(f"Getting details for {predicted_class_name} from Gemini..."):
            response = gemini.get_gemini_response(predicted_class_name)
            st.markdown(response)
    elif predicted_class_name == 'other':
        st.info("The model predicted 'other', so no specific details can be retrieved.")
