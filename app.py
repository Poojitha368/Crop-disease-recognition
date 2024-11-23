import streamlit as st
import gdown
import numpy as np
import tensorflow as tf
from PIL import Image
import time as t
import os

# Function to download the model file using gdown
def download_model():
    file_path = 'crop_disease_recognition.h5'
    
    # Direct download link
    url = 'https://drive.google.com/uc?id=1gckT57oxPVB7JMymZ_kdGRt4HMck4qCp'

    
    # Check if the file already exists
    if not os.path.exists(file_path):
        st.info("Downloading model file. This might take a while...")
        gdown.download(url, file_path, quiet=False)
    
    # Load and return the model
    return tf.keras.models.load_model(file_path)

# Load the model
model = download_model()

st.title('🌿 Crop Disease Prediction')
st.info("Upload an image to check if the plant is affected or not.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Preprocessing the image
        img_array = np.array(image)
        img_array = tf.image.resize(img_array, (224, 224))
        img_array = tf.cast(img_array, tf.float32) / 255.0  # Normalize
        img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension

        # Spinner while classifying
        with st.spinner("Classifying..."):
            t.sleep(3)
            predictions = model.predict(img_array)

        # Extract class index
        class_index = np.argmax(predictions, axis=1)[0]

        # Display classification results
        st.subheader("Prediction Results:")
        if class_index == 0:
            st.success("The plant is **healthy**.")
        elif class_index == 1:
            st.error("The plant is affected by **Powdery Mildew**.")
            st.write("### Suggested Treatment:")
            st.markdown("""
            - **Improve Air Circulation**: Space plants adequately and prune them to reduce humidity around leaves.
            - **Water Carefully**: Water at the base and avoid wetting leaves.
            - **Use Fungicides**: Apply organic options like neem oil or sulfur-based fungicides, or chemical fungicides if needed.
            - **Remove Infected Leaves**: Regularly prune and dispose of infected foliage to prevent spread.
            """)
        elif class_index == 2:
            st.error("The plant is affected by **Rust**.")
            st.write("### Suggested Treatment:")
            st.markdown("""
            - **Remove Infected Leaves**: Cut off and dispose of infected leaves to prevent fungus spread.
            - **Improve Air Circulation**: Space plants properly and prune them to reduce humidity.
            - **Water at the Base**: Water plants at the soil level to keep leaves dry.
            - **Apply Fungicides**: Use sulfur-based or copper fungicides.
            - **Use Resistant Varieties**: Plant rust-resistant varieties.
            """)
        else:
            st.warning("Unknown classification. Please check your input.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload an image to proceed.")

