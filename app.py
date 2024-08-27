import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model('disease_disease_recognition.h5')

st.title('Crop Disease Prediction')
st.write("Upload an image to check if the plant is affected or not.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocessing the image
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, (224, 224))  # Resize image to match your model's input size
    img_array = tf.cast(img_array, tf.float32) / 255.0  # Normalize the image
    img_array = tf.expand_dims(img_array, axis=0)  # Model expects a batch of images

    st.write("Type")
    predictions = model.predict(img_array)
    
    # Assuming the model returns class probabilities, convert to class index
    class_index = np.argmax(predictions, axis=1)[0]

    # Display the classification based on the class index
    if class_index == 0:
        st.write("The plant is healthy.")
    elif class_index == 1:
        st.write("The plant is affected by Powdery Mildew.")
        st.write("To treat powdery mildew on plants, farmers can:Improve Air Circulation:** Space plants adequately and prune them to reduce humidity around leaves.Water Carefully:** Water at the base and avoid wetting leaves.Use Fungicides:** Apply organic options like neem oil or sulfur-based fungicides, or chemical fungicides if needed.Remove Infected Leaves:** Regularly prune and dispose of infected foliage to prevent spread.")
    elif class_index == 2:
        st.write("The plant is affected by Rust.")
        st.write("For treating rusty leaves, farmers can:Remove Infected Leaves:** Cut off and dispose of any infected leaves to prevent the fungus from spreading to healthy parts of the plant.Improve Air Circulation:** Space plants properly and prune them to reduce humidity and create airflow, which helps to prevent rust.Water at the Base:** Water plants at the soil level to keep leaves dry, as wet foliage promotes rust growth.Apply Fungicides:** Use sulfur-based or copper fungicides, or other appropriate treatments to control and prevent the spread of rust.Use Resistant Varieties:** Plant rust-resistant varieties to reduce the likelihood of rust infections.")
    else:
        st.write("Unknown classification.")
