import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("cifar10_image_classifier.h5", compile=False)

# CIFAR-10 Classes
classes = [
    'Airplane',
    'Automobile',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck'
]

# Page Title
st.set_page_config(
    page_title="Image Classification",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ CIFAR-10 Image Classifier")
st.write("Upload an image and let the AI predict its class.")

# File Upload
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess Image
    img = image.resize((32, 32))
    img = np.array(img)

    # Handle grayscale images
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)

    # Remove alpha channel if present
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)

    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    # Probability Scores
    st.subheader("Class Probabilities")

    for i, cls in enumerate(classes):
        st.write(f"{cls}: {prediction[0][i]*100:.2f}%")