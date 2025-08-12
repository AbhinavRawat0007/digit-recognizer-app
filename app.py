# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

# --- MODEL LOADING ---

@st.cache_resource
def load_model():
    """Load the pre-trained MNIST model."""
    try:
        model = tf.keras.models.load_model('mnist_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please make sure the 'mnist_model.h5' file is in the same directory and you've run 'train_model.py' first.")
        return None

model = load_model()

# --- UI SETUP ---

st.set_page_config(page_title="Handwritten Digit Recognizer", page_icon="✍️")

st.title("✍️ Handwritten Digit Recognizer")
st.markdown("Draw a digit from 0 to 9 on the canvas below and click 'Predict' to see the model's guess!")

# --- DRAWABLE CANVAS ---

# Create a two-column layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Drawing Canvas")
    # Parameters for the canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=20, # Pen size
        stroke_color="#FFFFFF", # Pen color (white)
        background_color="#000000", # Canvas background (black)
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

# --- PREDICTION LOGIC ---

with col2:
    st.subheader("Prediction")
    if st.button("Predict"):
        if canvas_result.image_data is not None and model is not None:
            # 1. Get image from canvas
            img = canvas_result.image_data.astype('uint8')

            # 2. Process the image for the model
            # Convert to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            # Resize to 28x28
            img_resized = cv2.resize(img_gray, (28, 28))
            # Reshape for the model and normalize
            img_reshaped = img_resized.reshape(1, 28, 28, 1)
            img_normalized = img_reshaped / 255.0

            # 3. Make prediction
            prediction = model.predict(img_normalized)
            predicted_digit = np.argmax(prediction)

            st.success(f"I predict this digit is a: **{predicted_digit}**")
            st.bar_chart(prediction[0])
        else:
            st.warning("Please draw a digit first or check model loading status.")


st.markdown("---")
st.markdown("Built with 🧠 **TensorFlow** and 🌐 **Streamlit**.")