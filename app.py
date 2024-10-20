import streamlit as st
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np


model = keras.models.load_model('best.keras')

def preprocess_image(image):
    # Resize the image to (256, 256)
    image = cv2.resize(image, (256, 256))
    # Expand dimensions to match the model's expected input shape
    image = np.expand_dims(image, axis=0)
    # Normalize the image
    image = image / 255.0
    return image

def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction[0][0]

st.title('Cat or Dog Predictor')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
   
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    
    st.image(image, channels="BGR", caption='Uploaded Image', use_column_width=True)
    
    
    if st.button('Predict'):
        prediction = predict(image)
        if prediction > 0.5:
            st.write(f"This is a dog with {prediction:.2%} confidence.")
        else:
            st.write(f"This is a cat with {(1-prediction):.2%} confidence.")