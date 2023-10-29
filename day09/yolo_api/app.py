import streamlit as st
import requests
from object_detection import Object_Detection
import cv2
import numpy as np

# Load the YoLo model
OD = Object_Detection()

# Streamlit UI
st.title("Object Detection With YoLov8")

# Upload image
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Read the uploaded image with OpenCV to ensure it has a shape attribute
    img = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

    # Object Detection
    output = OD.image_object_detection(img)
    
    # Display the output image
    st.image(output, caption="Output Image", use_column_width=True)
    
