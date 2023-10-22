import streamlit as st
import requests

# Streamlit UI
st.title("Image Classification with FastAPI")

# Upload image
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Make a request to the FastAPI endpoint for classification
    url = "http://localhost:8000/classify/"  # Đảm bảo rằng FastAPI đang chạy ở cổng 8000
    files = {"file": uploaded_image}
    response = requests.post(url, files=files)

    if response.status_code == 200:
        result = response.json()
        class_label = result["class_label"]
        st.success(f"Predicted Class: {class_label}")
    else:
        st.error("Failed to classify the image. Please try again.")

