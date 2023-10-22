import io
import streamlit as st
from PIL import Image
from utils import *

# Load the pre-trained MobileNetV2 model
device = 'cuda'

# Load the pre-trained MobileNetV2 model ( chỉ một lần)
@st.cache_resource()
def load_model():
    print("Here")
    model = load_models(
        path = 'models/mobinetv2.pth',
        device = device
    )
    return model 
# Load model
model = load_model()

# Streamlit UI
st.title("Image Classification with Streamlit")

# Upload image
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Perform inference with the model
    image = Image.open(uploaded_image)
    input_tensor = preprocess_image(image, device)

    with torch.no_grad():
        output = model(input_tensor)

    # Get the predicted class label
    _, predicted_class = output.max(1)
    class_label = class_labels[predicted_class]

    st.success(f"Predicted Class: {class_label}")
