import io
import torch
from fastapi import FastAPI, UploadFile
from PIL import Image
from torchvision import models, transforms
from utils import *

api = FastAPI()

# Load the pre-trained MobileNetV2 model
device = 'cuda'
model = load_models(
        path = 'models/mobinetv2.pth',
        device = device
    )

# Define the class labels for your model
class_labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']  # Thay thế bằng các nhãn của mô hình thực tế

# Define an API endpoint for image classification
@api.post("/classify/")
async def classify_image(file: UploadFile):
    # Read and preprocess the uploaded image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    input_tensor = preprocess_image(image, device)

    # Perform inference with the model
    with torch.no_grad():
        output = model(input_tensor)

    # Get the predicted class label
    _, predicted_class = output.max(1)
    class_label = class_labels[predicted_class]

    return {"class_label": class_label}
