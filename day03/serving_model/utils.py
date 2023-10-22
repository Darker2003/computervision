from torchvision import models, transforms
import torch

# Define the class labels for your model
class_labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'] 

# Define a function to preprocess the image
def preprocess_image(image, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(device)

def load_models(path, device):
    model = torch.load(path, map_location=torch.device('cpu'))
    model.eval()
    return model.to(device)