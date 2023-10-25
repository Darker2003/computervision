from torchvision import transforms
import torchvision
import torch
from PIL import Image
from configs import *
import os
import warnings
warnings.filterwarnings('ignore')

IMAGE_SIZE = 224
DEVICE = 'cuda'
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def infer_pytorch(model, image_paths):
    # Tạo batch tensor cho tất cả ảnh
    batch_images = torch.zeros(len(image_paths), 3, IMAGE_SIZE, IMAGE_SIZE)  # Batch size x Channels x Height x Width

    # Lặp qua từng ảnh và tiền xử lý
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path)
        image = transform(image)
        batch_images[i] = image
    
    batch_images = batch_images.to(DEVICE)
    with torch.no_grad():
        output = model(batch_images) #Batch size x 1000
        
    _, predicted_classes = torch.max(output, 1) # batch size x 1 
    predicted_scores = torch.nn.functional.softmax(output, dim=1) # batch size x 1

    for i, (predicted_class, scores) in enumerate(zip(predicted_classes, predicted_scores)):
        predicted_class = predicted_class.item()
        image_filename = os.path.basename(image_paths[i])
        class_name = CLASSNAME[predicted_class]
        class_score = scores[predicted_class].item()
        
        print(f"Image {image_filename}: {class_name} (Score: {class_score:.4f})")


if __name__ == '__main__':
    from glob import glob
    from PIL import Image
    all_images = glob("sample/*")
    model = torchvision.models.resnet18(pretrained=True).to(DEVICE)
    model.eval()
    infer_pytorch(model, all_images)


    

