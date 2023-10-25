import os
from configs import CLASSNAME
import torch
from torchvision import transforms
from PIL import Image
import onnxruntime as ort
import warnings
warnings.filterwarnings('ignore')


IMAGE_SIZE = 224
DEVICE = 'cuda'
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def infer_onnx(onnx_session, image_paths):
    # Tạo batch tensor cho tất cả ảnh
    batch_images = torch.zeros(len(image_paths), 3, IMAGE_SIZE, IMAGE_SIZE)  # Batch size x Channels x Height x Width

    # Lặp qua từng ảnh và tiền xử lý
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path)
        image = transform(image)
        batch_images[i] = image
    
    # Chuyển batch tensor thành numpy array
    batch_images_np = batch_images.cpu().numpy()

    # Thực hiện suy luận với ONNX Runtime
    onnx_input = {onnx_session.get_inputs()[0].name: batch_images_np}
    onnx_output = onnx_session.run(None, onnx_input)

    for i, output in enumerate(onnx_output[0]):
        predicted_class = torch.argmax(torch.tensor(output)).item()
        image_filename = os.path.basename(image_paths[i])
        class_name = CLASSNAME[predicted_class]
        class_score = torch.softmax(torch.tensor(output), dim=0)[predicted_class].item()
        
        print(f"Image {image_filename}: {class_name} (Score: {class_score:.4f})")

if __name__ == '__main__':
    from glob import glob
    from PIL import Image
    import time
    all_images = glob("sample/*")
    onnx_model_path = "resnet18.onnx"
    #providers = ['CPUExecutionProvider', 'CUDAExecutionProvider']
    providers = ['CUDAExecutionProvider']
    onnx_session = ort.InferenceSession(onnx_model_path, providers=providers)
    print(onnx_session.get_providers())
    start = time.time()    
    infer_onnx(onnx_session, all_images)
    print(time.time() - start)
    