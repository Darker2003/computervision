import os
import numpy as np
import torch
import onnxruntime as ort
from PIL import Image
from configs import CLASSNAME
import warnings
warnings.filterwarnings('ignore')

IMAGE_SIZE = 224
def preprocess_images(image_paths):
    images = []

    for image_path in image_paths:
        image = Image.open(image_path)
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image = np.array(image) / 255.0  # Chuẩn hóa giá trị pixel về [0, 1]
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # Chuẩn hóa theo mean và std
        # H x W x C
        # 0   1   2
        # 2   0   1
        image = np.transpose(image, (2, 0, 1))  # Chuyển đổi thành định dạng (C, H, W)
        images.append(image)

    return np.array(images, dtype=np.float32)


def infer_onnx(onnx_session, image_paths):
    images = preprocess_images(image_paths)

    # Thực hiện suy luận với ONNX Runtime
    onnx_input = {onnx_session.get_inputs()[0].name: images}
    onnx_output = onnx_session.run(None, onnx_input)

    for i, output in enumerate(onnx_output[0]):
        # Sử dụng softmax để tính toán xác suất
        softmax_output = np.exp(output) / np.sum(np.exp(output), axis=0)
        predicted_class = np.argmax(softmax_output)
        class_name = CLASSNAME[predicted_class]
        class_score = softmax_output[predicted_class]  # Lấy xác suất của lớp dự đoán

        print(f"Image {os.path.basename(image_paths[i])}: {class_name} (Score: {class_score:.4f})")


if __name__ == '__main__':
    from glob import glob
    import time
    all_images = glob("sample/*")
    onnx_model_path = "resnet18.onnx"
    # Load mô hình ONNX bằng ONNX Runtime và thực hiện suy luận
    #providers = ['CPUExecutionProvider', 'CUDAExecutionProvider']
    providers = ['CUDAExecutionProvider']
    onnx_session = ort.InferenceSession(onnx_model_path, providers=providers)
    print(onnx_session.get_providers())
    start = time.time()   
    infer_onnx(onnx_session, all_images)
    print(time.time() - start)
