import torch
import torchvision.models as models
import torchvision.transforms as transforms
import onnx
from PIL import Image
import onnxruntime as ort

# Step 1: Tạo và lưu mô hình ResNet-50 từ PyTorch sang ONNX
model = models.resnet18(pretrained=True)
dummy_input = torch.randn(1, 3, 224, 224)  # Đầu vào giả lập
onnx_model_path = "resnet18.onnx"
torch.onnx.export(
    model, # Model load bằng pytorch
    dummy_input, # input với kích thước mong muốn
    onnx_model_path, # path_onnx bạn lưu
    verbose=False,  # Hiển thị thông báo trong quá trình chuyển đổi
    opset_version=12, # Đọc xem mỗi version sẽ có hỗ trợ onnx cho lớp nào 
    do_constant_folding=True , # Dùng Constant-folding optimizer giúp cải thiện tốc độ và dung lượng
    input_names = ['images'],   # the model's input names
    output_names = ['output'], # the model's output names
    dynamic_axes={
            'images' : {0 : 'batch_size',},  
            'output' : {0 : 'batch_size'}
        }
    )

# Step 2: Load mô hình ONNX bằng ONNX Runtime và thực hiện suy luận
providers = ['CPUExecutionProvider', 'CUDAExecutionProvider']
onnx_session = ort.InferenceSession(onnx_model_path, providers=providers)

# Step 3: Chuẩn bị ảnh đầu vào và thực hiện suy luận
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load và tiền xử lý ảnh
image_path = "sample/a.png"
image = Image.open(image_path)
image = transform(image).unsqueeze(0)  # Thêm batch dimension

# Thực hiện suy luận bằng ONNX Runtime
onnx_input = {onnx_session.get_inputs()[0].name: image.numpy()}
onnx_output = onnx_session.run(None, onnx_input)[0]

# Step 4: Thực hiện suy luận bằng PyTorch
model.eval()
with torch.no_grad():
    pytorch_output = model(image)

# Step 5: So sánh kết quả
torch.testing.assert_close(pytorch_output, torch.tensor(onnx_output), rtol=1e-10, atol=1e-7) # sai số 0.1%, sai số tuyệt đối 0.00001
