import torch
import torch.onnx
import onnxruntime
import numpy as np

class Onnx_convert():
    def __init__(self, cfg):
        self.model = cfg.model
        self.input_samples = cfg.input_samples
        self.path_onnx = cfg.path_onnx
        self.mode = cfg.mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Thiết bị sử dụng (cuda hoặc cpu)
    def convert_pytorch2onnx(self):
        if self.mode == 'float16bit':
            print("float16bit")
            self.model.float()
            self.model.half()  # Chuyển mô hình sang float16
            input_samples = input_samples.half()
        self.model.to(self.device)
        self.model.eval()
        input_samples = input_samples.to(self.device)
        torch.onnx.export(
            self.model, # Model load bằng pytorch
            input_samples, # input với kích thước mong muốn
            self.path_onnx, # path_onnx bạn lưu
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

    def load_onnx_model(self, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']):
        # Create an ONNX Runtime inference session for the ONNX model
        ort_session = onnxruntime.InferenceSession(
            self.path_onnx,
            providers=providers
            )
        return ort_session

    def onnx_infer(self,ort_session, input_data):
        ort_inputs = {ort_session.get_inputs()[0].name: input_data}
        ort_output = ort_session.run(None, ort_inputs)
        return ort_output


