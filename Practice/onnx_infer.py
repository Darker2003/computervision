import torch
import torch.onnx
import onnxruntime
import numpy as np
from convert_to_onnx import Onnx_convert

class Config():
    def __init__(self, model, path_onnx, input_samples,mode,type_model):
        self.model = model
        self.input_samples = input_samples
        self.mode = mode
        self.path_onnx = path_onnx + self.mode + type_model


cfg = Config(
    model = torch.load("models/torch/resnet18.pth"),
    mode ='float32bit',
    path_onnx = f'models/onnx/vgg19_',
    input_samples = torch.randn(16, 3, 224, 224),
    type_model=".onnx"
)

convert = Onnx_convert(cfg)

cfg = Config(
    model = torch.load("models/torch/resnet18.pth"),
    mode ='float16bit',
    path_onnx = f'models/onnx/vgg19_',
    input_samples = torch.randn(16, 3, 224, 224),
    type_model=".onnx"
)

convert = Onnx_convert(cfg)
batch_size = 4
input_data = torch.randn(batch_size, 3, 224, 224)

onnx_float16 = convert.load_onnx_model('models/onnx/vgg19_float16bit.onnx')
input_16bit = input_data.half()
input_numpy = input_16bit.numpy()
convert.onnx_infer(onnx_float16, input_numpy)


onnx_float32 = convert.load_onnx_model('models/onnx/vgg19_float32bit.onnx')
input_32bit = input_data.half()
input_numpy = input_32bit.numpy()
convert.onnx_infer(onnx_float32, input_numpy)

## Load model pytorch
model = torch.load("models/torch/resnet18.pth")
model.to('cuda')
with torch.no_grad():
    out = model(input_data.to('cuda'))
    
    
print(out)