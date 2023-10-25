from function.pytorch.utils import load_data, visualize_confusionmatrix
from function.pytorch.eval import eval_torch
import torch

from sklearn.metrics import classification_report

#Define config--------------------------------------------------------------------------------
class Config:
    def __init__(self):
        # Định nghĩa các thuộc tính cấu hình
        self.path_model = "models/torch/resnet18.pth"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Thiết bị sử dụng (cuda hoặc cpu)
        self.path_data = "dataset/test"  # Đường dẫn đến dữ liệu
        self.batch_size = 64 #Batch_size
        self.data = load_data(self.path_data)
        self.class_name = self.data.classes
        
        
cfg = Config()


#Resnet18 evaluating -----------------------------------------------------------------------
cfg.path_model = 'models/torch/resnet18.pth'
# Load model and evaluation
model = torch.load(cfg.path_model)
true_labels, predictions = eval_torch(model, cfg)
print(classification_report(true_labels, predictions, target_names=cfg.class_name))


visualize_confusionmatrix(
    true_labels=true_labels,
    predictions=predictions,
    classes_name=cfg.class_name
)


#VGG19 evaluating -----------------------------------------------------------------------
cfg.path_model = 'models/torch/vgg19.pth'
# Load model and evaluation
model = torch.load(cfg.path_model)
true_labels, predictions = eval_torch(model, cfg)
print(classification_report(true_labels, predictions, target_names=cfg.class_name))

visualize_confusionmatrix(
    true_labels=true_labels,
    predictions=predictions,
    classes_name=cfg.class_name
)
