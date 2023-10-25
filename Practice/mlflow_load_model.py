import mlflow
from dataset import DataPreparation
from models import *
from eval import *
import torch 
from sklearn.metrics import f1_score
mlflow.set_tracking_uri("http://127.0.0.1:5000")


import mlflow.pytorch

# Load model from a specific run_id---------------------------------------------------------
run_id = "5f8a4cb527184defbd7281890e6a3f32"
loaded_model = mlflow.pytorch.load_model(f"runs:/{run_id}/pytorch-model")


#Define config -----------------------------------------------------------------------------
class Config:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Thiết bị sử dụng (cuda hoặc cpu)
        self.train_dir = "dataset/train/"  # Đường dẫn đến dữ liệu huấn luyện
        self.test_dir = 'dataset/test/'
        self.num_classes = None
        self.class_names = None
        self.batch_size = 16
        self.max_epochs = 10
        self.patience = 3
        self.lr = 0.0001
        self.beta1 = 0.95
        self.beta2 = 0.993
        self.name_exp = 'classification'

# Loss
criterion = nn.CrossEntropyLoss()
# Config
cfg = Config()
data_preparation = DataPreparation(
    cfg = cfg
    )

data_preparation.create_data_loaders()
trainloader = data_preparation.trainloader
testloader = data_preparation.testloader
cfg.num_classes = data_preparation.num_classes
cfg.class_names = data_preparation.classes_name

#Prediction ---------------------------------------------------------------------------
all_predictions, all_true_labels = evaluate_model(
    model=loaded_model, 
    test_dataloader = testloader, 
    device = 'cuda', 
    class_names = data_preparation.classes_name
)

f1 = f1_score(all_true_labels, all_predictions, average='weighted')

print("f1_score", f1)