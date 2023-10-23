#mlflow server --port 5000
#sudo kill -9 $(sudo lsof -t -i:5000)
import torch
from train import Trainer
from dataset import DataPreparation
from models import *
from eval import evaluate_model
from sklearn.metrics import f1_score
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")


#Define config -----------------------------------------------------------------------------
class Config:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Thiết bị sử dụng (cuda hoặc cpu)
        self.train_dir = "../../dataset/train/"  # Đường dẫn đến dữ liệu huấn luyện
        self.test_dir = '../../dataset/test/'
        self.num_classes = None
        self.class_names = None
        self.batch_size = 16
        self.max_epochs = 10
        self.patience = 3
        self.lr = 0.001
        self.beta1 = 0.95
        self.beta2 = 0.993
        self.model_name = 'mobilenetv3'
        self.name_exp = 'classification'
        self.mlflow = mlflow
        
# Config
cfg = Config()
data_preparation = DataPreparation(
    cfg = cfg
    )

cfg.mlflow.start_run(run_name=f"{cfg.name_exp}")


# Loss-------------------------------------------------
criterion = nn.CrossEntropyLoss()

#Training -----------------------------------------------------------------------------

mlflow.log_param("batch_size", cfg.batch_size)
mlflow.log_param("learning_rate", cfg.lr)
mlflow.log_param("beta1", cfg.beta1)
mlflow.log_param("beta2", cfg.beta2)
mlflow.log_param("type_model", cfg.model_name) 


# Create dataloader
data_preparation.create_data_loaders()
trainloader = data_preparation.trainloader
testloader = data_preparation.testloader
cfg.num_classes = data_preparation.num_classes
cfg.class_names = data_preparation.classes_name

# Create Model
if cfg.model_name == 'resnet':
    model = resnet18_frozen(cfg.num_classes)
elif cfg.model_name == 'mobilenetv3':
    model = mobilenetv3_frozen(cfg.num_classes)
elif cfg.model_name == 'efficientnet':
    model = efficientnet_frozen(cfg.num_classes)


# Create Trainer
trainer = Trainer(
    model = model, 
    train_loader=trainloader, 
    test_loader=testloader,
    cfg=cfg
)

optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

# Training
trainer.train(
    optimizer=optimizer, 
    scheduler=None,
    criterion=criterion
)
best_model = trainer.model_best_loss
all_predictions, all_true_labels = evaluate_model(
    model=best_model, 
    test_dataloader = testloader, 
    device = cfg.device, 
    class_names = data_preparation.classes_name
)

f1 = f1_score(all_true_labels, all_predictions, average='weighted')
mlflow.log_metric("f1_score", f1)
mlflow.pytorch.log_model(best_model, artifact_path="pytorch-model")