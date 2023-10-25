import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.optim as optim
import optuna
from train import Trainer
from dataset_prepare import DataPreparation
from models_frozen import *
from eval import evaluate_model
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

#Define config--------------------------------------------------------------------------------
class Config:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Thiết bị sử dụng (cuda hoặc cpu)
        self.train_dir = "dataset/train/"  # Đường dẫn đến dữ liệu huấn luyện
        self.test_dir = 'dataset/test/'
        self.num_classes = None
        self.class_names = None
        self.batch_size = 64
        self.max_epochs = 5
        self.patience = 3
        self.lr = 0.0001
        self.beta1 = 0.95
        self.beta2 = 0.993

cfg = Config()
data_preparation = DataPreparation(
    cfg = cfg
    )
        
# Loss-----------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()

#----------------------------------------------------------------------------------
def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 6, 8, 16, 32])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    beta1 = trial.suggest_float("beta1", 0.9, 0.99, step=0.01)
    beta2 = trial.suggest_float("beta2", 0.99, 0.999, step=0.001)
    
    # Update configs
    cfg.beta1 = beta1
    cfg.beta2 = beta2
    cfg.batch_size = batch_size
    cfg.lr = learning_rate
    

    # Create dataloader
    data_preparation.create_data_loaders()
    trainloader = data_preparation.trainloader
    testloader = data_preparation.testloader
    cfg.num_classes = data_preparation.num_classes
    cfg.class_names = data_preparation.classes_name
    
    # Create Model
    model = mobilenetv3_frozen(cfg.num_classes)

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
    return f1

#Tuning -----------------------------------------------------------------
study_name = "Image Classificaion with Transfer Leanring"
study = optuna.create_study(
    study_name=study_name,
    direction="maximize"
    )
study.optimize(objective, n_trials=10)

#Result ------------------------------------------------------------------
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial
print("Value: ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
    
    
#Train best model ----------------------------------------------------------
print("trial.params:",trial.params)

# Update configs
cfg.beta1 = trial.params['beta1']
cfg.beta2 = trial.params['beta2']
cfg.batch_size = trial.params['batch_size']
cfg.lr = trial.params['learning_rate']


# Create dataloader
data_preparation.create_data_loaders()
trainloader = data_preparation.trainloader
testloader = data_preparation.testloader
cfg.num_classes = data_preparation.num_classes
cfg.class_names = data_preparation.classes_name

# Create Model
model = mobilenetv3_frozen(cfg.num_classes)

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

print("f1_score:", f1)

#Save model  --------------------------------------------------------------------
torch.save(best_model, 'best_model_tuning.pth')
