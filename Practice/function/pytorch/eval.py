from tqdm import tqdm 
from torch.utils.data import DataLoader
import torch

def eval_torch(model, cfg):
    testloader = DataLoader(cfg.data, batch_size=cfg.batch_size, shuffle=False)
    predictions = []
    true_labels = []
    device = cfg.device
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data in tqdm(testloader, desc='val'):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.tolist())
            true_labels.extend(labels.tolist())
    return true_labels, predictions

def eval_torch_16bit(model, cfg):
    testloader = DataLoader(cfg.data, batch_size=cfg.batch_size, shuffle=False)
    predictions = []
    true_labels = []
    device = cfg.device
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data in tqdm(testloader, desc='val'):
            inputs, labels = data
            inputs = inputs.half().to(device)  # Chuyển đổi inputs sang float16
            labels = labels.half().to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.tolist())
            true_labels.extend(labels.tolist())
    return true_labels, predictions