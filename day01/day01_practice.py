import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from Train.train import Train



#Load Fashion MNIST dataset-------------------------------------------------------------------------------
transform = transforms.ToTensor()
trainset = torchvision.datasets.FashionMNIST(root='./data',train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)


#Class labels----------------------------------------------------------------------------------------------
class_labels = trainset.classes

#Basic statistics----------------------------------------------------------------------------------------
num_train_samples = len(trainset)
num_test_samples = len(testset)
num_classes = len(class_labels)

print("Number of training samples:", num_train_samples)
print("Number of test samples:", num_test_samples)
print("Number of classes:", num_classes)

#Print image size-----------------------------------------------------------------------------------------
sample_image, _ = next(iter(trainset))
print("Image size:", sample_image.shape)


# Batching data + convert data to tensor
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 200


TRAINLOADER = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
TESTLOADER = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)




model = Train(class_labels, TRAINLOADER, TESTLOADER, DEVICE, EPOCHS)

model.train()
model.plot_train()
model.evaluate()
