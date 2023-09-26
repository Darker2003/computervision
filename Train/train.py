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

from Model.CustomNN import CustomNN
from sklearn.metrics import confusion_matrix, f1_score, classification_report



class Train():
    def __init__(self,class_labels,  TRAINLOADER, TESTLOADER, DEVICE, EPOCHS = 200):
        
        self.class_labels = class_labels

        self.DEVICE = DEVICE
        self.EPOCHS = EPOCHS

        #Creat data loader
        self.TRAINLOADER = TRAINLOADER
        self.TESTLOADER = TESTLOADER
        # Init Model
        self.input_size = 28*28
        hidden_layer = [256, 128]
        output_size = 10
        dropout_p = 0.2
        activation = nn.ReLU()
        self.model = CustomNN(self.input_size, hidden_layer, output_size, dropout_p, activation)

        #Model to CUDA
        self.model.to(self.DEVICE)
        
        
        
        

    def train(self):

        #Loss
        criterion = nn.CrossEntropyLoss()

        #Optimizer and Scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=0.001) #Choose the optimizer
        scheduler = CosineAnnealingLR(optimizer, T_max=5) # Choose the scheduler
        # scheduler = None


        self.loss_train = []
        self.loss_test = []
        self.f1_train = []
        self.f1_test = []

        for epoch in range(self.EPOCHS):
            self.model.train()  # Set the self.model to training mode
            running_loss = 0.0
            predictions_train = []
            true_labels_train = []
            for i, data in enumerate(self.TRAINLOADER, 0):
                inputs, labels = data
                inputs = inputs.to(self.DEVICE)
                labels = labels.to(self.DEVICE)
                optimizer.zero_grad()
                outputs =  self.model(inputs.view(-1, self.input_size))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                predictions_train.extend(predicted.tolist())
                true_labels_train.extend(labels.tolist())
            if scheduler is not None:
                scheduler.step()  # Update learning rate with scheduler
            train_loss = running_loss / len(self.TRAINLOADER)
            train_f1 = f1_score(true_labels_train, predictions_train, average='weighted')

            self.loss_train.append(train_loss)
            self.f1_train.append(train_f1)

            # Evaluation on test set
            self.model.eval()  # Set the self.model to evaluation mode
            test_loss_val = 0.0
            predictions = []
            true_labels = []

            with torch.no_grad():
                for data in self.TESTLOADER:
                    inputs, labels = data
                    inputs = inputs.to(self.DEVICE)
                    labels = labels.to(self.DEVICE)
                    outputs = self.model(inputs.view(-1, self.input_size))
                    #outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss_val += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    predictions.extend(predicted.tolist())
                    true_labels.extend(labels.tolist())

            test_loss_val /= len(self.TESTLOADER)
            test_f1_val = f1_score(true_labels, predictions, average='weighted')

            self.loss_test.append(test_loss_val)
            self.f1_test.append(test_f1_val)
            print(f'Epoch [{epoch + 1}/{self.EPOCHS}]  - Train Loss: {train_loss:.4f} - Train F1: {train_f1:.4f} - Test Loss: {test_loss_val:.4f} - Test F1: {test_f1_val:.4f}')
        torch.save(self.model.state_dict(), "model.pth")
        print('Finished Training')
        
    
    def plot_train(self):
        #plotting the metrics
        plt.figure(figsize=(12, 6))

        plt.subplot(1,2,1)
        plt.plot(self.loss_train, label="Train Loss")
        plt.plot(self.loss_test, label="Test Loss")
        plt.title("Loss over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(self.f1_train, label="Train F1 Score")
        plt.plot(self.f1_test, label="Test F1 Score")
        plt.title("F1 Score over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("F1 Score")
        plt.legend()

        plt.tight_layout()
        plt.show()
        
        
    #Evaluate the model
    def evaluate(self):
        self.model.eval()
        all_predictions = []
        all_true_labels = []

        with torch.no_grad():
            for data in self.TESTLOADER:
                inputs, labels = data
                inputs, labels = inputs.to(self.DEVICE), labels.to(self.DEVICE)
                outputs = self.model(inputs.view(-1, self.input_size))
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().tolist())
                all_true_labels.extend(labels.cpu().tolist())
                
            #Compute confusion matrix
            cm = confusion_matrix(all_true_labels, all_predictions)
            
            #Plot confusion matrix using Seaborn
            plt.figure(figsize=(10,7))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.class_labels, yticklabels=self.class_labels)
            plt.xlabel("Predictec labels")
            plt.ylabel("True labels")
            plt.title("Confusion Matrix")
            plt.show()
            
        f1 = f1_score(all_true_labels, all_predictions, average="weighted")
        print("Weighted F1 Score:", f1)
        print(classification_report(all_true_labels, all_predictions, target_names=self.class_labels))