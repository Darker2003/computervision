import torch
from sklearn.metrics import f1_score
from tqdm import tqdm 

class Trainer:
    def __init__(self, model, train_loader, test_loader, cfg):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.patience = cfg.patience
        self.max_epochs = cfg.max_epochs
        self.device = cfg.device
        

        self.model_best_loss = None
        self.train_F1 = []
        self.test_F1 = []
        self.loss_train = []
        self.loss_test = []

    def train_one_epoch(self, optimizer, scheduler, criterion):
        self.model.train()
        total_loss = 0.0
        y_true = []
        y_pred = []

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(output, 1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
        if scheduler is not None:
            scheduler.step()
        f1 = f1_score(y_true, y_pred, average='micro')
        self.train_F1.append(f1)
        self.loss_train.append(total_loss / len(self.train_loader))
        
    def test_one_epoch(self, criterion):
        self.model.eval()
        total_loss = 0.0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()

                _, predicted = torch.max(output, 1)
                y_true.extend(target.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        f1 = f1_score(y_true, y_pred, average='micro')
        self.test_F1.append(f1)
        self.loss_test.append(total_loss / len(self.test_loader))

    def train(self, optimizer, scheduler, criterion):
        self.model.to(self.device)
        early_stopping_counter = 0
        best_loss = float('inf')

        for epoch in tqdm(range(self.max_epochs), desc='epochs'):
            self.train_one_epoch(optimizer, scheduler, criterion)
            self.test_one_epoch(criterion)

            print(f"Epoch {epoch+1}/{self.max_epochs} - Train Loss: {self.loss_train[-1]:.4f} - Test Loss: {self.loss_test[-1]:.4f} - Train F1: {self.train_F1[-1]:.4f} - Test F1: {self.test_F1[-1]:.4f}")

            if self.loss_test[-1] < best_loss:
                best_loss = self.loss_test[-1]
                early_stopping_counter = 0
                self.model_best_loss = self.model
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= self.patience:
                print("Early stopping triggered.")
                break

        print("Training finished.")
