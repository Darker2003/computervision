import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm 

def evaluate_model(model, test_dataloader, device, class_names):
    # Set the model in evaluation mode
    model.eval()

    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for data in tqdm(test_dataloader, desc='eval'):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().tolist())
            all_true_labels.extend(labels.cpu().tolist())

    # Compute confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)

    # Plot confusion matrix using Seaborn
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
    return all_predictions, all_true_labels

def visualize(trainer):
    test_F1 = trainer.test_F1
    train_F1 = trainer.train_F1
    loss_test = trainer.loss_test
    loss_train = trainer.loss_train
    
    # Plotting the metrics
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(loss_train, label='Train Loss')
    plt.plot(loss_test, label='Test Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_F1, label='Train F1 Score')
    plt.plot(test_F1, label='Test F1 Score')
    plt.title('F1 Score over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.show()