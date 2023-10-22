import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

class DataPreparation:
    def __init__(self, cfg):
        self.train_dir = cfg.train_dir
        self.test_dir = cfg.test_dir
        self.batch_size = cfg.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = None
        self.classes_name = None
        self.classes2idx = None
        self.TRAINLOADER = None
        self.TESTLOADER = None

        self.prepare_transforms()
        self.load_data()
        self.create_data_loaders()

    def prepare_transforms(self):
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

    def load_data(self):
        train_data = torchvision.datasets.ImageFolder(root=self.train_dir, transform=self.train_transforms)
        test_data = torchvision.datasets.ImageFolder(root=self.test_dir, transform=self.test_transforms)

        self.num_classes = len(train_data.classes)
        self.classes_name = train_data.classes
        self.classes2idx = train_data.class_to_idx

        print(f"Số lượng lớp: {self.num_classes}")
        print(f"Tên lớp: {self.classes_name}")
        print(f"Ánh xạ từ tên lớp sang chỉ số: {self.classes2idx}")
        print("Number of train: ", len(train_data))
        print("Number of test: ", len(test_data))

        self.train_data = train_data
        self.test_data = test_data

    def create_data_loaders(self):
        self.trainloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.testloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

if __name__ == "__main__":
    train_dir = 'dataset/train'
    test_dir = 'dataset/test'
    data_preparation = DataPreparation(train_dir, test_dir)