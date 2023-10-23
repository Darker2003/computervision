import torch
import torch.nn as nn
import copy

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionBlock, self).__init__()

        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch1x1)
        )

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch3x3red),

            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch3x3)
        )

        # Branch 3: 1x1 convolution followed by 5x5 convolution
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),

            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5)
        )

        # Branch 4: MaxPooling followed by 1x1 convolution
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj)
        )

    def forward(self, x):
        # Combine outputs from all branches using concatenation along the channel dimension
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        # Khởi tạo các lớp cho mạng InceptionAux
        self.avg = nn.AvgPool2d(kernel_size=5, stride=3) # Lớp AvgPool2d với cửa sổ 5x5 và bước nhảy 3
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1), # Convolutional layer với kernel size 1x1 và 128 filters
            nn.BatchNorm2d(128) # Batch normalization với 128 channels
        )
        self.fc1 = nn.Linear(2048, 1024) # Fully connected layer với 2048 input features và 1024 output features
        self.fc2 = nn.Linear(1024, num_classes) # Fully connected layer với 1024 input features và num_classes output features
        self.dropout = nn.Dropout(0.7) # Dropout với tỷ lệ dropout 0.7 (70%)

    def forward(self, x):
        # Hàm forward cho mạng InceptionAux
        out = self.avg(x) # Áp dụng phép tổng hợp trung bình lên đầu vào
        out = self.conv(out) # Áp dụng lớp Convolution và BatchNorm
        out = out.view(out.size(0), -1) # Flattening: Chuyển đổi tensor thành vector
        out = self.fc1(out) # Áp dụng fully connected layer 1
        out = self.fc2(out) # Áp dụng fully connected layer 2 để đưa ra output
        out = self.dropout(out) # Áp dụng dropout
        return out
    
    
## Define mạng GoogleNet
class GoogleNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GoogleNet, self).__init__()

        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.a3 = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.b3 = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        # TODO: Thay đổi các khối Inception và các lớp Maxpooling trong mạng
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.a4 = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.b4 = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.c4 = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.d4 = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.e4 = InceptionBlock(528, 256, 160, 320, 32, 128, 128)

        self.a5 = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.b5 = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        # ............

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Muốn output là 1x1xN
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(1024, num_classes)
        self.aux1 = InceptionAux(512, num_classes)
        self.aux2 = InceptionAux(528, num_classes)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)

        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out_1 = self.aux1(out)

        out = self.b4(out)

        out = self.c4(out)
        out = self.d4(out)
        out_2 = self.aux2(out)
        out = self.e4(out)

        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out_final = self.linear(out)

        return [out_1, out_2, out_final]
    
    

