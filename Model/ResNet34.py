import torch.nn as nn


# Hàm tạo lớp Conv2d với kernel size 3x3, stride có thể tùy chỉnh
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1  # Hệ số mở rộng, dùng trong các khối ResNet khác
    def __init__(self, in_chanels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # Lớp Convolution đầu tiên
        self.conv1 = conv3x3(in_chanels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)  # Lớp BatchNormalization sau Conv1
        self.relu = nn.ReLU(inplace=True)  # Hàm kích hoạt ReLU
        self.conv2 = conv3x3(out_channels, out_channels)  # Lớp Convolution thứ hai
        self.bn2 = nn.BatchNorm2d(out_channels)  # Lớp BatchNormalization sau Conv2
        self.downsample = downsample  # Đường ngắn để chiếu véc-tơ đầu vào nếu cần
        self.stride = stride  # Bước nhảy của Convolution

    def forward(self, x):
        identity = x

        out = self.conv1(x)  # Áp dụng Convolution đầu tiên
        out = self.bn1(out)  # Áp dụng BatchNormalization
        out = self.relu(out)  # Áp dụng hàm kích hoạt ReLU

        out = self.conv2(out)  # Áp dụng Convolution thứ hai
        out = self.bn2(out)  # Áp dụng BatchNormalization

        if self.downsample is not None:  # Nếu có đường ngắn
            identity = self.downsample(x)  # Áp dụng đường ngắn để chiếu véc-tơ đầu vào

        out += identity  # Cộng đầu ra của Conv2 với đầu vào ban đầu (residual connection)
        out = self.relu(out)  # Áp dụng hàm kích hoạt ReLU lần nữa

        return out  # Trả về đầu ra của khối BasicBlock

class ResNet(nn.Module):
    def __init__(self, block, num_layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.curr_chanels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_layers[0])
        self.layer2 = self._make_layer(block, 128, num_layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

    # block chính là khối BasicBlock bạn define
    def _make_layer(self, block, in_chanels, num_layers, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.curr_chanels, in_chanels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(in_chanels * block.expansion),
            )

        layers = []
        layers.append(block(self.curr_chanels, in_chanels, stride, downsample))
        #Update curr_chanels
        self.curr_chanels = in_chanels * block.expansion

        # Thêm các block còn lại
        for _ in range(1, num_layers):
            layers.append(block(self.curr_chanels, in_chanels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet34(**kwargs):
    num_layers = [3, 3, 6, 3]
    block = BasicBlock
    model = ResNet(BasicBlock, num_layers, **kwargs)
    return model




## Kiểm tra tính hoạt động của mạng custom này
import torch
model = resnet34()
random_image = torch.rand(1, 3, 224, 224)
model(random_image).shape
print(model)