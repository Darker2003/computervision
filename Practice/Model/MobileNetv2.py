import torch
import torch.nn as nn
import copy

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = round(in_channels * expand_ratio)

        # Stride = 1 và channels không đổi
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []

        # Khi expand_ratio khác 1, thêm tầng Conv2d và tầng BatchNorm để tăng số kênh
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        # Thêm tầng Conv2d với kernel size 3 và stride có thể thay đổi (depthwise separable convolution)
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Tầng Conv2d cuối cùng để giảm số kênh trở lại
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        # Tạo một chuỗi các tầng đã xây dựng
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # Nếu điều kiện sử dụng kết nối shortcut đúng, thực hiện kết nối shortcut
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            # Nếu không, thực hiện các tầng theo chuỗi đã xây dựng
            return self.layers(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV2, self).__init__()
        self.curr_channels = 32
        self.features = [
            nn.Conv2d(3, self.curr_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.curr_channels),
            nn.ReLU6(inplace=True)
        ]
        ## 112x112x32
        inverted_residual_setting = [
            # t: expansion factor, c: output channels, n: number of layers, s: stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]
        for t, c, n, s in inverted_residual_setting:
            out_channels = c
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(self.curr_channels, out_channels, stride, t))
                self.curr_channels = out_channels


        self.features.append(nn.Conv2d(self.curr_channels, 1280, kernel_size=1, stride=1, padding=0, bias=False))
        self.features.append(nn.BatchNorm2d(1280))
        self.features.append(nn.ReLU6(inplace=True))

        self.features = nn.Sequential(*self.features)

        ## Avpool
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
                    
