import torch
import torch.nn as nn

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


import torch.nn as nn
import torch

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionBlock, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch1x1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch3x3red),

            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, stride=1, padding = 1),
            nn.BatchNorm2d(ch3x3)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
        )
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)



class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.avg = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128)
        )
        self.fc1 = nn.Linear(2048, 1024) # 2048 = 4x4x128
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        out = self.avg(x)
        out = self.conv(out)
        out = out.view(out.size(0), -1) # Flatten this output
        
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out


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
        # ..x..x192
        self.a3 = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.b3 = InceptionBlock(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.b4 = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.c4 = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.d4 = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.e4 = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        
        self.a5 = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.b5 = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

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