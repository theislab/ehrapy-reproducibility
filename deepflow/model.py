import torch
import torch.nn as nn


class SimpleConvFactory(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        activation=nn.ReLU(),
    ):
        super(SimpleConvFactory, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class DualDownsampleFactory(nn.Module):
    def __init__(self, in_channels, ch_3x3):
        super(DualDownsampleFactory, self).__init__()
        # conv 3x3
        self.conv = SimpleConvFactory(
            in_channels, ch_3x3, kernel_size=3, stride=2, padding=1
        )
        # pool
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        conv = self.conv(x)
        pool = self.pool(x)
        # concat
        x = torch.cat([conv, pool], dim=1)
        return x


class DualFactory(nn.Module):
    def __init__(self, in_channels, ch_1x1, ch_3x3):
        super(DualFactory, self).__init__()
        # 1x1
        self.conv1x1 = SimpleConvFactory(in_channels, ch_1x1, kernel_size=1)
        # 3x3
        self.conv3x3 = SimpleConvFactory(in_channels, ch_3x3, kernel_size=3, padding=1)

    def forward(self, x):
        conv1x1 = self.conv1x1(x)
        conv3x3 = self.conv3x3(x)
        # concat
        x = torch.cat([conv1x1, conv3x3], dim=1)
        return x


class DeepFlow(nn.Module):
    def __init__(self):
        super(DeepFlow, self).__init__()
        self.conv1 = SimpleConvFactory(
            in_channels=3, out_channels=96, kernel_size=3, padding=1
        )
        self.in3a = DualFactory(in_channels=96, ch_1x1=32, ch_3x3=32)
        self.in3b = DualFactory(in_channels=64, ch_1x1=32, ch_3x3=48)
        self.in3c = DualDownsampleFactory(in_channels=80, ch_3x3=80)
        self.in4a = DualFactory(in_channels=160, ch_1x1=112, ch_3x3=48)
        self.in4b = DualFactory(in_channels=160, ch_1x1=96, ch_3x3=64)
        self.in4c = DualFactory(in_channels=160, ch_1x1=80, ch_3x3=80)
        self.in4d = DualFactory(in_channels=160, ch_1x1=48, ch_3x3=96)
        self.in4e = DualDownsampleFactory(in_channels=144, ch_3x3=96)
        self.in5a = DualFactory(in_channels=240, ch_1x1=176, ch_3x3=160)
        self.in5b = DualFactory(in_channels=336, ch_1x1=176, ch_3x3=160)
        self.in6a = DualDownsampleFactory(in_channels=336, ch_3x3=96)
        self.in6b = DualFactory(in_channels=432, ch_1x1=176, ch_3x3=160)
        self.in6c = DualFactory(in_channels=336, ch_1x1=176, ch_3x3=160)
        self.pool = nn.AvgPool2d(kernel_size=8)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=5376, out_features=5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.first_part(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
    
    def first_part(self, x):
        x = self.conv1(x)
        x = self.in3a(x)
        x = self.in3b(x)
        x = self.in3c(x)
        x = self.in4a(x)
        x = self.in4b(x)
        x = self.in4c(x)
        x = self.in4d(x)
        x = self.in4e(x)
        x = self.in5a(x)
        x = self.in5b(x)
        x = self.in6a(x)
        x = self.in6b(x)
        x = self.in6c(x)
        x = self.pool(x)
        x = self.flatten(x)

        return x
