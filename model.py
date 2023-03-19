import torch
from config import CONFIG
from torch import nn
"""
    nn.Conv:
    nn.BatchNorm2d:
    nn.LeakyReLU:
    nn.Sequential:
    nn.maxPool2d:
    nn.AvgPool2d:
    nn.Flatten:
"""
class ConvBlock(nn.Module):
    def __init__(self, in_channels, negative_slope) -> None:
        super().__init__()
        """ 没有转换通道数 """
        self.conv1d = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1,1))
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(in_channels)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope)
        )

        self.leaky_relu = nn.LeakyReLU(negative_slope)
    
    def forward(self, x):
        road1_out = self.conv_3(self.conv_2(x))
        road2_out = self.conv1d(x)
        return self.leaky_relu(road1_out + road2_out)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, negative_slope) -> None:
        super().__init__()
        self.improved_residual_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope),
            nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(in_channels)
        )
        self.leaky_relu = nn.LeakyReLU(negative_slope)
    
    def forward(self, x):
        # print("stop here, shape out")
        y = self.improved_residual_block(x)
        # print(x.shape, y.shape)

        return self.leaky_relu(x + y)

""" 
    INPUT: (batch_size, 3, 28, 28)
    OUTPUT: (batch_size, num_class)
"""
class Net(nn.Module):
    def __init__(self, in_channels, negative_slope) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3,3)),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope)
        )
        self.pool = nn.AvgPool2d(kernel_size=(3,3))
        """ input: [batch_size, channels, h, w]"""
        self.linear = nn.Linear(in_channels, CONFIG["num_classes"])
        # self.softmax = nn.Softmax(dim=1)
        self.residual_blocks = [ResidualBlock(in_channels, negative_slope)] * 2
        self.conv_continue_residual = [ConvBlock(in_channels, negative_slope),
                             ResidualBlock(in_channels, negative_slope)] * 2
        self.flatten = nn.Flatten()

    def forward(self, x):
        y = self.conv(x)
        for block in self.residual_blocks:
            y = block(y)
        for block in self.conv_continue_residual:
            y = block(y)
        y = self.flatten(self.pool(y))
        # print(y.shape)
    
        self.linear = nn.Linear(y.shape[1], CONFIG["num_classes"])
        y = self.linear(y)
        # print(y.shape)
        # y = self.softmax(y)
        return y






