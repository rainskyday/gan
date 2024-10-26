import torch
import torchvision
import torch.nn as nn
import numpy as np

# 该生成器只用于MNIST数据集
# MNIST数据集 单通道28*28
image_size = [1, 28, 28]
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # 搭建网络  三层结构的网络 100z到128*7*7 原文中写使用了ReLU函数在生成器中
            nn.ConvTranspose2d(z_dim, 128, 7, 1, 0, bias=False),
            # 数据归一化
            nn.BatchNorm2d(128),
            # 激活函数
            nn.ReLU(),
            # 输出尺寸（7-1）* 2 + 4 - 2 * 2 = 14
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 输出尺寸（14-1）* 2 + 4 - 2 * 2 = 28
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z):
        # 生成器的输入是一个100维的随机噪声，输出是一个784维的向量，代表一张28*28的图片,转置卷积是一个需要4维输入的函数，分别是batch_size, channel, height, width
        z = z.view(z.size(0), z.size(1), 1, 1)  # 输出是一张图像
        # 输出是一张图像
        output = self.model(z)
        image = output.view(output.size(0), *image_size)
        return image


# 判别器（WGAN中的判别器称为“评分器”）
class Discriminator(nn.Module):
    """
        Convolutional Discriminator for MNIST
    """
    def __init__(self, in_channel=1, num_classes=1):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # 输入是图像1*28*28 输出为  (28-5+2*2)/2 + 1 = 14
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            # (14-5+2*2)/2 + 1 = 7
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # (7-5+2*2)/2 + 1 = 4
            nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=0),
            # WGAN不用激活函数sigmoid
            # nn.Sigmoid()  # 输出范围在 [0, 1]
        )
    def forward(self, img, y=None):
        # 判别器的输入是一张28*28的图片，输出是一个0-1之间的概率值
        y_ = self.model(img)
        return y_.view(y_.size(0), -1)