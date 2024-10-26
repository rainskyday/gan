import torch
import torch.nn as nn

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