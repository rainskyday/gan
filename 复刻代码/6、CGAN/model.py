import torch
import torch.nn as nn

# 该生成器只用于MNIST数据集
# MNIST数据集 单通道28*28
image_size = [1, 28, 28]
class Generator(nn.Module):
    def __init__(self, z_dim, label_dim):
        super(Generator, self).__init__()
        # 合并输入
        self.label_emb = nn.Embedding(label_dim, label_dim)
        # 生成器网络
        self.model = nn.Sequential(
            # CGAN网络的输入是一个100维的随机噪声和一个10维的标签，输出是一个784维的向量，代表一张28*28的图片
            # 搭建网络  三层结构的网络 100z到128*7*7 原文中写使用了ReLU函数在生成器中
            nn.ConvTranspose2d(z_dim + label_dim, 128, 7, 1, 0, bias=False),
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
    def forward(self, z, labels):
        label_embedded = self.label_emb(labels)
        input = torch.cat([z, label_embedded], dim=1).view(-1, z.size(1) + label_embedded.size(1), 1, 1)
        # 输出是一张图像
        output = self.model(input)
        # image = output.view(output.size(0), image_size[1] * image_size[2])
        # print(output.size())
        return output


# 判别器
class Discriminator(nn.Module):
    def __init__(self, label_dim):
        super(Discriminator, self).__init__()
        # 判别器网络
        self.label_emb = nn.Embedding(label_dim, 28*28)
        self.model = nn.Sequential(
            # 输入尺寸28*28
            nn.Conv2d(2, 64, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出尺寸（28-4）/2 + 1 = 14
            nn.Conv2d(64, 128, 5, 2, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出尺寸（14-4）/2 + 1 = 7
            nn.Conv2d(128, 1, 7, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, image, labels):
        # 判别器的输入是一张28*28的图片，输出是一个0-1之间的概率值
        image = image.view(-1, 1, 28, 28)
        label_embedded = self.label_emb(labels)
        label_embedded = label_embedded.view(-1, 1, 28, 28)
        # print(image.size(), label_embedded.size())
        output = torch.cat([image, label_embedded], dim=1).view(-1, 2, 28, 28)
        # print(output.size())
        output = self.model(output)

        return output.view(output.size(0), -1)
