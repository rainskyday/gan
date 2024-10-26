import torch
import torchvision
import torch.nn as nn
import numpy as np
import os

# MNIST数据集 单通道28*28
image_size = [1, 28, 28]
# 潜在变量 噪声大小
z_dim = 100
# 批训练大小
batch_size = 16
# 设备选取
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1、准备数据集
dataset = torchvision.datasets.MNIST(root='../data/mnist', train=True, download=True,
                                     transform=torchvision.transforms.Compose(
                                         [
                                             torchvision.transforms.Resize(28),
                                             torchvision.transforms.ToTensor(),
                                         ]
                                     ))
if not os.path.exists("results"):
    os.makedirs("results")

# 2、加载数据集
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 3、搭建神经网络
# 生成器
class Generator(nn.Module):
    def __init__(self):
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


# 判别器
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
            nn.Sigmoid()  # 输出范围在 [0, 1]
        )
    def forward(self, img, y=None):
        # 判别器的输入是一张28*28的图片，输出是一个0-1之间的概率值
        y_ = self.model(img)
        return y_.view(y_.size(0), -1)


# 4、创建网络模型
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 5、设置损失函数、优化器
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
# 损失函数
loss_fn = nn.BCELoss()
# 真假图像的标签
labels_one = torch.ones(batch_size, 1).to(device)
labels_zero = torch.zeros(batch_size, 1).to(device)

# 6、训练网络
num_epoch = 50
for epoch in range(num_epoch):
    for i, mini_batch in enumerate(dataloader):
        # 读入真实图像
        gt_images = mini_batch[0].to(device)
        # 生成速记噪声
        z = torch.randn(batch_size, z_dim).to(device)
        # 生成假图
        fake_images = generator(z)
        # 生成器损失
        g_loss = loss_fn(discriminator(fake_images), labels_one)
        # 生成器反向传播梯度更新
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        # 判别器损失
        real_loss = loss_fn(discriminator(gt_images), labels_one)
        fake_loss = loss_fn(discriminator(fake_images.detach()), labels_zero)
        # 总损失
        d_loss = real_loss + fake_loss
        # 判别器反向传播梯度更新
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        if i % 500 == 0:
            print(f"Epoch:{epoch}, step:{len(dataloader) * epoch + i}, g_loss:{g_loss.item()}, d_loss:{d_loss.item()}")
        if i % 7500 == 0:
            image = fake_images[:16].data
            torchvision.utils.save_image(image, f"results/image_{len(dataloader) * epoch + i}.png", nrow=4)
