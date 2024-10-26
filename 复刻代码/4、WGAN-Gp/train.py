import torch
import torchvision
import torch.nn as nn
import numpy as np
import os
from model import Generator, Discriminator
from utils import gradient_penalty

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
                                             torchvision.transforms.Normalize((0.5,), (0.5,)),
                                         ]
                                     ))
# 储存结构的位置
if not os.path.exists("results"):
    os.makedirs("results")

# 2、加载数据集
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 3、搭建神经网络
# 在model文件中完成了



# 4、创建网络模型
generator = Generator(z_dim).to(device)
discriminator = Discriminator().to(device)

# 5、设置优化器
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.0, 0.9))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.0, 0.9))

z = []
fake_images = []

# 6、训练网络
lambda_reg = 10
num_epoch = 50
critic_iterations = 5  # WGAN中的判别器训练次数
for epoch in range(num_epoch):
    for i, mini_batch in enumerate(dataloader):
        # 读取真实图像
        gt_images = mini_batch[0].to(device)

        # 训练判别器
        for _ in range(critic_iterations):
            z = torch.randn(batch_size, z_dim).to(device)  # 生成随机噪声
            fake_images = generator(z)  # 生成假图像

            # 判别器的损失
            real_loss = -torch.mean(discriminator(gt_images))  # 真实图像损失
            fake_loss = torch.mean(discriminator(fake_images.detach()))  # 假图像损失
            gp = gradient_penalty(discriminator, gt_images, fake_images.detach(), device=device)
            # 总损失
            d_loss = real_loss + fake_loss + 10 * gp

            # 判别器反向传播梯度更新
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

        # 生成器的损失
        g_loss = -torch.mean(discriminator(fake_images))  # 生成器损失
        # 生成器反向传播梯度更新
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if i % 500 == 0:
            print(f"Epoch:{epoch}, Step:{len(dataloader) * epoch + i}, g_loss:{g_loss.item()}, d_loss:{d_loss.item()}")
        if i % 7500 == 0:
            image = fake_images[:16].data
            torchvision.utils.save_image(image, f"results/image_{len(dataloader) * epoch + i}.png", nrow=4)
