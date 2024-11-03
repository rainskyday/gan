import torch
import torchvision
import torch.nn as nn
import numpy as np
import os
from model import Generator, Discriminator

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
if not os.path.exists("weight"):
    os.makedirs("weight")

# 2、加载数据集
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 3、搭建神经网络
# 在model文件中完成了



# 4、创建网络模型
generator = Generator(z_dim, 10).to(device)
discriminator = Discriminator(10).to(device)

# 5、设置优化器
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
# 损失函数
loss_fn = nn.BCELoss()
# 真假图像的标签
labels_one = torch.ones(batch_size, 1).to(device)
labels_zero = torch.zeros(batch_size, 1).to(device)

z = []
fake_images = []

# 6、训练网络
lambda_reg = 10
num_epoch = 30

for epoch in range(num_epoch):
    for i, (gt_images, gt_labels) in enumerate(dataloader):
        # 读入真实图像和标签
        real_images = gt_images.view(batch_size, -1).to(device)
        real_labels = gt_labels.to(device)
        labels_one = torch.ones(batch_size, 1).to(device)
        labels_zero = torch.zeros(batch_size, 1).to(device)

        # 训练判别器: 1.真实图像的loss 2.生成图像的loss
        real_pred = discriminator(real_images, real_labels)
        d_loss_real = loss_fn(real_pred, labels_one)
        # 2.生成图像的loss
        z = torch.randn(batch_size, z_dim).to(device)
        fake_images = generator(z, gt_labels.to(device))
        fake_pred = discriminator(fake_images, gt_labels.to(device))
        d_loss_fake = loss_fn(fake_pred, labels_zero)

        # 3.总的loss
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # 训练生成器
        z = torch.randn(batch_size, z_dim).to(device)
        fake_images = generator(z, gt_labels.to(device))
        gen_pred = discriminator(fake_images, gt_labels.to(device))
        g_loss = loss_fn(gen_pred, labels_one)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # d_optimizer.zero_grad()
        # d_real_labels = torch.ones(batch_size, 1).to(device)
        # d_fake_labels = torch.zeros(batch_size, 1).to(device)

        # output = discriminator(gt_images, gt_labels).view(batch_size, 1)
        # real_loss = loss_fn(output, d_real_labels)
        #
        # z = torch.randn(batch_size, z_dim).to(device)  # 生成随机噪声
        # fake_images = generator(z, gt_labels)
        # g_fake_labels = torch.randint(0, 10, (batch_size,)).to(device)
        # outputs = discriminator(fake_images.detach(), g_fake_labels).view(batch_size, 1)
        # fake_loss = loss_fn(outputs, d_fake_labels)
        #
        # d_loss = real_loss + fake_loss
        # d_loss.backward()
        # d_optimizer.step()
        #
        # # 训练生成器
        # g_optimizer.zero_grad()
        # outputs = discriminator(fake_images, g_fake_labels).view(batch_size, 1)
        # g_loss = loss_fn(outputs, d_real_labels)
        # g_loss.backward()
        # g_optimizer.step()

        if i % 500 == 0:
            print(f"Epoch:{epoch}, Step:{len(dataloader) * epoch + i}, g_loss:{g_loss.item()}, d_loss:{d_loss.item()}")
        if i % 7500 == 0:
            image = fake_images[:16].data
            torchvision.utils.save_image(image, f"results/image_{len(dataloader) * epoch + i}.png", nrow=4)
    torch.save(generator.state_dict(), "weight/generator.pth")
    torch.save(discriminator.state_dict(), "weight/discriminator.pth")
    
