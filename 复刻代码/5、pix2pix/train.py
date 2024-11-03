import torch
import torchvision
import torch.nn as nn
import numpy as np
import os
from model import Generator, Discriminator
from dataset import Map_dataset
from tqdm import tqdm
from utils import save_some_examples
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = True
# 1.加载数据
# 超参数
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epoch = 300
L1_lambda = 100
# 储存结构的位置
if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists("weight"):
    os.makedirs("weight")
if not os.path.exists("evaluation"):
    os.makedirs("evaluation")

# 2.加载数据集
train_dataset = Map_dataset("..\\data\\archive\\maps\\maps")
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# 测试集
test_dataset = Map_dataset("..\\data\\archive\\maps\\maps", mode='val')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# 3.搭建网络模型 在model中完成了
# 4.创建网络模型
discriminator = Discriminator(in_channels=3).to(device)
generator = Generator(in_channels=3, features=64).to(device)
# 5.设置优化器和损失函数
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999), )
# 损失函数，两个，在原论文中使用的L1+Cgan
BCE = nn.BCEWithLogitsLoss()
L1 = nn.L1Loss()
# 混合精度
g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()
# 6.训练网络
# 需要定义好的变量有：生成器、判别器、数据加载器、生成器优化器、判别器优化器、损失函数（两个）、设备（）cuda

# 开始训练
for epoch in range(num_epoch):
    # print(f"Epoch [{epoch}/{num_epoch}]")
    # 每次训练
    # 读入tqdm进度
    loop = tqdm(train_loader, leave=True)
    for i, (real_image, real_target) in enumerate(loop):
        # 读入真实图像和标签
        real_image = real_image.to(device)
        real_target = real_target.to(device)
        # image是左侧的卫星图，target是右侧的地图，生成器生成的是地图
        # 训练判别器
        with torch.cuda.amp.autocast():
            # 生成假图像
            fake_target = generator(real_image)
            # 训练判别器
            # 计算真实图像和真实目标的分类结果，patchnet
            d_real = discriminator(real_image, real_target)
            # 计算判别器的bce损失
            d_real_loss = BCE(d_real, torch.ones_like(d_real))
            # 计算真实图像和生成的图像的分类结果，patchnet
            d_fake = discriminator(real_image, fake_target.detach())
            # 计算判别器的bce损失
            d_fake_loss = BCE(d_fake, torch.zeros_like(d_fake))
            d_loss = (d_real_loss + d_fake_loss) / 2

        # 反向传播
        discriminator.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(d_optimizer)
        d_scaler.update()
        # 训练生成器
        with torch.cuda.amp.autocast():
            d_fake = discriminator(real_image, fake_target)
            # bce损失
            g_fake_loss = BCE(d_fake, torch.ones_like(d_fake))
            # 计算L1损失
            l1_loss = L1(fake_target, real_target) * L1_lambda
            # 总的损失
            g_loss = g_fake_loss + l1_loss
        # 反向传播
        g_optimizer.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(g_optimizer)
        g_scaler.update()

        if i % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(d_real).mean().item(),
                D_fake=torch.sigmoid(d_fake).mean().item(),
            )
    # 保存样例
    save_some_examples(generator, test_loader, epoch, folder="evaluation")
    # 保存模型
    torch.save(generator.state_dict(), f"weight/generator.pth")
    torch.save(discriminator.state_dict(), f"weight/discriminator.pth")

        # 输出
        # if i % 100 == 0:
        #     print(f"Epoch [{epoch}/{num_epoch}] Batch {i}/{len(train_loader)} \
        #           Loss D: {d_loss:.4f}, loss G: {g_loss:.4f}")
        # if i % len(train_loader) == len(train_loader) - 1:
        #     fake_image = fake_image * 0.5 + 0.5
        #     real_image = real_image * 0.5 + 0.5
        #     torchvision.utils.save_image(fake_image, f"results/fake_image_{epoch}_{i}.png")
        #     torchvision.utils.save_image(real_image, f"results/real_image_{epoch}_{i}.png")


