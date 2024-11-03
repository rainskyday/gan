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

# 创建模型实例
generator = Generator(z_dim, 10).to(device)
discriminator = Discriminator(10).to(device)

# 加载模型权重
generator.load_state_dict(torch.load('weight/generator.pth'))  # 根据需要选择最后一个epoch
discriminator.load_state_dict(torch.load('weight/discriminator.pth'))
# 目的值
target_num = 4

# 生成图像
generator.eval()  # 将生成器设置为评估模式
with torch.no_grad():
    z = torch.randn(16, z_dim).to(device)  # 生成随机噪声
    labels = torch.full((16,), target_num).to(device)

    fake_images = generator(z, labels)
    torchvision.utils.save_image(fake_images, 'results/generated_images.png', nrow=4)
