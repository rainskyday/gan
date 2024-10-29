import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# 定义参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0002
batch_size = 16

num_epochs = 50
z_dim = 100
image_dim = 28 * 28

# 储存结构的位置
if not os.path.exists("results"):
    os.makedirs("results")

transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 载入数据集
train_dataset = torchvision.datasets.MNIST(root='../data/mnist', train=True, download=True,
                                     transform=torchvision.transforms.Compose(
                                         [
                                             torchvision.transforms.Resize(28),
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize((0.5,), (0.5,)),
                                         ]
                                     ))

# train_dataset = torchvision.datasets.FashionMNIST(root="../data/", train=True, transform=transform, download=True)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.fc1 = nn.Linear(z_dim + 10, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, image_dim)

    def forward(self, noise, labels):
        emb = self.label_emb(labels)
        x = torch.cat([noise, emb], 1)
        x = nn.LeakyReLU(0.2)(self.fc1(x))
        x = nn.LeakyReLU(0.2)(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.fc1 = nn.Linear(image_dim + 10, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, image, labels):
        emb = self.label_emb(labels)

        # emb.view(64, -1)
        # print(image.size(), emb.size())
        x = torch.cat([image, emb], 1)
        x = x.view(image.size(0), -1)
        x = nn.LeakyReLU(0.2)(self.fc1(x))
        x = nn.LeakyReLU(0.2)(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# 初始化网络
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 定义损失函数
criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=lr)
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)


# 训练网络
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        real_images = images.view(batch_size, -1).to(device)
        real_labels = labels.to(device)
        labels_one = torch.ones(batch_size, 1).to(device)
        labels_zero = torch.zeros(batch_size, 1).to(device)

        # 训练判别器:真实图像
        real_pred = discriminator(real_images, real_labels)
        d_loss_real = criterion(real_pred, labels_one)

        # 训练判别器:假图像
        z = torch.randn(batch_size, z_dim).to(device)
        fake_images = generator(z, labels.to(device))
        fake_pred = discriminator(fake_images, labels.to(device))
        d_loss_fake = criterion(fake_pred, labels_zero)

        # 计算判别器损失
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # 训练生成器
        z = torch.randn(batch_size, z_dim).to(device)
        fake_images = generator(z, labels.to(device))
        gen_pred = discriminator(fake_images, labels.to(device))
        g_loss = criterion(gen_pred, labels_one)
        # 反向传播
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 200 == 0:
            print("Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f},".format(
                epoch, num_epochs, i + 1, len(train_loader), d_loss.item(), g_loss.item()
            ))
        if i % 7500 == 0:
            image = fake_images[:16].data
            torchvision.utils.save_image(image, f"results/image_{len(train_loader) * epoch + i}.png", nrow=4)
    # with torch.no_grad():
    #     z = torch.randn(10, z_dim).to(device)
    #     labels = torch.LongTensor(np.array(10)).to(device)
    #     samples = generator(z, labels).cpu().data
    #     fig, axs = plt.subplots(1, 10, figsize=(10, 1))
    #     for j in range(10):
    #         axs[j].imshow(samples[j].view(28, 28), cmap="gray")
    #         axs[j].axis("off")
    #     plt.show()

