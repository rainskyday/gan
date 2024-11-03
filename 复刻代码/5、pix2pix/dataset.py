import os
import numpy as np
import config
from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

class Map_dataset():
    def __init__(self, root_dir, mode='train'):
        self.root_dir = root_dir
        self.mode = mode
        self.list_files = os.listdir(os.path.join(self.root_dir, self.mode))
        # 定义转换
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize 到 256x256
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.5,), (0.5,))  # 归一化
        ])
    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, self.mode, img_file)
        image = np.array(Image.open(img_path))
        # 拆分标签和图像
        input_image = image[:, :600, :]
        input_label = image[:, 600:, :]

        # 转换为张量
        input_image = self.transform(Image.fromarray(input_image))
        input_label = self.transform(Image.fromarray(input_label))

        #
        # augmentations = config.both_transform(image=input_image, image0=input_label)
        # input_image = augmentations["image"]
        # input_label = augmentations["image0"]
        # 随机翻转
        # input_image = config.transform_only_input(image=input_image)["image"]
        # input_label = config.transform_only_mask(image=input_label)["image"]

        return input_image, input_label

if __name__ == "__main__":
    # import warnings
    # # 忽视特定的 SSL 警告
    # warnings.filterwarnings("ignore", category=UserWarning)
    dataset = Map_dataset("..\\data\\archive\\maps\\maps")

    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys
        sys.exit()
