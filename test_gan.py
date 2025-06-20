import torch
import torchvision
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#超参数
image_size = [3, 96, 96]
latent_dim = 100
batch_size = 64
# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
#生成器
class Generator(nn.Module):

    def __init__(self, latent_dim=100, channels=3, img_size=96):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.GELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.GELU(),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
#判别器
class Discriminator(nn.Module):
    def __init__(self, channels=3, img_size=96):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(512 * (img_size // 16) ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

generator = Generator(latent_dim=latent_dim, channels=3, img_size=96).to(device)
discriminator = Discriminator(channels=3, img_size=96).to(device)



transform = transforms.Compose([
    transforms.Resize((96, 96)),  # 调整图片大小为 96x96
    transforms.ToTensor(),       # 将图片转换为 Tensor
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])  # 归一化
])

dataset = datasets.ImageFolder(root="anime_face", transform=transform)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=True)

g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)

loss_fn = nn.BCELoss()
labels_one = torch.ones(batch_size, 1)
labels_zero = torch.zeros(batch_size, 1)

try:
    generator.load_state_dict(torch.load('generator_weights.pth', map_location=device))
    discriminator.load_state_dict(torch.load('discriminator_weights.pth', map_location=device))
    g_optimizer.load_state_dict(torch.load('g_optimizer_state.pth', map_location=device))
    d_optimizer.load_state_dict(torch.load('d_optimizer_state.pth', map_location=device))
    progress = torch.load('training_progress.pth', map_location=device)
    start_epoch = progress['epoch']
    start_batch = progress['batch']
    print("Testing")
except FileNotFoundError:
    print("No weights")
    start_epoch = 0
    start_batch = 0

z = torch.randn(batch_size, latent_dim).to(device)

pred_images = generator(z)
image = (pred_images[:16].data + 1) / 2
torchvision.utils.save_image(image, f"image_test.png", nrow=4)
