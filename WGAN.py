import torch
import torchvision
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 超参数
image_size = [3, 96, 96]
latent_dim = 100
batch_size = 64
n_critic = 5  # 判别器训练次数
clip_value = 0.01  # 权重裁剪范围
learning_rate = 0.00005  # 学习率

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 生成器
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

# 判别器
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
            nn.Linear(512 * (img_size // 16) ** 2, 1)
        )

    def forward(self, img):
        return self.model(img)

generator = Generator(latent_dim=latent_dim, channels=3, img_size=96).to(device)
discriminator = Discriminator(channels=3, img_size=96).to(device)

# 数据加载
transform = transforms.Compose([
    transforms.Resize((96, 96)),  # 调整图片大小为 96x96
    transforms.ToTensor(),       # 将图片转换为 Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

dataset = datasets.ImageFolder(root="anime_face", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 优化器
g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=learning_rate)
d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=learning_rate)
try:
    generator.load_state_dict(torch.load('WGAN_generator_weights.pth', map_location=device))
    discriminator.load_state_dict(torch.load('WGAN_discriminator_weights.pth', map_location=device))
    g_optimizer.load_state_dict(torch.load('WGAN_g_optimizer_state.pth', map_location=device))
    d_optimizer.load_state_dict(torch.load('WGAN_d_optimizer_state.pth', map_location=device))
    progress = torch.load('WGAN_training_progress.pth', map_location=device)
    start_epoch = progress['epoch']
    start_batch = progress['batch']
    print(f"Resuming training from epoch {start_epoch}, batch {start_batch}")
except FileNotFoundError:
    print("Starting training from scratch")
    start_epoch = 0
    start_batch = 0

# 训练过程
num_epochs = 200
for epoch in range(num_epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        # 训练判别器
        for _ in range(n_critic):
            d_optimizer.zero_grad()
            real_imgs = real_imgs.to(device)
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z).detach()  # 生成假图像

            # 计算损失
            loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))
            loss_D.backward()
            d_optimizer.step()

            # 权重裁剪
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

        # 训练生成器
        g_optimizer.zero_grad()
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = generator(z)
        loss_G = -torch.mean(discriminator(fake_imgs))
        loss_G.backward()
        g_optimizer.step()

        # 打印日志
        if i % 50 == 0:
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                  f"[D loss: {loss_D.item()}] [G loss: {loss_G.item()}]")

        # 保存生成图像
        if i % 400 == 0:
            image = (fake_imgs[:16].data + 1) / 2
            torchvision.utils.save_image(image, f"image_{epoch}_{i}.png", nrow=4)

        # 保存模型
        progress = {'epoch': epoch, 'batch': 0}
        torch.save(progress, 'WGAN_training_progress.pth')
        torch.save(generator.state_dict(), 'WGAN_generator_weights.pth')
        torch.save(discriminator.state_dict(), 'WGAN_discriminator_weights.pth')
        torch.save(g_optimizer.state_dict(), 'WGAN_g_optimizer_state.pth')
        torch.save(d_optimizer.state_dict(), 'WGAN_d_optimizer_state.pth')
