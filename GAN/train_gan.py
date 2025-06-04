import torch
import torch.optim as optim
from data import dataloader
from models.discriminator import Discriminator
from models.dynca import DyNCA, CPE2D
from pytorch_fid import fid_score
import os
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_pil_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化模型
generator = DyNCA(c_in=3, c_out=3, device=device).to(device)
discriminator = Discriminator().to(device)

# 优化器
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 损失函数
adversarial_loss = torch.nn.BCELoss()

# 文件夹路径
real_images_path = 'real_images'
generated_images_path = 'generated_images'
temp_real_images_path = 'temp_real_images'
temp_generated_images_path = 'temp_generated_images'
os.makedirs(generated_images_path, exist_ok=True)
os.makedirs(temp_real_images_path, exist_ok=True)
os.makedirs(temp_generated_images_path, exist_ok=True)


def get_all_images(directory):
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    return image_files


def resize_and_save_images(image_paths, save_directory, transform):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        img = to_pil_image(img)  # 将 Tensor 转换回 PIL 图像
        save_path = os.path.join(save_directory, os.path.basename(img_path))
        img.save(save_path)


# 统一图像尺寸的 transform
resize_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

if __name__ == "__main__":
    # 训练过程
    num_epochs = 50
    fid_scores = []
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            real_imgs = data.to(device)
            valid = torch.ones((real_imgs.size(0),), dtype=torch.float, device=device)
            fake = torch.zeros((real_imgs.size(0),), dtype=torch.float, device=device)

            # 训练生成器
            optimizer_G.zero_grad()
            z = generator.seed(real_imgs.size(0))
            generated_imgs = generator(z)[1]
            g_loss = adversarial_loss(discriminator(generated_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # 训练判别器
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        # 保存生成图像到文件夹
        for j, img in enumerate(generated_imgs):
            save_image(img, f"{generated_images_path}/{epoch}_{j}.png")

        # 获取所有图像文件路径
        real_images = get_all_images(real_images_path)
        generated_images = get_all_images(generated_images_path)

        # 调试输出图像文件数
        print(f"Epoch {epoch} - Real images: {len(real_images)}, Generated images: {len(generated_images)}")

        # 确保文件夹中有图像
        if len(real_images) > 0 and len(generated_images) > 0:
            # 调整并保存图像到临时目录
            resize_and_save_images(real_images, temp_real_images_path, resize_transform)
            resize_and_save_images(generated_images, temp_generated_images_path, resize_transform)

            # 打印要传递的路径
            print(f"Calculating FID for paths: {temp_real_images_path} and {temp_generated_images_path}")

            # 计算并记录 FID 分数
            fid = fid_score.calculate_fid_given_paths([temp_real_images_path, temp_generated_images_path], batch_size=32, device=device, dims=2048)
            fid_scores.append(fid)
            print(f"[Epoch {epoch}/{num_epochs}] FID score: {fid}")
        else:
            print("Warning: One of the image directories is empty. Skipping FID calculation for this epoch.")

    # 保存模型
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
