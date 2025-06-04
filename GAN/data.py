from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


class TextureDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = []
        # 使用 os.walk() 来获取所有子目录中的文件
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.images.append(os.path.join(root, file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 假设模型需要128x128的输入
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
])
dataset = TextureDataset(directory='images', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
