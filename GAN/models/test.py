import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 定义每个单独的层
        self.conv1 = nn.Conv2d(3, 64, 4, stride=2, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)
        self.conv5 = nn.Conv2d(512, 1, 4, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.lrelu3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.lrelu4(x)
        # 在最后一个卷积层之前打印特征图尺寸
        print("Feature map size before last Conv:", x.size())
        x = self.conv5(x)
        x = self.sigmoid(x)
        return x.view(-1, 1).squeeze(1)


# 实例化并使用鉴别器
discriminator = Discriminator()
