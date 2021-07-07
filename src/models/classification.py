import torchvision
import torch.nn as nn
from torch import Tensor


class ClassificationBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)

        # input for grayscale images
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # adjust number of output classes
        self.model.fc = nn.Linear(self.model.fc.in_features, cfg.num_classes)

    def forward(self, x):
        logits = self.model(x)

        return logits


class ClassificationCustom(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.features = nn.Sequential(
            DefaultBlock(1, 16, kernel_size=7, stride=2, padding=1),
            DownBlock(),
            DefaultBlock(16, 32),
            DownBlock(),
            DefaultBlock(32, 64),
            DownBlock(),
            DefaultBlock(64, 128)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, cfg.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.squeeze()
        logits = self.fc(x)

        return logits


class DefaultBlock(nn.Module):
    """ Implements a default block of a network """

    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super(DefaultBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out


class DownBlock(nn.Module):
    """ Implements the logic for a block downsampling the feature resolution """

    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = 2,
        padding: int = 0,
    ) -> None:
        super(DownBlock, self).__init__()
        self.pool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: Tensor) -> Tensor:
        out = self.pool(x)

        return out
