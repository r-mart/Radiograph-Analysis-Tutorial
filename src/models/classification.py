import torchvision
import torch.nn as nn


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
    def __init__(self, cfg):
        super().__init__()
        self.flatten = nn.Flatten()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding='valid'),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding='valid'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3, padding='valid'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding='valid'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, padding='valid'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding='valid'),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 10),
            nn.ReLU(),
            nn.Linear(10, cfg.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.squeeze()
        logits = self.fc(x)

        return logits
