import torch.nn as nn


class ClassificationBaseline(nn.Module):
    def __init__(self, cfg):
        super(ClassificationBaseline, self).__init__()
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
        self.classifier = nn.Sequential(
            nn.Linear(128, 10),
            nn.ReLU(),
            nn.Linear(10, cfg.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        feature_vector = x.mean(dim=(2, 3))
        logits = self.classifier(feature_vector)

        return logits
