from math import sqrt
import torchvision
import torch
import torch.nn as nn
from torch import Tensor


class DetectionBaseline(nn.Module):
    """ Based on the SSD approach """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.n_anchors = cfg.n_anchors  # number of anchor boxes used for each position
        self.n_classes = cfg.num_classes
        self.n_outputs = (4 + self.n_classes)
        self.n_preds = cfg.n_anchors * self.n_outputs
        self.batch_size = cfg.batch_size

        self.anchors = self.create_anchors()  # in cxcywh format

        feat_model = torchvision.models.resnet18(pretrained=True)
        feature_depth = feat_model.fc.in_features
        # input for grayscale images
        feat_model.conv1 = nn.Conv2d(1, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # get feature layers only
        modules = list(feat_model.children())[:-2]  # drop avgpool, fc
        feat_model = nn.Sequential(*modules)

        if cfg.freeze_backbone:
            # turn off gradients for pretrained feature layers
            for i, p in enumerate(feat_model.parameters()):
                if i > 0:  # keep gradients for the new grayscale conv1
                    p.requires_grad = False
        self.backbone = feat_model

        self.pred = nn.Conv2d(feature_depth, self.n_preds,
                              kernel_size=3, padding='same')

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        pred = self.pred(x)
        pred = torch.reshape(pred, [self.batch_size, -1, self.n_outputs])

        return pred

    def create_anchors(self):
        """
        Create the anchors for the model.

        :return: anchors in relative center-size coordinates, a tensor of dimensions (n, 4)
        """
        fmap_dims = {'pred': 24}
        obj_scales = {'pred': 0.25}
        aspect_ratios = {'pred': [1.]}  # [1., 2., 0.5]

        fmaps = list(fmap_dims.keys())
        anchors = []

        for fmap in fmaps:
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        anchors.append(
                            [cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

        anchors = torch.FloatTensor(anchors).to(self.cfg.device)
        anchors.clamp_(0, 1)

        return anchors
