import numpy as np
import torchvision
import torch
import torch.nn as nn
from torch import Tensor


class DetectionBaseline(nn.Module):
    """ Based on the SSD approach """

    def __init__(self, cfg):
        super().__init__()

        self.n_anchors = cfg.n_anchors  # number of anchor boxes used for each position
        self.n_classes = cfg.num_classes
        self.n_outputs = (4 + self.n_classes)
        self.n_preds = cfg.n_anchors * self.n_outputs
        self.batch_size = cfg.batch_size

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
        #pred = torch.reshape(pred, [self.batch_size, -1, self.n_outputs])

        return pred


def get_feature_map_anchor_boxes(feature_map_shape_list, **anchor_kwargs):
    """
    :param feature_map_shape_list: list of tuples containing feature map resolutions
    :returns: dict with feature map shape tuple as key and list of [ymin, xmin, ymax, xmax] box co-ordinates
    """
    anchor_generator = create_anchors(**anchor_kwargs)

    anchor_box_lists = anchor_generator.generate(feature_map_shape_list)

    feature_map_boxes = {}

    # with tf.Session() as sess:
    for shape, box_list in zip(feature_map_shape_list, anchor_box_lists):
        feature_map_boxes[shape] = box_list.data['boxes']
        # feature_map_boxes[shape] = sess.run(box_list.data['boxes'])

    return feature_map_boxes


def create_anchors(cell_counts, aspect_ratios, n_anchors, img_dim):

    l_anchors_list = []

    for n_cells in cell_counts:
        l_anchors = create_layer_anchors(
            n_cells, aspect_ratios, n_anchors, img_dim)
        l_anchors_list.append(l_anchors)

    anchors = np.concatenate(l_anchors_list)

    return anchors


def create_layer_anchors(n_pos, aspect_ratios, n_anchors, img_dim):
    """ Currently only implemented for images with width = height """

    n_aspect_ratios = len(aspect_ratios)

    coords = np.linspace(0, img_dim-1, n_pos, endpoint=False)
    coords = np.append(coords, [img_dim-1])

    c_starts = coords[0:-1]
    c_ends = coords[1:]
    c_mids = (coords[0:-1] + coords[1:]) / 2

    c_dim = c_ends[0] - c_starts[0]

    anchors = np.zeros(
        (n_pos, n_pos, n_anchors * n_aspect_ratios, 4), np.float32)

    for i_y in range(n_pos):
        for i_x in range(n_pos):
            c_center_x = c_mids[i_x]
            c_center_y = c_mids[i_y]

            for i_b in range(n_anchors):
                c_w = c_h = c_dim * (i_b+1) / n_anchors

                for i_ar, aspect_ratio in enumerate(aspect_ratios):
                    # width and height factors such that
                    # area' = area
                    # aspect_ratio' = aspect_ratio
                    w = np.sqrt(aspect_ratio) * c_w
                    h = np.sqrt(1/aspect_ratio) * c_h
                    i_anc = i_b * n_aspect_ratios + i_ar

                    anchors[i_y, i_x, i_anc, 0] = c_center_x - w/2  # x1
                    anchors[i_y, i_x, i_anc, 1] = c_center_y - h/2  # y1
                    anchors[i_y, i_x, i_anc, 2] = c_center_x + w/2  # x2
                    anchors[i_y, i_x, i_anc, 3] = c_center_y + h/2  # y2

    anchors = anchors.reshape((-1, 4))

    return anchors


def one_hot_encoding(indices, depth):
    indices = np.array(indices, dtype=np.int32)
    one_hot = np.zeros((len(indices), depth), dtype=np.float32)
    one_hot[np.arange(len(indices)), indices] = 1
    return one_hot
