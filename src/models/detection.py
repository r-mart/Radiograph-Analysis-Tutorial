from math import sqrt
import torchvision
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

from src.train.utils import find_jaccard_overlap, cxcywh_to_xyxy, gcxgcywh_to_cxcywh


class DetectionBaseline(nn.Module):
    """ Based on the SSD approach """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.n_classes = cfg.num_classes
        self.aspect_ratios = cfg.aspect_ratios
        # number of anchors per feature map cell
        self.anchors_per_cell = len(self.aspect_ratios)
        self.n_locs = self.anchors_per_cell * 4
        self.n_confs = self.anchors_per_cell * self.n_classes

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

        self.l_pred = nn.Conv2d(feature_depth, self.n_locs,
                                kernel_size=3, padding='same')
        self.c_pred = nn.Conv2d(feature_depth, self.n_confs,
                                kernel_size=3, padding='same')

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)

        x = self.backbone(x)

        # box localization prediction
        l_pred = self.l_pred(x)  # (N, n_locs, out_size, out_size)
        # reshape to match prior-box order
        # .contiguous() ensures it is stored in a contiguous chunk of memory (needed for .view())
        l_pred = l_pred.permute(0, 2, 3,
                                1).contiguous()  # (N, out_size, out_size, n_locs)
        l_pred = l_pred.view(batch_size, -1, 4)  # (N, n_layer_anchors, 4)

        # class confidence score prediction
        c_pred = self.c_pred(x)  # (N, n_classes, out_size, out_size)
        c_pred = c_pred.permute(0, 2, 3,
                                1).contiguous()  # (N, out_size, out_size, n_classes)
        c_pred = c_pred.view(batch_size, -1,
                             self.n_classes)  # (N, n_layer_anchors, n_classes)

        return l_pred, c_pred

    def create_anchors(self):
        """
        Create the anchors for the model.

        :return: anchors in relative center-size coordinates, a tensor of dimensions (n, 4)
        """
        fmap_dim_dict = {
            'pred': self.cfg.img_size[0] // 32}  # assuming last feature layer of a resnet (has a reduction factor of 32)
        obj_scale_dict = {'pred': 0.35}
        aspect_ratio_dict = {'pred': self.aspect_ratios}

        fmaps = list(fmap_dim_dict.keys())
        anchors = []

        for fmap in fmaps:
            for i in range(fmap_dim_dict[fmap]):
                for j in range(fmap_dim_dict[fmap]):
                    cx = (j + 0.5) / fmap_dim_dict[fmap]
                    cy = (i + 0.5) / fmap_dim_dict[fmap]

                    for ratio in aspect_ratio_dict[fmap]:
                        anchors.append(
                            [cx, cy, obj_scale_dict[fmap] * sqrt(ratio), obj_scale_dict[fmap] / sqrt(ratio)])

        anchors = torch.FloatTensor(anchors).to(self.cfg.device)
        anchors.clamp_(0, 1)

        return anchors

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the anchor locations and class scores to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the anchors, a tensor of dimensions (N, n_anchors, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, n_anchors, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_anchors = self.anchors.size(0)
        # (N, n_anchors, n_classes)
        predicted_scores = F.softmax(predicted_scores, dim=2)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_anchors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcywh_to_xyxy(
                gcxgcywh_to_cxcywh(predicted_locs[i], self.anchors))  # (n_anchors, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (n_anchors)
                # torch.uint8 (byte) tensor, for indexing
                score_above_min_score = class_scores > min_score
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                # (n_qualified), n_min_score <= n_anchors
                class_scores = class_scores[score_above_min_score]
                # (n_qualified, 4)
                class_decoded_locs = decoded_locs[score_above_min_score]

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(
                    dim=0, descending=True)  # (n_qualified), (n_min_score)
                # (n_min_score, 4)
                class_decoded_locs = class_decoded_locs[sort_ind]

                # Find the overlap between predicted boxes
                # (n_qualified, n_min_score)
                overlap = find_jaccard_overlap(
                    class_decoded_locs, class_decoded_locs)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(
                    self.cfg.device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                keep = (1 - suppress).to(torch.bool)
                image_boxes.append(class_decoded_locs[keep])
                image_labels.append(torch.LongTensor(
                    (keep).sum().item() * [c]).to(self.cfg.device))
                image_scores.append(class_scores[keep])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor(
                    [[0., 0., 1., 1.]]).to(self.cfg.device))
                image_labels.append(torch.LongTensor([0]).to(self.cfg.device))
                image_scores.append(
                    torch.FloatTensor([0.]).to(self.cfg.device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(
                    dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        # lists of length batch_size
        return all_images_boxes, all_images_labels, all_images_scores
