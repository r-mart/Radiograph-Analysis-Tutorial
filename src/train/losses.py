import torch
import torch.nn as nn

from .utils import cxcywh_to_xyxy, xyxy_to_cxcywh, cxcywh_to_gcxgcywh, find_jaccard_overlap


# Taken from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, anchors, cfg, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.anchors_cxcywh = anchors
        self.anchors_xywh = cxcywh_to_xyxy(anchors)
        self.cfg = cfg
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the n_anchors prior boxes, a tensor of dimensions (N, n_anchors, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, n_anchors, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_anchors = self.anchors_cxcywh.size(0)
        n_classes = predicted_scores.size(2)

        assert n_anchors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_anchors, 4), dtype=torch.float).to(
            self.cfg.device)  # (batch_size, n_anchors, 4)
        true_classes = torch.zeros((batch_size, n_anchors), dtype=torch.long).to(
            self.cfg.device)  # (batch_size, n_anchors)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],
                                           self.anchors_xywh)  # (n_objects, n_anchors)

            # For each anchor box, find the object that has the maximum overlap
            overlap_for_each_anchor, object_for_each_anchor = overlap.max(
                dim=0)  # (n_anchors)

            # We want to avoid an object being not represented in the positive (non-background) anchors. E.g. when:
            # 1. The object is not the best object for all anchors
            # 2. All anchors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, anchor_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap anchor. (This fixes 1.)
            object_for_each_anchor[anchor_for_each_object] = torch.LongTensor(
                range(n_objects)).to(self.cfg.device)

            # To ensure these anchors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_anchor[anchor_for_each_object] = 1.

            # Labels for each anchor
            # (n_anchors)
            label_for_each_anchor = labels[i][object_for_each_anchor]
            # Set anchors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_anchor[overlap_for_each_anchor <
                                  self.threshold] = 0  # (n_anchors)

            # Store
            true_classes[i] = label_for_each_anchor

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcywh_to_gcxgcywh(xyxy_to_cxcywh(
                boxes[i][object_for_each_anchor]), self.anchors_cxcywh)  # (n_anchors, 4)

        # Identify anchors that are positive (object/non-background)
        positive_anchors = true_classes != 0  # (batch_size, n_anchors)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) anchors
        loc_loss = self.smooth_l1(
            predicted_locs[positive_anchors], true_locs[positive_anchors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & n_anchors)
        # So, if predicted_locs has the shape (N, n_anchors, 4), predicted_locs[positive_anchors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive anchors and the most difficult (hardest) negative anchors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative anchors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative anchors per image
        n_positives = positive_anchors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all anchors
        conf_loss_all = self.cross_entropy(
            predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * n_anchors)
        conf_loss_all = conf_loss_all.view(
            batch_size, n_anchors)  # (N, n_anchors)

        # We already know which anchors are positive
        conf_loss_pos = conf_loss_all[positive_anchors]  # (sum(n_positives))

        # Next, find which anchors are hard-negative
        # To do this, sort ONLY negative anchors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, n_anchors)
        # (N, n_anchors), positive anchors are ignored (never in top n_hard_negatives)
        conf_loss_neg[positive_anchors] = 0.
        # (N, n_anchors), sorted by decreasing hardness
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)
        hardness_ranks = torch.LongTensor(range(n_anchors)).unsqueeze(
            0).expand_as(conf_loss_neg).to(self.cfg.device)  # (N, n_anchors)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(
            1)  # (N, n_anchors)
        # (sum(n_hard_negatives))
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]

        # As in the paper, averaged over positive anchors only, although computed over both positive and hard-negative anchors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()
                     ) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS

        return conf_loss + self.alpha * loc_loss
