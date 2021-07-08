import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccuracyMeter(object):
    """Computes and stores the accuracy score"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.acc = 0
        self.correct_count = 0
        self.total_count = 0

    def update(self, n_correct, n):
        self.correct_count += n_correct
        self.total_count += n
        self.acc = self.correct_count / self.total_count


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def cxcywh_to_xyxy(boxes):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([boxes[:, :2] - (boxes[:, 2:] / 2),  # x_min, y_min
                      boxes[:, :2] + (boxes[:, 2:] / 2)], 1)  # x_max, y_max


def xyxy_to_cxcywh(boxes):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).

    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(boxes[:, 2:] + boxes[:, :2]) / 2,  # c_x, c_y
                      boxes[:, 2:] - boxes[:, :2]], 1)  # w, h


def cxcywh_to_gcxgcywh(boxes, anchors):
    """
    Encode bounding boxes in center-size XYWH form w.r.t. the corresponding anchor boxes (that are in center-size form).

    For the center coordinates, find the offset with respect to the anchor box, and scale by the size of the anchor box.
    For the size coordinates, scale by the size of the anchor box, and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    :param boxes: bounding boxes in center-size coordinates, a tensor of size (n_anchors, 4)
    :param anchors: prior boxes with respect to which the encoding must be performed, a tensor of size (n_anchors, 4)
    :return: encoded bounding boxes, a tensor of size (n_anchors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(boxes[:, :2] - anchors[:, :2]) / (anchors[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(boxes[:, 2:] / anchors[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcywh_to_cxcywh(boxes, anchors):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned in the encoding function.

    They are decoded into center-size XYWH coordinates.    

    :param boxes: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param anchors: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([boxes[:, :2] * anchors[:, 2:] / 10 + anchors[:, :2],  # c_x, c_y
                      torch.exp(boxes[:, 2:] / 5) * anchors[:, 2:]], 1)  # w, h


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(
        1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(
        1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(
        upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * \
        (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * \
        (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(
        1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


def boxes_xyxy_abs_to_rel(bboxes, img_shape):
    """ Converts boxes from absolute to relative coordinates 

    bboxes: array of boxes in XYXY absolute coordinates
    img_shape: tuple (height, width)
    """

    h, w = img_shape
    bboxes[:, ::2] = bboxes[:, ::2] / w
    bboxes[:, 1::2] = bboxes[:, 1::2] / h

    return bboxes


def boxes_xyxy_rel_to_abs(bboxes, img_shape):
    """ Converts boxes from relative to absolute coordinates 

    bboxes: array of boxes in XYXY relative coordinates
    img_shape: tuple (height, width)
    """

    h, w = img_shape
    bboxes[:, ::2] = bboxes[:, ::2] * w
    bboxes[:, 1::2] = bboxes[:, 1::2] * h

    return bboxes
