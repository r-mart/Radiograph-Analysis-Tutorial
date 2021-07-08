import numpy as np


def xyxy_to_xywh(a):
    """ Converts a single box from XYXY to XYWH format 

    a: box in XYXY coordinates
    """

    x1 = min(a[0], a[2])
    y1 = min(a[1], a[3])
    x2 = max(a[0], a[2])
    y2 = max(a[1], a[3])
    return [x1, y1, x2-x1+1, y2-y1+1]


def xywh_to_xyxy(a):
    """ Converts a single box from XYWH to XYXY format 

    a: box in XYWH coordinates
    """

    return [a[0], a[1], a[0]+a[2]-1, a[1]+a[3]-1]


def cxcywh_to_xyxy(boxes):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return np.concatenate([boxes[:, :2] - (boxes[:, 2:] / 2),  # x_min, y_min
                           boxes[:, :2] + (boxes[:, 2:] / 2)], axis=1)  # x_max, y_max
