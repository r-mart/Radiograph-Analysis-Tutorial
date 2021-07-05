
def boxes_xyxy_abs_to_rel(bboxes, img_shape):
    """ Converts boxes from absolute to relative coordinates 

    bboxes: array of boxes in XYXY absolute coordinates
    img_shape: tuple (height, width)
    """

    h, w = img_shape
    bboxes[:, ::2] = bboxes[:, ::2] / w
    bboxes[:, 1::2] = bboxes[:, 1::2] / h

    return bboxes


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
