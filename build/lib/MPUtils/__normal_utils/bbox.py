def xywh2xyxy(xywh):
    xyxy = xywh.copy()
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2]/2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3]/2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2]/2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3]/2
    return xyxy

def xyxy2xywh(xyxy):
    xywh = xyxy.copy()
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return xywh

def inv_normalize_box(bboxes, w, h):
    bboxes[:, 0] *= w
    bboxes[:, 1] *= h
    bboxes[:, 2] *= w
    bboxes[:, 3] *= h
    return bboxes

def normalize_box(bboxes, w, h):
    bboxes[:, 0] /= w
    bboxes[:, 1] /= h
    bboxes[:, 2] /= w
    bboxes[:, 3] /= h
    return bboxes