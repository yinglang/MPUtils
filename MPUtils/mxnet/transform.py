from mxnet import nd
from random import uniform, randint
import math

class RandomPadCrop:
    def __init__(self, pad):
        """
        pad: tuple of (lh, rh, lw, rw) of pading length 
        """
        self.pad = pad
        self.random_range = nd.array([pad[0]+pad[1], pad[2]+pad[3]])

    def __call__(self, data):
        pad, h, w = self.pad, data.shape[1], data.shape[2]
        data = data.expand_dims(axis=0).pad(mode="constant", constant_value=0,
                                            pad_width=(0, 0, 0, 0, pad[0], pad[1], pad[2], pad[3]))
        x0, y0 = (nd.random.uniform(shape=(2,)) * self.random_range).astype('uint8')
        x0, y0 = x0.asscalar(), y0.asscalar()
        return data[0, :, x0:x0+h, y0:y0+w]
    
class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
        
        used after transforms.ToTensor and transforms.Normalize
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):
        if uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.shape[1] * img.shape[2]
       
            target_area = uniform(self.sl, self.sh) * area
            aspect_ratio = uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[2] and h < img.shape[1]:
                x1 = randint(0, img.shape[1] - h)
                y1 = randint(0, img.shape[2] - w)
                img[:, x1:x1+h, y1:y1+w] = 0
                return img

        return img  

# from gluoncv.data.transforms.presets import experimental
# import gluoncv.data.transforms.presets.experimental as experimental
from gluoncv.data.transforms import presets
import numpy as np
from mxnet import nd
import mxnet as mx
import random

def random_crop_with_constraints(bbox, size, min_scale=0.3, max_scale=1,
                                 max_aspect_ratio=2, constraints=None,
                                 max_trial=50):
    """Crop an image randomly with bounding box constraints.

    This data augmentation is used in training of
    Single Shot Multibox Detector [#]_. More details can be found in
    data augmentation section of the original paper.
    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    size : tuple
        Tuple of length 2 of image shape as (width, height).
    min_scale : float
        The minimum ratio between a cropped region and the original image.
        The default value is :obj:`0.3`.
    max_scale : float
        The maximum ratio between a cropped region and the original image.
        The default value is :obj:`1`.
    max_aspect_ratio : float
        The maximum aspect ratio of cropped region.
        The default value is :obj:`2`.
    constraints : iterable of tuples
        An iterable of constraints.
        Each constraint should be :obj:`(min_iou, max_iou)` format.
        If means no constraint if set :obj:`min_iou` or :obj:`max_iou` to :obj:`None`.
        If this argument defaults to :obj:`None`, :obj:`((0.1, None), (0.3, None),
        (0.5, None), (0.7, None), (0.9, None), (None, 1))` will be used.
    max_trial : int
        Maximum number of trials for each constraint before exit no matter what.

    Returns
    -------
    numpy.ndarray
        Cropped bounding boxes with shape :obj:`(M, 4+)` where M <= N.
    tuple
        Tuple of length 4 as (x_offset, y_offset, new_width, new_height).

    """
    # default params in paper
    if constraints is None:
        constraints = (
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, 1),
        )

    if len(bbox) == 0:
        constraints = []

    w, h = size

    candidates = [(0, 0, w, h)]
    for min_iou, max_iou in constraints:
        min_iou = -np.inf if min_iou is None else min_iou
        max_iou = np.inf if max_iou is None else max_iou

        for _ in range(max_trial):
            scale = random.uniform(min_scale, max_scale)
            aspect_ratio = random.uniform(
                max(1 / max_aspect_ratio, scale * scale),
                min(max_aspect_ratio, 1 / (scale * scale)))
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))

            crop_t = random.randrange(h - crop_h)
            crop_l = random.randrange(w - crop_w)
            crop_bb = np.array((crop_l, crop_t, crop_l + crop_w, crop_t + crop_h))

            # below 3 line add by hui
            if len(bbox) == 0:
                candidates.append((left, top, right-left, bottom-top))
                break
            
            iou = presets.ssd.experimental.bbox.bbox_iou(bbox, crop_bb[np.newaxis])
            if min_iou <= iou.min() and iou.max() <= max_iou:
                top, bottom = crop_t, crop_t + crop_h
                left, right = crop_l, crop_l + crop_w
                candidates.append((left, top, right-left, bottom-top))
                break

    # random select one
    while candidates:
        crop = candidates.pop(np.random.randint(0, len(candidates)))
        if len(bbox) == 0:
            return bbox, crop
        
        new_bbox = presets.ssd.experimental.bbox.bbox_crop(bbox, crop, allow_outside_center=False)
        if new_bbox.size < 1:
            continue
        new_crop = (crop[0], crop[1], crop[2], crop[3])
        return new_bbox, new_crop
    return bbox, (0, 0, w, h)

def returned_value(res, args, kwargs):
    if len(args) > 0: res.append(args)
    if len(kwargs) > 0: res.append(kwargs)
    return res

class TrainTransform(presets.ssd.SSDDefaultTrainTransform):
    """
        reference to SSDDefaultTrainTransform
    """
    def __init__(self, width, height, anchors=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), iou_thresh=0.5, box_norm=(0.1, 0.1, 0.2, 0.2),
                 color_distort_kwargs={}, random_expand_kwargs={}, random_crop_kwargs={},
                 **kwargs):
        """
            ?? presets.ssd.experimental.image.random_color_distort
            ?? presets.ssd.timage.random_expand
            ?? presets.ssd.experimental.bbox.random_crop_with_constraints
        """
        super(TrainTransform, self).__init__(width, height, anchors, mean,
                 std, iou_thresh, box_norm, **kwargs)
        self.color_distort_kwargs = color_distort_kwargs
        self.random_expand_kwargs = random_expand_kwargs
        self.random_crop_kwargs = random_crop_kwargs

    def __call__(self, src, label, *args, **kwargs):
        """Apply transform to training image/label."""
        # random color jittering
        img = presets.ssd.experimental.image.random_color_distort(src, **self.color_distort_kwargs)

        # random expansion with prob 0.5
        if np.random.uniform(0, 1) > 0.5:
            img, expand = presets.ssd.timage.random_expand(img, fill=[m * 255 for m in self._mean], **self.random_expand_kwargs)
            if label.shape[0] > 0: 
                bbox = presets.ssd.tbbox.translate(label, x_offset=expand[0], y_offset=expand[1])
            else:
                bbox = label
        else:
            img, bbox = img, label
        # random cropping
        h, w, _ = img.shape
        bbox, crop = random_crop_with_constraints(bbox, (w, h), **self.random_crop_kwargs)
        x0, y0, w, h = crop
        img = mx.image.fixed_crop(img, x0, y0, w, h)

        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img = presets.ssd.timage.imresize(img, self._width, self._height, interp=interp)
        if len(bbox) > 0: bbox = presets.ssd.tbbox.resize(bbox, (w, h), (self._width, self._height))

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = presets.ssd.timage.random_flip(img, px=0.5)
        if len(bbox) > 0: bbox = presets.ssd.tbbox.flip(bbox, (w, h), flip_x=flips[0])

        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        if len(bbox) == 0: bbox = np.array([[-1] * 6])
        if self._anchors is None:
            return returned_value([img, bbox.astype(img.dtype)], args, kwargs)

        # generate training target so cpu workers can help reduce the workload on gpu
        gt_bboxes = mx.nd.array(bbox[np.newaxis, :, :4])
        gt_ids = mx.nd.array(bbox[np.newaxis, :, 4:5])
        cls_targets, box_targets, _ = self._target_generator(
            self._anchors, None, gt_bboxes, gt_ids)
        return returned_value([img, cls_targets[0], box_targets[0]], args, kwargs)

    
class ValidTransform(object):
    """Default SSD validation transform.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].

    """
    def __init__(self, width, height, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._width = width
        self._height = height
        self._mean = mean
        self._std = std

    def __call__(self, src, label, *args, **kwargs):
        """Apply transform to validation image/label."""
        # resize
        h, w, _ = src.shape
        img = presets.ssd.timage.imresize(src, self._width, self._height, interp=9)
        if label.shape[0] > 0:
            bbox = presets.ssd.tbbox.resize(label, in_size=(w, h), out_size=(self._width, self._height))
        else: bbox=np.array([[-1]*6])

        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return returned_value([img, bbox.astype(img.dtype)], args, kwargs)

