from .transform import TrainTransform, ValidTransform
from gluoncv.data.batchify import Tuple, Stack, Pad
from mxnet.gluon.data import DataLoader
from mxnet import autograd, nd, gluon
from .mutils import tri_data_loader
import numpy as np

"""
    ssd_data_loader: func
    MPUSSDAnchorGenerator: class
    get_center_ids: func
    get_objcet_id: func
"""
    
def ssd_data_loader(batch_size, width, height, mean, std, anchors=None, net=None, num_workers=0, 
                    train_dataset=None, valid_dataset=None, valid_train=False, batchify_fns=None, train_transform_kwargs={}):
    """
        train_transform_kwargs = {
            'color_distort_kwargs': {'brightness_delta': 32, 'contrast_low': 0.5, 'contrast_high': 1.5,
                                 'saturation_low': 0.5, 'saturation_high': 1.5, 'hue_delta': 18},
            'random_expand_kwargs': {'max_ratio': 1.5, 'keep_ratio': True},
            'random_crop_kwargs': {'min_scale': 0.3, 'max_scale': 1, 'max_aspect_ratio': 1.0, 
                                 'constraints': None, 'max_trial': 50}
        }
    """
    def get_batchify(fns, batch_dim, pad=Stack()):
        """ use Stack() to pad fns length to batch_dim.  """
        return fns + [pad] * (batch_dim-len(fns))
    
    def get_anchors(net):
        if net is None: return None
        for v in net.collect_params().values():
            ctx = v.data().context
            break
        x = nd.zeros(shape=(1, 3, height, width)).as_in_context(ctx)
        with autograd.train_mode():
            cls_preds, box_preds, anchors = net(x)
        return anchors
    # 1. anchors    
    if anchors is not None:
        if not isinstance(anchors, mx.ndarray.ndarray.NDArray) or len(anchors.shape) != 3 or anchors.shape[-1] != 4:
            raise ValueError('anchors is not right format, must be ndarray and shape is (b, ac, 4)')
    else:
        anchors = get_anchors(net)
    
    # 2. transform
    transform_train = TrainTransform(width, height, anchors, mean=mean, std=std, **train_transform_kwargs)
    transform_valid = ValidTransform(width, height, mean=mean, std=std)
    
    # 3. batchify_fn
    if batchify_fns is None: 
        batchify_fns = {'train': Tuple(get_batchify([Stack(), Stack(), Stack()], len(train_dataset[0])+1)),
                        'valid': Tuple(get_batchify([Stack(), Pad(pad_val=-1)] , len(valid_dataset[0]))),
                        'valid_train': Tuple(get_batchify([Stack(), Pad(pad_val=-1)], len(train_dataset[0])))}
    elif 'valid_train' not in batchify_fns:
        batchify_fns['valid_train'] = Tuple(get_batchify([Stack(), Pad(pad_val=-1)], len(train_dataset[0])))
    
    return tri_data_loader(batch_size, transform_train, transform_valid, num_workers, train_dataset=train_dataset,
                       valid_dataset=valid_dataset, valid_train=valid_train, batchify_fns=batchify_fns)


class MPUSSDAnchorGenerator(gluon.HybridBlock):
    """Bounding box anchor generator for Single-shot Object Detection.
    
    ** changed form gluoncv.model_zoo.ssd.SSDAnchorGenerator **

    Parameters
    ----------
    index : int
        Index of this generator in SSD models, this is required for naming.
    sizes : iterable of floats
        Sizes of anchor boxes.
    ratios : iterable of floats
        Aspect ratios of anchor boxes.
    step : int or float
        Step size of anchor boxes.
    alloc_size : tuple of int
        Allocate size for the anchor boxes as (H, W).
        Usually we generate enough anchors for large feature map, e.g. 128x128.
        Later in inference we can have variable input sizes,
        at which time we can crop corresponding anchors from this large
        anchor map so we can skip re-generating anchors for each input.
    offsets : tuple of float
        Center offsets of anchor boxes as (h, w) in range(0, 1).
        
    use_default_size: defalut size means len(sizes)=2, used_sizes = [sizes[0], sqrt(size[0]*szie[1])]

    """
    def __init__(self, index, im_size, sizes, ratios, step, alloc_size=(128, 128),
                 offsets=(0.5, 0.5), clip=False, use_default_size=True, **kwargs):
        super(MPUSSDAnchorGenerator, self).__init__(**kwargs)
        assert len(im_size) == 2
        self._im_size = im_size
        self._clip = clip
        if use_default_size: self._sizes = (sizes[0], np.sqrt(sizes[0] * sizes[1]))
        else: self._sizes = sizes[:]
        self._ratios = ratios
        anchors = self._generate_anchors(self._sizes, self._ratios, step, alloc_size, offsets)
        self.anchors = self.params.get_constant('anchor_%d'%(index), anchors)

    def _generate_anchors(self, sizes, ratios, step, alloc_size, offsets):
        """Generate anchors for once."""
        assert len(sizes) == 2, "SSD requires sizes to be (size_min, size_max)"
        anchors = []
        for i in range(alloc_size[0]):
            for j in range(alloc_size[1]):
                cy = (i + offsets[0]) * step
                cx = (j + offsets[1]) * step
                # ratio = ratios[0], size = size_min or sqrt(size_min * size_max)
                r = ratios[0]
                for s in sizes:
                    sr = np.sqrt(r)
                    w = s * sr
                    h = s / sr
                    anchors.append([cx, cy, w, h])

                # size = sizes[0], ratio = ...
                for r in ratios[1:]: # s[0] -> all r[1:]
                    sr = np.sqrt(r)
                    w = sizes[0] * sr
                    h = sizes[0] / sr
                    anchors.append([cx, cy, w, h])
        return np.array(anchors).reshape(1, 1, alloc_size[0], alloc_size[1], -1)

    @property
    def num_depth(self):
        """Number of anchors at each pixel."""
        return len(self._sizes) + len(self._ratios) - 1

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, anchors, concat=True):
        a = F.slice_like(anchors, x * 0, axes=(2, 3))
        a = a.reshape((1, -1, 4))
        if self._clip:
            cx, cy, cw, ch = a.split(axis=-1, num_outputs=4)
            H, W = self._im_size
            a = [cx.clip(0, W), cy.clip(0, H), cw.clip(0, W), ch.clip(0, H)]
            if concat: a = F.concat(*a, dim=-1)
        return a.reshape((1, -1, 4))

    
def get_center_ids(anchors, anchor_generators, feat_size=None):
    """
        anchors: un-concat anchors when feat_size is None or len==0, or use feat_size specify every feat's size(w*h)
        anchor_generators: list(SSDAnchorGenerator)
        feat_size: list(int), specify every feat's size(w*h)
        
        get center_id to every anchors
        return: [anchor1_center_id, anchor2_center_id, .....]
    """
    if feat_size is not None and len(feat_size) > 0:
        all_anchors = anchors.shape[1]
        begin = 0
        for fs, ag in zip(feat_size, anchor_generators):
            anchors.slice_axis(axis=1, begin=begin, end=begin+fs*ag.num_depth)
            begin += fs*ag.num_depth
        assert all_anchors == begin  # feat_size and anchor_depth must match anchor size 
        
    muti_scale_center_ids = []
    center_id = 0
    for i in range(len(anchors)):
        # one center point can generate how namy equal-area box = (size+aspect-1)
        center_depth = anchor_generators[i].num_depth  
        anchors[i] = anchors[i].reshape((-1, 4))
        center_count = int(anchors[i].shape[0] / center_depth) # how many center == feature map W *H * center_depth
        center_ids = np.array(list(range(center_id, center_id+center_count)))
        center_ids = center_ids.reshape((-1, 1)).repeat(center_depth, axis=1).reshape((-1,))
        muti_scale_center_ids.append(center_ids)
        center_id += center_count
    return np.concatenate(muti_scale_center_ids)

def get_objcet_id(bbox_t, EPS = 1):
    """
        bbox_t: valid target box that anchor after decode  (label)
        
        get object id for each valid target box
        return: [box1_object_id, box2_object_id ....]
    """
    obj_id = 0
    obj_t = np.array([-1]* len(bbox_t))
    for i, (oid, box) in enumerate(zip(obj_t, bbox_t)):  # count repeat box numbers, and give euqaled box a same obj_id.
        if oid < 0:  # not compared
            error = np.mean(np.abs(bbox_t - box), axis=-1)
            obj_t[error < EPS] = obj_id
            obj_id += 1
    return obj_t