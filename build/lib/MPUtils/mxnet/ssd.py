from .transform import TrainTransform, ValidTransform
from gluoncv.data.batchify import Tuple, Stack, Pad
from mxnet.gluon.data import DataLoader
from mxnet import autograd, nd, gluon
from mxnet.gluon import nn
from .mutils import tri_data_loader
import numpy as np
import mxnet as mx

from gluoncv import model_zoo as gmodel_zoo

"""
    data prepared:
        ssd_data_loader: func

    SSD Detector:
        FeatureNet
        MPUSSDAnchorGenerator: class
        MPUSSDTargetGenerator: class
        MPUSSD: class
    
    debug:
        get_center_ids: func
        get_objcet_id: func
"""

"""
    1. data prepare
"""
def ssd_data_loader(batch_size, width, height, mean, std, anchors=None, net=None, num_workers=0, 
                    train_dataset=None, valid_dataset=None, valid_train_dataset=None, batchify_fns=None, train_transform_kwargs={}, train_shuffle=True):
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
    if anchors is None:
        batchify_fns['train'] = Tuple(get_batchify([Stack(), Pad(pad_val=-1)] , len(train_dataset[0])))
    
    return tri_data_loader(batch_size, transform_train, transform_valid, num_workers, train_dataset=train_dataset,
                       valid_dataset=valid_dataset, valid_train_dataset=valid_train_dataset, batchify_fns=batchify_fns, train_shuffle=train_shuffle)


"""
    2. SSD Detector
"""
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
                 offsets=(0.5, 0.5), clip=False, generate_type='default', use_whsize=False, verbose=False, **kwargs):
        """
            generate_type: 'default' means anchor type is s[0] compose with r[i] + r[0] compose with s[1], s = (min_size, sqrt(min_size * max_size)), will use anchor_num=m+n-1=2+n-1=n+1
                           'paper' mean anchor type is s[0] compose with r[i] + r[0] compose with all s[i], anchor_num=m+n-1
                           'pair' means anchor_num=m * n
            use_whsize: use_whsize=True, sizes is [(w1, h1), (w2, h2), ...], ratios and genetate_type is not meaning.
        """
        super(MPUSSDAnchorGenerator, self).__init__(**kwargs)
        assert len(im_size) == 2
        self.verbose = verbose
        self._im_size = im_size
        self._clip = clip
        self.generate_type = generate_type.lower()
        if self.generate_type=='default' and not use_whsize: 
            assert len(sizes) == 2, "when generate_type='default', SSD requires sizes to be (size_min, size_max)"
            self._sizes = (sizes[0], np.sqrt(sizes[0] * sizes[1]))
        else: self._sizes = sizes[:]
        self.use_whsize = use_whsize
        self._ratios = ratios
        anchors = self._generate_anchors(self._sizes, self._ratios, step, alloc_size, offsets)
        self.anchors = self.params.get_constant('anchor_%d'%(index), anchors)

    def _get_wh(self, sizes, ratios):
        if self.use_whsize: return sizes
        whsize = []
        if self.generate_type == 'pair':
            for s in sizes:
                for r in ratios:
                    sr = np.sqrt(r)
                    w = s * sr
                    h = s / sr
                    whsize.append([w, h])
        else:
            r = ratios[0]
            for s in sizes:
                sr = np.sqrt(r)
                w = s * sr
                h = s / sr
                whsize.append([w, h])

            # size = sizes[0], ratio = ...
            for r in ratios[1:]: # s[0] -> all r[1:]
                sr = np.sqrt(r)
                w = sizes[0] * sr
                h = sizes[0] / sr
                whsize.append([w, h])
        return whsize
            
    def _generate_anchors(self, sizes, ratios, step, alloc_size, offsets):
        """Generate anchors for once.""" 
        whsize = self._get_wh(sizes, ratios)
        anchors = []
        for i in range(alloc_size[0]):
            for j in range(alloc_size[1]):
                cy = (i + offsets[0]) * step
                cx = (j + offsets[1]) * step
                for w, h in whsize:
                    anchors.append([cx, cy, w, h])
        return np.array(anchors).reshape(1, 1, alloc_size[0], alloc_size[1], -1)

    @property
    def num_depth(self):
        """Number of anchors at each pixel."""
        if self.generate_type == 'pair':
            return len(self._sizes) * len(self._ratios)
        return len(self._sizes) + len(self._ratios) - 1

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, anchors, concat=True):
        if self.verbose: print("feature shape", x.shape, "alloc anchor", anchors.shape)
        a = F.slice_like(anchors, x * 0, axes=(2, 3))
        a = a.reshape((1, -1, 4))
        if self._clip:
            cx, cy, cw, ch = a.split(axis=-1, num_outputs=4)
            H, W = self._im_size
            a = [cx.clip(0, W), cy.clip(0, H), cw.clip(0, W), ch.clip(0, H)]
            if concat: a = F.concat(*a, dim=-1)
        return a.reshape((1, -1, 4))

from mxnet import nd
from mxnet.gluon import Block
from gluoncv.nn.matcher import CompositeMatcher, BipartiteMatcher, MaximumMatcher
from gluoncv.nn.sampler import OHEMSampler, NaiveSampler
from gluoncv.nn.coder import MultiClassEncoder, NormalizedBoxCenterEncoder
from gluoncv.nn.bbox import BBoxCenterToCorner


class MPUSSDTargetGenerator(Block):
    """Training targets generator for Single-shot Object Detection.

    Parameters
    ----------
    iou_thresh : float
        IOU overlap threshold for maximum matching, default is 0.5.
    neg_thresh : float
        IOU overlap threshold for negative mining, default is 0.5.
    negative_mining_ratio : float
        Ratio of hard vs positive for negative mining.
    stds : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
        Std value to be divided from encoded values.
    """
    def __init__(self, iou_thresh=0.5, neg_thresh=0.5, negative_mining_ratio=3,
                 stds=(0.1, 0.1, 0.2, 0.2), **kwargs):
        super(MPUSSDTargetGenerator, self).__init__(**kwargs)
        self._matcher = CompositeMatcher([BipartiteMatcher(), MaximumMatcher(iou_thresh)])
        if negative_mining_ratio > 0:
            self._sampler = OHEMSampler(negative_mining_ratio, thresh=neg_thresh)
            self._use_negative_sampling = True
        else:
            self._sampler = NaiveSampler()
            self._use_negative_sampling = False
        self._cls_encoder = MultiClassEncoder()
        self._box_encoder = NormalizedBoxCenterEncoder(stds=stds)
        self._center_to_corner = BBoxCenterToCorner(split=False)

    # pylint: disable=arguments-differ
    def forward(self, anchors, cls_preds, gt_boxes, gt_ids):
        """Generate training targets."""
        anchors = self._center_to_corner(anchors.reshape((-1, 4)))
        ious = nd.transpose(nd.contrib.box_iou(anchors, gt_boxes), (1, 0, 2))
        matches = self._matcher(ious)
        if self._use_negative_sampling:
            samples = self._sampler(matches, cls_preds, ious)
        else:
            samples = self._sampler(matches)
        cls_targets = self._cls_encoder(samples, matches, gt_ids)
        box_targets, box_masks = self._box_encoder(samples, matches, anchors, gt_boxes)
        return cls_targets, box_targets, box_masks


class MPUSSD(nn.HybridBlock):
    def __init__(self, im_size, features, sizes, ratios, steps, classes,
                 stds=(0.1, 0.1, 0.2, 0.2), nms_thresh=0.45, nms_topk=400, post_nms=100,
                 anchor_alloc_size=128, ctx=mx.cpu(), verbose=False,
                 anchor_generate_type='default', anchor_use_whsize=False, **kwargs):
        super(MPUSSD, self).__init__(**kwargs)
        num_layers = len(list(sizes))
        if not anchor_use_whsize:
            assert len(sizes) == len(ratios), \
                "Mismatched (number of layers) vs (sizes) vs (ratios): {}, {}, {}".format(
                    num_layers, len(sizes), len(ratios))
        assert num_layers > 0, "SSD require at least one layer, suggest multiple."
        self._num_layers = num_layers
        self.classes = classes
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        self.verbose = verbose

        with self.name_scope():
            self.features = features # VGGDownSampleNet(ctx, len(sizes), verbose=True)
            self.class_predictors = nn.HybridSequential()
            self.box_predictors = nn.HybridSequential()
            self.anchor_generators = nn.HybridSequential()
            asz = anchor_alloc_size
            for i, s, r, st in zip(range(num_layers), sizes, ratios, steps):
                anchor_generator = MPUSSDAnchorGenerator(i, im_size, s, r, st, (asz, asz), verbose=verbose,
                                                 generate_type=anchor_generate_type, use_whsize=anchor_use_whsize)
                self.anchor_generators.add(anchor_generator)
                asz = max(asz // 2, 16)  # pre-compute larger than 16x16 anchor map
                num_anchors = anchor_generator.num_depth
                self.class_predictors.add(gmodel_zoo.ssd.ConvPredictor(num_anchors * (len(self.classes) + 1)))
                self.box_predictors.add(gmodel_zoo.ssd.ConvPredictor(num_anchors * 4))
            self.bbox_decoder = gmodel_zoo.ssd.NormalizedBoxCenterDecoder(stds)
            self.cls_decoder = gmodel_zoo.ssd.MultiPerClassDecoder(len(self.classes) + 1, thresh=0.01)
        
        if hasattr(features, 'pretrained_block') and hasattr(features, 'other_block'):
            self.pretrained_block = nn.HybridSequential()
            self.pretrained_block.add(features.pretrained_block)
            self.other_block = nn.HybridSequential()
            self.other_block.add(features.other_block)
            self.other_block.add(self.class_predictors, self.box_predictors)

    @property
    def num_classes(self):
        return len(self.classes)

    def set_nms(self, nms_thresh=0.45, nms_topk=400, post_nms=100):
        self._clear_cached_op()
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        features = self.features(x)
        cls_preds = [F.flatten(F.transpose(cp(feat), (0, 2, 3, 1)))
                     for feat, cp in zip(features, self.class_predictors)]
        box_preds = [F.flatten(F.transpose(bp(feat), (0, 2, 3, 1)))
                     for feat, bp in zip(features, self.box_predictors)]
        anchors = [F.reshape(ag(feat), shape=(1, -1))
                   for feat, ag in zip(features, self.anchor_generators)]
        
        if self.verbose:
            for i, (feature, anchor) in enumerate(zip(features, anchors)):
                print("predict scale ", i, feature.shape, "with", anchor.reshape((0, -1, 4)).shape, 'anchors')
        
        cls_preds = F.concat(*cls_preds, dim=1).reshape((0, -1, self.num_classes + 1))
        box_preds = F.concat(*box_preds, dim=1).reshape((0, -1, 4))
        anchors = F.concat(*anchors, dim=1).reshape((1, -1, 4))
        if autograd.is_training():
            return [cls_preds, box_preds, anchors]
        bboxes = self.bbox_decoder(box_preds, anchors)
        cls_ids, scores = self.cls_decoder(F.softmax(cls_preds, axis=-1))
        results = []
        for i in range(self.num_classes):
            cls_id = cls_ids.slice_axis(axis=-1, begin=i, end=i+1)
            score = scores.slice_axis(axis=-1, begin=i, end=i+1)
            # per class results
            per_result = F.concat(*[cls_id, score, bboxes], dim=-1)
            results.append(per_result)
        result = F.concat(*results, dim=1)
        if self.nms_thresh > 0 and self.nms_thresh < 1:
            result = F.contrib.box_nms(
                result, overlap_thresh=self.nms_thresh, topk=self.nms_topk, valid_thresh=0.01,
                id_index=0, score_index=1, coord_start=2, force_suppress=False)
            if self.post_nms > 0:
                result = result.slice_axis(axis=1, begin=0, end=self.post_nms)
        ids = F.slice_axis(result, axis=2, begin=0, end=1)
        scores = F.slice_axis(result, axis=2, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=2, begin=2, end=6)
        return ids, scores, bboxes

    def reset_class(self, classes):
        self._clear_cached_op()
        self.classes = classes
        # replace class predictors
        with self.name_scope():
            class_predictors = nn.HybridSequential(prefix=self.class_predictors.prefix)
            for i, ag in zip(range(len(self.class_predictors)), self.anchor_generators):
                prefix = self.class_predictors[i].prefix
                new_cp = gmodel_zoo.ssd.ConvPredictor(ag.num_depth * (self.num_classes + 1), prefix=prefix)
                new_cp.collect_params().initialize()
                class_predictors.add(new_cp)
            self.class_predictors = class_predictors

"""
     3. for statistic and debug
"""
def get_anchor_targets(labels, anchors, target_generator=MPUSSDTargetGenerator(negative_mining_ratio=-1)):
    """
        not like used in TrainTransform, get_anchor_targets input wiht a bacth's data, not a image's data.

        labels: nd.array or np.array, shape=(batch_size, n, t),(t>5), [[[xmin, ymin, xmax, ymax, label,...], ...]], label can use -1 pad
        anchors: nd.array, shape=(1, ac, 4) or (ac, 4), anchors generate by SSDAnchorGeneraor, ac means anchor count
        target_generator: anchor target generator, MPUSSDTargetGenerator or SSDTargetGenerator
    """
    if isinstance(labels, type(nd.array([0]))):
        labels = labels.asnumpy()
    #print(gt_bboxes.shape, gt_ids.shape, anchors.shape)
    batch_cls_targets, batch_box_targets = [], []
    for label in labels:
        label = label[label[:, 4] >= 0]
        gt_bboxes = mx.nd.array(label[np.newaxis, :, :4])
        gt_ids = mx.nd.array(label[np.newaxis, :, 4:5])
        cls_targets, box_targets, _ = target_generator(anchors, None, gt_bboxes, gt_ids)
        batch_cls_targets.append(cls_targets)
        batch_box_targets.append(box_targets)
    return nd.concat(*batch_cls_targets, dim=0), nd.concat(*batch_box_targets, dim=0)


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

def get_objects(bbox_t, EPS = 1):
    """
        bbox_t: valid target box that anchor after decode  (label)
        
        get object id for each valid target box
        return: 
            obj_boxes: [[obj1_x1, obj1_y1, obj1_x2, obj1_y2]...], shape=(num_obj, 4)
            obj_t: [box1_object_id, box2_object_id ....], shape=(num_anchors,)
    """
    obj_id = 0
    obj_t = np.array([-1]* len(bbox_t))
    obj_boxes = []
    for i, (oid, box) in enumerate(zip(obj_t, bbox_t)):  # count repeat box numbers, and give euqaled box a same obj_id.
        if oid < 0:  # not compared yet.
            error = np.mean(np.abs(bbox_t - box), axis=-1)
            obj_t[error < EPS] = obj_id
            obj_boxes.append(box)
            obj_id += 1
    return np.array(obj_boxes), obj_t

def get_feat_size(net, data_loader):
    for i, batch in enumerate(data_loader):  
        x = batch[0]
        feat_size = []
        for feat in net.features(x):
            feat_size.append(feat.shape[2] * feat.shape[3])
        break
    return feat_size

def get_anchors(net, data_loader):
    for batch in data_loader:
        with autograd.train_mode():
            _, _, anchors = net(batch[0])
        break
    return anchors
