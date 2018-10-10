"""
  normal util, later will be merge to utils.py
"""
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

"""
    1. get recomand box size, aspect ratio for ssd detector.
"""

def euqal_part_point(data, bins=5, TYPE=None, geometric=False):
    """
        寻找bins等分点，find bins equal part point
        data: np.array, shape=(m,)
        bins: how many part to split data
        TYPE: optinal in ['split', 'median', 'mean', None], None for output all
    return:
        split_point: (part_min, part_max) / 2, for every part
        meduim_point: meduim point, for every part
        mean_point: mean(part), for every part
    """
    idxs = {'split': 0, 'median': 1, 'mean': 2, None: None}
    idx = idxs[TYPE]
    n = bins
    data = data.copy()
    data.sort()
    if geometric: data=np.log(data)
    l = len(data)
    bound_idx = [0] + list(np.array(list(range(l//n, l+1, l // (n)))) - 1)
    bound_point = data[bound_idx]
    split_point = (bound_point[:-1] + bound_point[1:]) / 2
    mean_point = data[:l //  n * n].reshape((n, l//n)).mean(axis=1)
    meduim_point = data[list(range(l//n//2, l, l // n))]
    res = (split_point, meduim_point, mean_point)
    if geometric: res = (np.exp(split_point), np.exp(meduim_point), np.exp(mean_point))
    if idx is None:
        return res
    else:
        return res[idx]
find_split_point = euqal_part_point

def recomand_box_aspect_point(WHSAs, aspect_bins=[2, 2, 2, 2], TYPE=None, geometric=False):
    """
        first split box size to len(aspect_bins) part equally,
        then split echo part's aspect to aspect_bins[i] part equally.
        
        WHSAs: np.ndarray, [[w, h, s, a], [w, h, s, a]....]
        aspect_bins: list(int),
        TYPE: optinal in ['split', 'median', 'mean', None], None for output all
    """
    idxs = {'split': 0, 'median': 1, 'mean': 2, None: None}
    idx = idxs[TYPE]
    box_result = euqal_part_point(WHSAs[:, 2], len(aspect_bins), None, geometric)
    WHSAs = WHSAs.copy()
    WHSAs = np.array(sorted(WHSAs, key=lambda whsa: whsa[2]))   # sort by s
    l = len(WHSAs)
    n = len(aspect_bins)   # box_size bins
    aspect = WHSAs[:, -1]
    # bound_point = aspect[list(range(0, l, l//n))]
    aspects = aspect[:l //  n * n].reshape((n, l//n))
    splits, medians, means = [], [], []
    for aspect, aspect_bin in zip(aspects, aspect_bins):
        split, median, mean = euqal_part_point(aspect, aspect_bin, None, geometric)
        splits.append(split)
        medians.append(median)
        means.append(mean)
    aspect_result = (splits, medians, means)
    if idx is None:
        return list(zip(*box_result)), list(zip(*aspect_result))
    else:
        return box_result[idx], aspect_result[idx]
find_box_aspect_point = recomand_box_aspect_point
    
"""
    2. StatisticMetric
"""

class StatisticMetric(object):
    """
    function:
        update: main function, need implemnet for diffrent needs.
        get_mean/get_geometric_mean/get_max/get_min: statistic function
        get: return self._data,record all data
        hist/hist_log/plot: function for simple show
        reset: clear self._data
    Attribute:
        self._data: [[e1, e2, e3...en], [e1, e2, e3, ...en]...], shape=(m, n), m is data count, n is data dim. or [d1, d2, ...], shape=(m,)
        self.name: the object name, like 'caltech_perdestrian', use to plot/hist
        self.axis: the data count axis, defalut is 0, use ofr get_mean/get_min/....
        self.column_names: every column of a data's name, means name of e1, e2, e3, ...en, use to plot/hist
    """
    def __init__(self, name="", axis=0, column_names=None):
        self.name = name
        self.axis = axis
        self._data = []
        self.column_names=column_names
        
    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)
    
    def update(self, data):
        self._data.extend(data)
        
    def reset(self):
        del self._data
        self._data = []
                
    def get_mean(self, round=None):
        r = np.mean(np.array(self._data), axis=self.axis)
        if round is not None: r = np.round(r, round)
        return r
    
    def get_min(self, round=None):
        r = np.min(np.array(self._data), axis=self.axis)
        if round is not None: r = np.round(r, round)
        return r
    
    def get_max(self, round=None):
        r = np.max(np.array(self._data), axis=self.axis)
        if round is not None: r = np.round(r, round)
        return r
    
    def get(self, round=None):
        r = np.array(self._data)
        if round is not None: r = np.round(r, round)
        return r
    
    def hist(self, idx=None, *args, **kwargs):
        r = np.array(self._data)
        xlabel = self.name
        if idx is not None:
            r = r[:, idx]
            if self.column_names is not None:
                xlabel = xlabel + " " + self.column_names[idx]
        plt.xlabel(xlabel)
        plt.ylabel("count")
        return plt.hist(r, *args, **kwargs)
    
    def get_geometric_mean(self, round=None):
        r = np.exp(np.log(np.array(self._data)).mean(axis=self.axis))
        if round is not None: r = np.round(r, round)
        return r
    
    def hist_log(self, idx=None, *args, **kwargs):
        r = np.log(np.array(self._data))        
        xlabel = "log(" + self.name
        if idx is not None:
            r = r[:, idx]
            if self.column_names is not None:
                xlabel = xlabel + " " + self.column_names[idx]
        plt.xlabel(xlabel + ")")
        plt.ylabel("count")
        return plt.hist(r, *args, **kwargs)
    
    def plot(self, xidx, yidx, plot_fmt='-', *args, **kwargs):
        r = np.array(self._data)
        xlabel, ylabel = self.name,self.name
        if self.column_names is not None:
            xlabel += " " + self.column_names[xidx]
            ylabel += " " + self.column_names[yidx]
        else:
            xlabel += " " + str(xidx)
            ylabel += " " + str(yidx)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        return plt.plot(r[:, xidx], r[:, yidx], plot_fmt, *args, **kwargs)
    
    def equal_part_point(self, idx, bins=5, TYPE=None, geometric=False):
        return euqal_part_point(np.array(self._data)[:, idx], bins, TYPE, geometric)
            

class AspectRatioMetric(StatisticMetric):
    """
    """
    def update(self, boxes):
        r = []
        for box in boxes:
            x1, y1, x2, y2 = box
            if x1 >= 0:
                r.append((x2-x1) / (y2-y1))
                self._data.append((x2-x1) / (y2-y1))
        return np.array(r)
    
    
class BoxSizeMetric(StatisticMetric):
    """
        [w, h, sqrt(w*h)]
    """
    def update(self, boxes):
        r = []
        for box in boxes:
            x1, y1, x2, y2 = box
            if x1 >= 0:
                w , h = (x2-x1), (y2-y1)
                r.append([w, h, sqrt(w*h)])
                self._data.append([w, h, sqrt(w*h)])
        return np.array(r)


class BoxSizeAspectRatioMetric(StatisticMetric):
    """
        [w, h, sqrt(w*h), aspect]
    """
    def update(self, boxes):
        boxes = np.array(boxes)
        W = boxes[:, 2] - boxes[:, 0]
        H = boxes[:, 3] - boxes[:, 1]
        S = np.sqrt(W * H)
        aspect = W / H
        for w, h, s, a in zip(W, H, S, aspect):
            self._data.append([w, h, s, a])
        return np.concatenate([W, H, S, aspect]).reshape((-1, 4))
    
    def show(self, size_type='size', plot_fmt='.', *args, **kwargs):
        """
            size_type: option in ['size', 'width', 'height']
        """
        size_type = size_type.lower()
        idxes = {'width':0, 'height':1, 'size':2}
        idx = idxes[size_type.lower()]
        r = np.array(self._data)
        plt.xlabel('box ' + size_type.lower())
        plt.ylabel('aspect ratio')
        return plt.plot(r[:, idx], r[:, 3], plot_fmt, *args, **kwargs)

import pandas
EPS=1e-10
class AnchorLevelMetric(StatisticMetric):
    """
        检测的本质是分类,AnchorLevelMetric是利用分类的思想研究检测问题,分类中比较关心的,每个类别的样本数量,
        AnchorLevelMetric主要分析以下几个指标:
         * C = 每个类别的样本总数,即属于某个类别的anchor数量
         * c = 平均每张图中能产生多少个该类数据样本(正例anchor个数) --> 用于debug anchor的size,aspect,step等参数设置是否合理(符合数据)
         * M = 包含目标的图片数量
         C = c * M
         
         * 每个图片最多/最少包含多少个class=cid的anchor
    """
    def __init__(self, class_names, *args, **kwargs):
        super(AnchorLevelMetric, self).__init__(*args, **kwargs)
        self.class_names = class_names
        self._cids = np.array(range(1, len(self.class_names) + 1))
        
    def update(self, cls_target):
        """
            cls_target: np.ndarray, shape=(batch_size, anchor_count)
            return : shape=(batch_size, class_count)
                     [[pos_anchor_num_of_class1, pos_anchor_num_of_class2, .....],  # image 1
                      [pos_anchor_num_of_class1, pos_anchor_num_of_class2, .....],  # image 2
                     ....
                      [pos_anchor_num_of_class1, pos_anchor_num_of_class2, .....]]  # image batch_size
        """
        r = []
        for cls_t in cls_target:
            cls_t = cls_t[cls_t > 0]
            if len(cls_t) == 0: continue
            cls_t = np.concatenate([cls_t, self._cids], axis=0)
            cids, counts = np.unique(cls_t, return_counts=True)
            counts -= 1
            r.append(counts)
            self._data.append(counts)
            
        return np.array(r)
    
    def get_mean(self, idx=None):
        """
            (mean positive anchor of class idx) = (the count of anchor that
                            is class idx) / (the count of images that containt class idx)
                            
            idx: class id - 1, None means all class
        """
        self._data = np.array(self._data)
        if idx is None:
            r = self._data
        else:
            r = self._data[:, idx]
        count = (r != 0).sum(axis=self.axis)
        r = r.sum(axis=self.axis)
        return r / (count + EPS)
    
    def get_table(self, round=None):
        """
            image_count: how many image conataint object that is class cid?
            anchor_mean: how many anchors in every image that contain class cid?
            class_names:
        """
        res = {}
        self._data = np.array(self._data)
        M = (self._data != 0).sum(axis=self.axis)  # images count for contain every class
        C = self._data.sum(axis=self.axis)
        c = C / (M+EPS)
        res['total positive anchor count'] = C if round is None else np.round(C, round)
        res['mean positive anchor per image'] = c if round is None else np.round(c, round)
        res['image count per class'] = M if round is None else np.round(M, round)
        res['class name'] = self.class_names
        res['max positive anchor per image'] = self.get_max()
        
        # to get no zero min, first set zero to biggest number; 
        # then get_min; at end, set biggest number back to zero
        inf = np.max(self._data) + 1
        self._data[self._data==0] = inf
        mina = self.get_min()
        mina[mina==inf] = 0                            # means all is zeros
        res['min positive anchor per image'] = mina
        self._data[self._data==inf] = 0
        return pandas.DataFrame(res).T
    
class ObjectLevelMetric(StatisticMetric):
    """
        ObjectLevelMetric 从 object 的层次分析数据, 分别有以下几个内容:
         * 每个object mean/max/min有多少个 anchor box
    """
    def __init__(self, class_names, *args, **kwargs):
        super(ObjectLevelMetric, self).__init__(*args, **kwargs)
        self.class_names = class_names
    
    def update(self, cls_target, bboxes):
        """
            cls_target: np.ndarray, shape=(batch_size, anchor_count), ssd net train mode generate
            bboxes: np.ndarray, shape=(batch_size, anchor_count, 4), ssd net generate anchor after decode
            
            return:
                 [[obj1_cid, obj1_anchor_num], [obj2_cid, obj2_anchor_num] ... [objn_cid, objn_anchor_num]]
        """
        r = []
        for cls_t, bbox_t in zip(cls_target, bboxes):
            valid = cls_t > 0
            cls_t = cls_t[valid]
            bbox_t = bbox_t[valid]
            for cid in set(cls_t):
                valid = (cls_t == cid)
                bbox = np.round(bbox_t[valid]) 
                boxs, counts = np.unique(list(bbox), return_counts=True, axis=0)
                r.extend([[int(cid), count] for count in counts])
        self._data.extend(r)
        return r
    
    def _group_by_cid(self):
        self._data = np.array(self._data)
        data = pandas.DataFrame({'cid': self._data[:, 0], 'anchor count': self._data[:, 1]})
        data = data['anchor count'].groupby(data['cid'])
        return data
    
    def _get_pd_result(self, data):
        r = np.zeros(shape=(len(self.class_names), ))
        cid = data.keys().values
        count = data.values
        r[cid-1]=count
        return r
    
    def get_mean(self, data=None):
        if data is None: 
            data = self._group_by_cid()
        return self._get_pd_result(data.mean())
    
    def get_max(self, data=None):
        if data is None: 
            data = self._group_by_cid()
        return self._get_pd_result(data.max())
    
    def get_min(self, data=None):
        if data is None: 
            data = self._group_by_cid()
        return self._get_pd_result(data.min())
    
    def get_table(self, round=None):
        data = self._group_by_cid()
        res = {}
        res['mean pos anchor per obj'] = self.get_mean(data)
        res['max pos anchor per obj'] = self.get_max(data)
        res['min pos anchor per obj'] = self.get_min(data)
        res['class name'] = self.class_names
        if round is not None:
            for key in ['mean pos anchor per obj', 'max pos anchor per obj', 
                        'min pos anchor per obj']:
                res[key] = np.round(res[key], round)
        return pandas.DataFrame(res).T
    
    def hist(self, idx=None, *args, **kwargs):
        self._data = np.array(self._data)
        if idx == None:
            data = self._data[:, 1]
            plt.ylabel('object count of all class')
        else:
            data = self._data[:, 1]
            data = data[self._data[:, 0] == idx+1]
            plt.ylabel('object count of ' + self.class_names[idx])
        plt.xlabel('positive anchor count')
        plt.hist(data, *args, **kwargs)
        
    @property
    def object_count(self, idx=None):
        """
            may be not accuracy, cause box unique error(误差)
        """
        if idx is not None:
            class_data = self._data[:, 0] == (idx)
        return len(self._data)

class CenterVObjectMetric(StatisticMetric):
    """
        it want to calculate CVO = a center match how many object, cuase a center(diffrent anchor box but same feature)
        match a objcet is meaning, match more than one not make scense. so CVO bigger than one, more error cause for classifier.

        COB = a center match a object with how many box, it use to kwon the anchor use rate.
    """
    def __init__(self, TYPE='CVO', *args, **kwargs):
        super(CenterVObjectMetric, self).__init__(*args, **kwargs)
        func = {'cvo': self._centerVobject, 'cob': self._centerObjectBoxCount}
        self.cal_func = func[TYPE.lower()]
    
    def _centerVobject(self, count):
        # a center match how many object
        return [len(sub_count) for center, sub_count in count.items()]
            
    def _centerObjectBoxCount(self, count):
        # a center match a object with how many box
        r = []
        for center, sub_count in count.items():     
            r.extend(sub_count.values())   # a center match a object with how many box
        return r
                
    def update(self, center_t, obj_t):
        """
            it want to calculate CVO = a center match how many object, cuase a center(diffrent anchor box but same feature)
            match a objcet is meaning, match more than one not make scense. so CVO bigger than one, more error cause for classifier.
            
            COB = a center match a object with how many box, it use to kwon the anchor use rate.
            
            obj_t:    [box1_obj, box2_obj ... boxn_obj]
            center_t: [box1_cen, box2_cen ... boxn_cen]
            
            var:
                count: {dict(dict(int)), count[center_id][obj_id] is the how many box of center_id match to obj_id
        """
        count = {}
        for obj, center in zip(obj_t, center_t):
            if center not in count:
                count[center] = {obj: 1}
            else:
                sub_count = count[center]
                if obj not in sub_count:
                    sub_count[obj] = 1
                else:
                    sub_count[obj] += 1
        r = self.cal_func(count)
        self._data.extend(r)
        # print(count)
        return r
    
class ObjectVCenterMetric(StatisticMetric):
    def _objectVcenter(self, count):
        # a object match how many center
        return [len(sub_count) for center, sub_count in count.items()]
    
    def update(self, center_t, obj_t):
        """
            obj_t:    box1_obj, box2_obj ... boxn_obj
            center_t: box1_cen, box2_cen ... boxn_cen
            var:
                count: {dict(dict(int)), count[obj_id][center_id] is the how many box of center_id match to obj_id
        """
        count = {}
        for obj, center in zip(obj_t, center_t):
            if obj not in count:
                count[obj] = {center: 1}
            else:
                sub_count = count[obj]
                if center not in sub_count:
                    sub_count[center] = 1
                else:
                    sub_count[center] += 1
        r = self._objectVcenter(count)
        self._data.extend(r)
        return r

class MatStatisticMetric(object):
    """
        each element is a mat shape=(Li,S), each element can have diffrent Li and same S.
        _data: [[(a1, a2), (b1, b2), (c1, c2), (d1, d2)], [e,f, g] ...], list(np.array(shape=(Li, S)))
        
        example:
        -------
        >> i = MatStatisticMetric()
        >> i.update([[1, 2, 3], [4, 5]])
        >> i.concat_get()
           array([1, 2, 3, 4, 5])
    """
    def __init__(self, name=''):
        self.name = name
        self._data = []
        
    def update(self, x):
        self._data.extend(x)
    
    def get_mean(self, idx=None):
        if idx is None:
            r = [np.mean(d) for d in self._data]
        else:
            r = [np.mean(np.array(d)[:, idx]) for d in self._data]
        return np.array(r)
    
    def get_max(self, idx=None):
        if idx is None:
            r = [np.max(d) for d in self._data]
        else:
            r = [np.max(np.array(d)[:, idx]) for d in self._data]
        return np.array(r)
    
    def get_min(self, idx=None):
        if idx is None:
            r = [np.min(d) for d in self._data]
        else:
            r = [np.min(np.array(d)[:, idx]) for d in self._data]
        return np.array(r)
    
    def get(self):
        return self._data
    
    def concat_get(self):
        self._data = [np.array(d) for d in self._data]
        return np.concatenate(self._data, axis=0)
        

#from MPUtils import MatStatisticMetric
#from MPUtils import AspectRatioMetric, BoxSizeMetric

class IOUMetric(MatStatisticMetric):
    """
        1. 命中gt(object)的anchor的平均IOU (顺便统计平均max, min)
        2. 哪些gt(object)的的平均IOU很小 (< threshold)
        3. 哪些gt(object)存在很小的IOU  (< threshold)
    """
    def __init__(self, *args, **kwargs):
        super(IOUMetric, self).__init__(*args, **kwargs)
        self._boxes = []
    
    def update(self, ious, obj_boxes, obj_t):
        """
            ious: shape=(num_obj, num_anchors), ious[obj_id, anchor_id] means iou of obj_id and anchor_id
            obj_boxes: [[obj1_x1, obj1_y1, obj1_x2, obj1_y2]...], shape=(num_obj, 4)
            obj_t: [box1_object_id, box2_object_id ....], shape=(num_anchors,)
            return:
                each element is iou array of object i with all anchor box matched it, so may have diffrent Li
                [(obj1_anchor1_iou, obj1_anchor2_iou, ...obj1_anchori_iou),
                 (obj2_anchor1_iou, obj2_anchor2_iou, ...obj1_anchori_iou, ...,obj2_anchorj_iou),...].
        """
        obj_ious = [[] for _ in range(len(obj_boxes))]
        for anchor_id, obj_id in enumerate(obj_t):
            obj_ious[obj_id].append(ious[obj_id, anchor_id])
            
        self._data.extend(obj_ious)
        self._boxes.extend(obj_boxes)
        return obj_ious
    
    def get_small_iou_box_info(self, threshold=0.5, type='mean'):
        """
        param:
        ------
            threshold: < threshold is small iou
            type: optinal in ('mean', 'max', 'min'). which iou used to compare threshold for every object, 
                cuase a object can match many anchor, have many iou data, so need to choose which to compare
            
        return:
        -------
            small_rate: how many rate object iou smaller than threshold
            bm: BoxSizeMetric, all small iou objects' BoxSizeMetric
            am: AspectRatioMetric, all small iou objects' AspectRatioMetric
        """
        func = {'mean': self.get_mean, 'max':self.get_max, 'min': self.get_min}
        ious = func[type.lower()]()                # get iou info for every object
        boxes = np.array(self._boxes)              # get box for every object
        small_iou = ious <= threshold 
        small_boxes = boxes[small_iou]
        
        am = AspectRatioMetric(name='aspect ratio')
        bm = BoxSizeMetric(name='box size')
        am.update(small_boxes)
        bm.update(small_boxes)
        
        small_rate = (small_iou).sum() / ious.size
        return small_rate, bm, am
        
