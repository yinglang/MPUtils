"""
    0. tool function
"""
import sys
sys.path.insert(0, '../')
from utils import show_images, inv_normalize
import mxnet as mx
from mxnet import gluon, nd, autograd
from mxnet.gluon import loss, nn
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss as SCELoss
import numpy as np

def show_data(data, clip=True):
    images = inv_normalize(data, clip=clip)
    show_images(images, clip=clip)
    
def MSE(origin_data, data):
    return nd.mean(nd.sum((origin_data - data) ** 2, axis=0, exclude=True)).asscalar()

def SSE(origin_data, data):
    # sum squrare error
    return (nd.sum((origin_data - data) ** 2)).asscalar()

def mean_prob(probs):
    """
    (prob[0] * prob[1] * ... prob[n-1]) ** (1.0/n)
    """
    return np.exp(np.mean(np.log(probs)))

def log_prob_sum(probs):
    return np.sum(np.log(probs))

"""
    1. batchnorm control
"""
class BNControl(object):
    """
        only support renet18 by me now.
    """
    @staticmethod
    def collect_BN(blocks):
        BN = []
        for blk in blocks:
            _type = str(blk).split('(')[0]
            if _type == 'BatchNorm':
                BN.append(blk)
            elif _type == 'Residual':
                BN.extend([blk.bn1, blk.bn2])
        return BN
    
    def __init__(self, blocks, use_batch=True):
        self.bns = BNControl.collect_BN(blocks)
        self.use_batch = use_batch
        self.data_list = []
        
    def store(self):
        if self.use_batch: # use batch data and no change running mean/std
            if len(self.data_list) == 0:
                for i, bn in enumerate(self.bns):
                    self.data_list.append(bn.params.get('running_mean').data().copy())
                    self.data_list.append(bn.params.get('running_var').data().copy())
            else:
                for i, bn in enumerate(self.bns):
                    self.data_list[2*i][:] = bn.params.get('running_mean').data()
                    self.data_list[2*i+1][:] = bn.params.get('running_var').data()
        else: # no use batch data and no change running mean/std
            for i, bn in enumerate(self.bns):
                self.data_list.append(bn._kwargs['use_global_stats'])
                bn._kwargs['use_global_stats'] = True
        
    def load(self):
        if self.use_batch:
            for i in range(len(self.bns)):
                bn, mean, std = self.bns[i], self.data_list[2*i], self.data_list[2*i+1]
                bn.params.get('running_mean').set_data(mean)
                bn.params.get('running_var').set_data(std)
        else:
            for i in range(len(self.bns)):
                bn, data = self.bns[i], self.data_list[i]
                bn._kwargs['use_global_stats'] = data

class ResNetBNControl(BNControl):
    """
        test on Resnet18_v1 and Resnet50_v1
    """
    def collect_BN(self, blocks):
        BN = []
        for blk in blocks:
            _type = str(blk).split('(')[0]
            if _type == 'BatchNorm':
                BN.append(blk)
            elif _type == 'Residual':
                BN.extend([blk.bn1, blk.bn2])
            elif _type == 'HybridSequential' or _type == 'Sequential':
                BN.extend(self.collect_BN(blk))
            elif _type in ['BasicBlockV1', 'BottleneckV1']:
                BN.extend(self.collect_BN(blk.body))
                if hasattr(blk, 'downsample') and blk.downsample is not None:
                    BN.extend(self.collect_BN(blk.downsample))
        return BN
    
    def __init__(self, blocks, use_batch=True):
        if isinstance(blocks, list):
            self.bns = []
            for block in blocks:
                self.bns.extend(self.collect_BN(block))
        else:
            self.bns = self.collect_BN(blocks)
        self.use_batch = use_batch
        self.data_list = []
        
"""
    2. sgd and constraint
"""
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss as SCELoss
from utils import accuracy

class SGD(object):
    def __init__(self, lr):
        self.lr = lr
        
    def __call__(self, data, **kwargs):
        data[:, :, :, :] = data - data.grad * self.lr
        
    def clear(self):
        pass

class SGD_momentum(object):
    """
        v = momentum * v - learning_rate * gradient
        weight += v
    """
    def __init__(self, lr, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = 0
        
    def __call__(self, data, **kwargs):
        self.v = self.momentum * self.v - self.lr * data.grad
        data[:, :, :, :] += self.v
        
    def clear(self):
        self.v = 0
        pass
        
class SGD_with_MSE_constraint(object):
    def __init__(self, fixed_mse):
        self.fixed_mse = fixed_mse
        
    def __call__(self, data, **kwargs):
        mse = nd.mean(nd.sum(data.grad ** 2, axis=0, exclude=True)).asscalar()
        scale = self.fixed_mse / mse
        data[:, :, :, :] = data - data.grad * scale
        
    def clear(self):
        pass
    
class Constraint(object):
    def __init__(self, threshold):
        self.threshold = threshold
    
    def __call__(self, data, iters, max_iters):
        """
            invoke every backgrad iter
        """
        # if iters < max_iters: return # itres begin with 1
        threshold = self.threshold
        if self.threshold is not None:
            for i in range(3):
                data[:, i, :, :] = data[:, i, :, :].clip(threshold[0, i].asscalar(), threshold[1, i].asscalar())

"""
    3. logger
"""
class LogRecorder(object):
    def __init__(self):
        self.acc, self.sse, self.losses, self.log_result_score, self.log_label_score = [], [], [], [], []
    
    def __call__(self, output, sparse_label_ctx, origin_data, data, loss):
        """
            invoke every iter
        """
        self.acc.append(accuracy(output, sparse_label_ctx))
        self.sse.append(SSE(origin_data, data))
        self.losses.append(nd.sum(loss).asscalar())
        # add mean result score
        output = nd.softmax(output).asnumpy()
        idx = np.argmax(output, axis=1)
        _label = sparse_label_ctx.asnumpy().reshape((-1, )).astype('int')
        self.log_result_score.append(log_prob_sum(output[range(idx.shape[0]), idx]))
        self.log_label_score.append(log_prob_sum(output[range(idx.shape[0]), _label]))
        
    def print_log(self, iter_log_period, iters, mean_loss, record_detail, data, show_clip, **kwargs):
        """
            invoke every iter
        """
        if iter_log_period is not None and iters % iter_log_period == 0:
            n = min([5, data.shape[0]])
            show_data(data[:n], show_clip)
        if iter_log_period is not None and iters % 5 == 0:
            print 'iter:', iters, 'loss:', mean_loss, 
            if record_detail:
                n = data.shape[0]
                print "acc:", self.acc[-1]/n, "MSE:", self.sse[-1]/ n, "result_score:", np.exp(self.log_result_score[-1]/n),
                print "label_score:", np.exp(self.log_label_score[-1]/n),
            print
            
    def asnumpy(self):
        self.acc, self.sse, self.losses = np.array(self.acc), np.array(self.sse), np.array(self.losses)
        self.log_result_score, self.log_label_score  = np.array(self.log_result_score), np.array(self.log_label_score )
        
"""
    4.1 loss calculator
           SoftmaxCrossEntopyLoss
           ContentAndSoftmaxCrossEntropyLoss
"""
class LossCalculator(object):
    def __init__(self):
        self.loss_f = SCELoss(sparse_label=False)

    def __call__(self, net, data, label, labelmap, labeloutputmap, sparse_label):
        output = net(data)
        if labeloutputmap is not None: label = labeloutputmap(output, sparse_label).as_in_context(ctx)
        loss = self.loss_f(output, label)
        if labeloutputmap is None and labelmap is None:
            loss = -loss
        return output, loss, None

def extract_features(net, x, feature_layers_index=[]):
    features = []
    for i, block in enumerate(net):
        x = block(x)
        if i in feature_layers_index:
            features.append(x)
    return features, x

def content_loss(content_y_hat, content_y, weights):
    loss = []
    for y, y_hat, w in zip(content_y, content_y_hat, weights):
        loss.append(w * nd.mean(nd.abs(y - y_hat), axis=0, exclude=True))
    if len(loss) == 0: return 0
    return nd.add_n(*loss)

class ContentLossCalculator(LossCalculator):
    def __init__(self, net, feature_layers_index=[], content_weights=[]):
        self.class_loss = SCELoss(sparse_label=False)
        self.feature_layers_index = feature_layers_index
        self.content_weights = content_weights
        self.net = net
        
    def set_content_y(self, data):
        self.content_y, _ = extract_features(self.net, data, self.feature_layers_index)

    def __call__(self, net, data, label, labelmap=None, labeloutputmap=None, sparse_label=None):
        def mean(d):
            return nd.mean(d).asscalar()
        content_y_hat, output = extract_features(self.net, data, self.feature_layers_index)
        _content_loss = content_loss(content_y_hat, self.content_y, self.content_weights)
        
        if labeloutputmap is not None: label = labeloutputmap(output, sparse_label).as_in_context(ctx)
        if net != self.net:
            tmp = nd.mean(output).asscalar() # for momory free, complete graph
            output = net(data)
        class_loss = self.class_loss(output, label)

        if labeloutputmap is None and labelmap is None:
            class_loss = -class_loss
        loss = class_loss + _content_loss
        mean_content_loss = mean(_content_loss) if isinstance(_content_loss, mx.ndarray.ndarray.NDArray) else 0
        return output, loss, (mean_content_loss, mean(class_loss))
"""
    4.2 labelmap
"""
# better loss
import random
class AvgLabelMap(object):
    # give wrong class same prob
    def __init__(self, num_class):
        self.num_class = num_class
    
    def __call__(self, label):
        label = label.reshape((-1,)).one_hot(self.num_class)
        label = (1 - label) / (self.num_class)
        return label
    
class RandomOneLabelMap(object):
    def __init__(self, num_class):
        self.num_class = num_class
    
    def __call__(self, label):
        label = label.asnumpy().reshape((-1,))
        t = np.random.uniform(0, self.num_class, size=(label.shape[0],)).astype('int')
        t[t == 200] = 199
        diff_idx = (t != label.astype('int'))
        same_idx = (1 - diff_idx).astype('bool')
        label[diff_idx] = t[diff_idx]
        label[same_idx] = (label[same_idx] + 1) % self.num_class
        return nd.array(label.astype('float32')).one_hot(self.num_class)

class LabelMap(object):
    def __init__(self, num_class, labelmap):
        self.num_class = num_class
        self.labelmap = labelmap
    
    @staticmethod
    def generate_randomone_labelmap(num_class):
        t = random.shuffle(num_class)
        t[t == 200] = 199
        diff_idx = (t != label.astype('int'))
        same_idx = (1 - diff_idx).astype('bool')
    
    def __call__(self, label):
        label = label.asnumpy()
        label = self.labelmap[label]
        return nd.array(label)
    
class MaxOutputLabelMap(object):
    def __init__(self, num_class):
        self.num_class = num_class
        
    def __call__(self, output, label):
        output = nd.softmax(output).asnumpy()
        label = label.asnumpy().astype('int').reshape((-1,))
        output[range(label.shape[0]), label] = 0
        label = nd.array(output).argmax(axis=1).one_hot(self.num_class).astype('float32')
        return label
    
class MinOutputLabelMap(object):
    def __init__(self, num_class):
        self.num_class = num_class
        
    def __call__(self, output, label):
        output = nd.softmax(output).asnumpy()
        label = label.asnumpy().astype('int').reshape((-1,))
        output[range(label.shape[0]), label] = 1
        label = nd.array(output).argmin(axis=1).one_hot(self.num_class).astype('float32')
        return label
"""
    num_class = 200
    # labelmap = AvgLabelMap(num_class)
    labelmap = RandomOneLabelMap(num_class)
    loss_f = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
    for data, label in train_data:
        print label.asnumpy().T
        label = labelmap(label)
        #print label[0, :10]
        print label.argmax(axis=1).asnumpy(), label.sum().asnumpy()
        print loss_f(nd.random.uniform(shape=(label.shape[0], num_class)), label)
        break

    # labelmap = MaxOutputLabelMap(num_class)
    labelmap = MinOutputLabelMap(num_class)
    loss_f = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
    for data, label in train_data:
        print label.asnumpy().T
        output = nd.random.uniform(shape=(label.shape[0], num_class))
        label = labelmap(output, label)
        print label.argmax(axis=1).asnumpy(), label.sum().asnumpy(), output.argmax(axis=1).asnumpy(), output.argmin(axis=1).asnumpy()
        print loss_f(nd.random.uniform(shape=(label.shape[0], num_class)), label)
        break

    a = np.array([[1, 2], [3, 4]])
    a[[1, 0], [1, 0]]
"""

"""
    5. generator and record, and evaluate
"""
from backgrad_utils_v02 import *
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss as SCELoss
    
def generate_backgrad_data_def_loss(net, data, label, ctx=mx.gpu(0), bn_control=None, 
                           max_iters=60, sgd=SGD(lr=0.1), # optimizer args
                           iter_log_period=None, show_clip=False, record_detail=False, logger=None,# log args
                           labelmap=None, labeloutputmap=None, loss_f=LossCalculator(), # loss_args
                           threshold=None ):#constraint_agrs
    """
        param:
            net: base net model
            data: data will be changed.recomand data context in cpu.
            label: data's label, recomand label context in cpu.
            ctx: backgrad context, if ctx is gpu and data/label in gpu(or said they in same), then backgradwill change the data iteself, 
            max_iters: max_iters for backgrad
            lr: lr for backgrad
            iter_log_period: output log period, None means never output.
            show_clip: log show backgrad image is clip?
            loss_f: loss function for backgrad
            bn_control: bn_control be fore backgrad, None means never use BN Control.
            sgd: backgrad optimizer method.
            trheshold: returned data's color clip trheshold.
            labelmap, labeloutputmap: decide use what label to backgrad generate adversal smaple. use -loss when all None. only one canbe specified.

        data is better in cpu, if data in ctx(global var), the returned backgrad_data is shallow copy of data.
    """        
    if bn_control is not None:
        bn_control.store()

    sparse_label = label.copy()
    if labelmap is not None: label = labelmap(sparse_label)
    data, label, sparse_label_ctx = data.as_in_context(ctx), label.as_in_context(ctx), sparse_label.as_in_context(ctx)
    if logger is None: logger = LogRecorder()
    if record_detail:
        origin_data = data.copy()
    constraint = Constraint(threshold)
        
    for iters in range(1, max_iters+1):
        with autograd.record():
            data.attach_grad()
            output ,loss, loss_args = loss_f(net, data, label, labelmap, labeloutputmap, sparse_label)
        loss.backward()
        mean_loss = nd.mean(loss).asscalar()     # reduce will make memory release
        
        if record_detail:
            logger(output, sparse_label_ctx, origin_data, data, loss)
        logger.print_log(iter_log_period, iters, mean_loss, record_detail, data, show_clip, loss_args=loss_args)
        
        sgd(data)
        
        constraint(data, iters, max_iters)
    if bn_control is not None:
        bn_control.load()

    logger.asnumpy()
    sgd.clear()
    return data, (logger, )

def adversal_acc(net, data_iter, bn_control, **kwargs):
    bn_control.store()
    i= 0
    acc, mse, loss, result_score, label_score, image_count = None, None, None, None, None, 0
    for data, label in data_iter:
        data, (logger,) = generate_backgrad_data_def_loss(net, data, label, bn_control=None, **kwargs)
        if acc is None: 
            acc, mse, loss = logger.acc, logger.sse, logger.losses
            result_score, label_score = logger.log_result_score, logger.log_label_score
        else: 
            acc, mse, loss = acc + logger.acc, mse + logger.sse, loss + logger.losses
            result_score, label_score = result_score + logger.log_result_score, label_score + logger.log_label_score
        image_count += data.shape[0]
        #if i > 3 : break
        i += 1
    acc, mse, loss = acc / image_count, mse / image_count, loss / image_count
    result_score, label_score = np.exp(result_score / image_count), np.exp(label_score / image_count)
    bn_control.load()
    return acc, mse, loss, result_score, label_score


