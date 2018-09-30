# better loss
import mxnet as mx
from mxnet import gluon, nd, autograd
from mxnet.gluon import loss, nn
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss as SCELoss
import numpy as np
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
