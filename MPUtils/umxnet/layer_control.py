from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision
import numpy as np
import random
import mxnet as mx
from mxnet.gluon.data.vision import transforms
from mxnet import gluon
from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon import nn
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss as SCELoss

def get_all_type_layers(block, TYPE):
    def _get_all_type_layers(block, TYPE, result):
        if isinstance(block, TYPE):
            result.append(block)
        if isinstance(block, nn.Block):
            if isinstance(block, (nn.HybridSequential, nn.Sequential)):
                for blk in block:
                    _get_all_type_layers(blk, TYPE, result)
            else:
                for key, blk in block.__dict__.items():
                    _get_all_type_layers(blk, TYPE, result)
    result = []
    _get_all_type_layers(block, TYPE, result)
    return result
    
class DPControl(object):
    def __init__(self, block):
        self.dropouts = get_all_type_layers(block, nn.Dropout)
    
    def save(self):
        self._rates = {}
        for dp in self.dropouts:
            self._rates[dp] = dp._rate
            dp._rate = 0
    
    def load(self):
        for dp in self.dropouts:
            dp._rate = self._rates[dp]
            

class BNControl(object):
    """
        only support renet18 by me now.
    """
    
    def __init__(self, blocks, use_batch=True):
        self.bns = get_all_type_layers(blocks, nn.BatchNorm)
        self.use_batch = use_batch
        self.data_list = []
        
    def save(self):
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