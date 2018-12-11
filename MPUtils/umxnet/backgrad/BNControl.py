# insert ../ directory
# import os
# import sys
# _ = os.path.abspath(os.path.abspath(__file__) + '/../../')
# if _ not in sys.path:
#     sys.path.insert(0, _)
from mxnet import nd


def bn_checker(BNs):
    params = []
    for bn in BNs:
        params.append(bn.params.get('running_mean').data().copy())
        
    def compare():
        nparams = []
        diff = 0.
        print('p, b {} {}'.format(params[0][0], BNs[0].params.get('running_mean').data()[0]))
        for param, bn in zip(params, BNs):
            nparam = bn.params.get('running_mean').data()
            diff += nd.sum(nd.abs(nparam - param)).asscalar()
        return diff
    return compare


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