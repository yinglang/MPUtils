# insert ../ directory
import os
import sys
_ = os.path.abspath(os.path.abspath(__file__) + '/../../')
if _ not in sys.path:
    sys.path.insert(0, _)
    
from mutils import show_images, inv_normalize, accuracy
import mxnet as mx
from mxnet import gluon, nd, autograd
from mxnet.gluon import loss, nn
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss as SCELoss
import numpy as np
"""
     tool function
"""
def show_data(data, clip=True, **kwargs):
    images = inv_normalize(data, clip=clip)
    show_images(images, clip=clip, **kwargs)
    
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
    logger
"""
class LogRecorder(object):
    def __init__(self):
        self.acc, self.sse, self.losses, self.log_result_score, self.log_label_score = [], [], [], [], []
        self.result = []
    
    def __call__(self, output, sparse_label_ctx, origin_data, data, loss, ):
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
        self.result.append(list(idx))
        
    def print_log(self, iter_log_period, iters, mean_loss, record_detail, data, show_clip, **kwargs):
        """
            invoke every iter
        """
        if iter_log_period is not None and iters % iter_log_period == 0:
            n = min([5, data.shape[0]])
            show_data(data[:n], show_clip)
        if iter_log_period is not None and iters % 5 == 0:
            print('iter: {} loss: {}'.format(iters, mean_loss), end=' ') 
            if record_detail:
                n = data.shape[0]
                print("acc: {} MSE: {} result_score: {} label_score: {}".format(self.acc[-1]/n, self.sse[-1]/ n, np.exp(self.log_result_score[-1]/n), np.exp(self.log_label_score[-1]/n)), end=' ')
            print("")
            #print self.result[-1]
    def asnumpy(self):
        self.acc, self.sse, self.losses = np.array(self.acc), np.array(self.sse), np.array(self.losses)
        self.log_result_score, self.log_label_score  = np.array(self.log_result_score), np.array(self.log_label_score )
        
        
