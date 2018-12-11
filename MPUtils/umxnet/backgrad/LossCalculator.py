import mxnet as mx
from mxnet import gluon, nd, autograd
from mxnet.gluon import loss, nn
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss as SCELoss
import numpy as np

class LossCalculator(object):
    def __init__(self):
        self.loss_f = SCELoss(sparse_label=False)

    def __call__(self, net, data, label, labelmap, labeloutputmap, sparse_label, ctx=mx.cpu(0)):
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

    def __call__(self, net, data, label, labelmap=None, labeloutputmap=None, sparse_label=None, ctx=mx.cpu(0)):
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
    
    
