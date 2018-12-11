from mxnet import nd

# insert ../ directory
# import os
# import sys
# _ = os.path.abspath(os.path.abspath(__file__) + '/../../')
# if _ not in sys.path:
#     sys.path.insert(0, _)

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