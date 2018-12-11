import mxnet as mx
from mxnet import nd

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


class RoundConstraint(Constraint):
    def _inv_normalize(self, data, mean=None, std=None, clip=True):
        std, mean = std.as_in_context(data.context), mean.as_in_context(data.context)
        images = data.transpose((0, 2, 3, 1))
        images = images * std + mean
        if clip: 
            images = images.clip(0, 1)
        images = nd.round(images * 255) / 255
        images = (images - mean) / std
        data[:, :, :, :] = images.transpose((0, 3, 1, 2))
    
    def __init__(self, inv_normalize=None, **kwargs):
        if inv_normalize is None:
            self.inv_normalize = self._inv_normalize
        else:
            self.inv_normalize = inv_normalize
        self.kwargs = kwargs
    
    def __call__(self, data, iters, max_iters):
        """
            invoke every backgrad iter
        """
        self.inv_normalize(data, **self.kwargs)
        #show_data(data[:5])