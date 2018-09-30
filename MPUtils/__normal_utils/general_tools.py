"""
general tools function
"""
import os
class bzip(object):
    """
        just like zip, but diffrent is not need get all data to memory first, use lazy load.
    """
    def __init__(self, *args):
        self.args = args
        
    def __iter__(self):
        self.iters = []
        for arg in self.args:
            self.iters.append(arg.__iter__())
        return self
        
    def next(self):
        res = []
        for _iter in self.iters:
            res.append(_iter.next())
        return tuple(res)
    
def mkdir_if_not_exist(*path):
    path = os.path.join(*path)
    if not os.path.exists(path):
        os.makedirs(path)

def inv_normalize(data, rgb_mean, std):
    return (data.transpose((0, 2, 3, 1)) * std + rgb_mean).transpose((0, 3, 1, 2))

