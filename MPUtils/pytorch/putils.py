import torch

def clip_f(data, down, up):
    one = torch.ones(1)
    data.where(data >= down , down * one) # data if condition else down
    data.where(data <= up, up * one)
    return data
clip = clip_f

def inv_normalize(data, mean=None, std=None, clip=True, asnumpy=True):
    if mean is None: mean=0
    if std is None: std=1
    images = data.transpose(1, 2).transpose(2, 3)
    images = images * std + mean
    images = images.transpose(3, 2).transpose(2, 1) * 255
    if clip: 
        images = clip_f(images, 0, 255)
    if asnumpy:
        images = images.numpy()
    return images