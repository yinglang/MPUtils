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
from math import pi, cos, sin
from .layer_control import *
import matplotlib.pyplot as plt
from .mutils import *

"""
1. transform net
"""

class Theata2(nn.HybridBlock):
    def __init__(self, in_channels=1, use_clip=False, collect_transform=False, **kwargs):
        super(Theata2, self).__init__()
        with self.name_scope():
            pass
        if in_channels != 0:
            self.in_channels = in_channels
        # parameter visit: ?? gluon.Parameter
        # self.theata = self.params.get('theata', shape=(in_channels, 6), init='stinit', allow_deferred_init=True, grad_req='null')
        self.rotate = self.params.get('rotate', shape=(in_channels, 1), init='zeros', allow_deferred_init=True)
        self.log_scale_x = self.params.get('log_scale_x', shape=(in_channels, 1), init='zeros', allow_deferred_init=True)
        self.log_scale_y = self.params.get('log_scale_y', shape=(in_channels, 1), init='zeros', allow_deferred_init=True)
        self.translate_x = self.params.get('translate_x', shape=(in_channels, 1), init='zeros', allow_deferred_init=True)
        self.translate_y = self.params.get('translate_y', shape=(in_channels, 1), init='zeros', allow_deferred_init=True)
        self.log_distortion_x = self.params.get('log_distortion_x', shape=(in_channels, 1), init='zeros', allow_deferred_init=True)
        self.log_distortion_y = self.params.get('log_distortion_y', shape=(in_channels, 1), init='zeros', allow_deferred_init=True)
        self.data_shape=(28, 28)
        
        self.use_clip = use_clip
        if use_clip:
            ss = np.array(self.data_shape) * 1.0
            self.range = {'rotate': None, 'translate_x': (-1, 1), 'translate_y': (-1, 1),
                          'scale_x': (1/ss[1], ss[1]), 'scale_y': (1/ss[0], ss[0]),
                          'distortion_x': (1/ss[1], ss[1]), 'distortion_y': (1/ss[0], ss[0]),
                          'log_scale_x': (-1, -1), 'log_scale_y':(-1, -1), 'log_distortion_x': (-1, -1),#should not specify by user
                          'log_distortion_y': (-1, -1)}
            for key in self.range:
                if key + '_range' in kwargs:
                    self.range[key] = kwargs[key + '_range']
                    if key not in ['rotate', 'translate_x', 'translate_y']:
                        self.range['log_' + key] = np.log(np.array(self.range[key]))

        self.collect_transform = collect_transform
    
    # log make 2.0 and 0.5 have same offset from 1, |log2 - log1| == |log0.5 - log1|
    # translate [-1, 1], rotate angle [-pi, pi] / pi, scale [MIN, MAX]
    def set(self, rotate=0, log_scale_x=0, log_scale_y=0, translate_x=0, translate_y=0, log_distortion_x=0, log_distortion_y=0):
        self.rotate.data()[:], self.log_scale_x.data()[:], self.log_scale_y.data()[:] = rotate, log_scale_x, log_scale_y
        self.translate_x.data()[:], self.translate_y.data()[:] = translate_x, translate_y
        self.log_distortion_x.data()[:], self.log_distortion_y.data()[:] = log_distortion_x, log_distortion_y
        
    def reset(self, TYPE='zeros'):
        if TYPE == 'zeros':
            self.set()
        elif TYPE == 'random':
            shape = self.rotate.data().shape
            values = {'rotate': 0, 'log_scale_x': 0, 'log_scale_y': 0, 'translate_x': 0, 'translate_y':0,
                 'log_distortion_x': 0, 'log_distortion_y': 0}
            for k in values:
                if self.range[k] is not None:
                    values[k] = nd.random.uniform(*self.range[k], shape=shape)
                elif k == 'rotate':
                    values[k] = nd.random.uniform(-1, 1, shape=shape)
            self.set(**values)
        else:
            raise ValueError("TYPE must be one of [zeros, random]")
        
    # use params.get can make param pass to hybrid_forward function from forward function
    def hybrid_forward(self, F, x, rotate, log_scale_x, log_scale_y, translate_x, translate_y, 
                       log_distortion_x, log_distortion_y):
        scale_x, scale_y = F.exp(log_scale_x), F.exp(log_scale_y)
        distortion_x, distortion_y  = F.exp(log_distortion_x), F.exp(log_distortion_y)
        
        if self.use_clip:
            if self.range['rotate'] is not None:
                rotate = rotate.clip(*self.range['rotate'])
            scale_x, scale_y = scale_x.clip(*self.range['scale_x']), scale_y.clip(*self.range['scale_y'])
            translate_x, translate_y = translate_x.clip(*self.range['translate_x']), translate_y.clip(*self.range['translate_y'])
            distortion_x, distortion_y = distortion_x.clip(*self.range['distortion_x']), distortion_y.clip(*self.range['distortion_y'])
        
        cos_a, sin_a = F.cos(rotate * pi), F.sin(rotate * pi)
        theata = F.concat(cos_a/scale_x, -sin_a/distortion_y, translate_x,
                          sin_a/distortion_x, cos_a/scale_y, translate_y, dim=1)
        if theata.shape[0] == 1:
            xs = []
            for i in range(x.shape[0]):
                y = F.SpatialTransformer(x[i:i+1], theata, transform_type ='affine',
                                         sampler_type='bilinear', target_shape=self.data_shape)
                xs.append(y)
            x = F.concat(*xs, dim=0)
        else:
            x = F.SpatialTransformer(x, theata[:x.shape[0], :], transform_type ='affine',
                                         sampler_type='bilinear', target_shape=self.data_shape)
            
        if self.collect_transform:
            self.transforms = F.concat(rotate, scale_x, scale_y, translate_x, translate_y, distortion_x, 
                                            distortion_y, dim=1).asnumpy()
        return x
    
from math import pi
class STN_mnist2(nn.HybridBlock):
    def __init__(self, use_clip=False, collect_transform=False, **kwargs):
        super(STN_mnist2, self).__init__()
        with self.name_scope():
            self.localization = nn.HybridSequential()
            self.localization.add(
                nn.Conv2D(8, kernel_size=7),
                nn.MaxPool2D(2),
                nn.Activation('relu'),
                nn.Conv2D(10, kernel_size=5),
                nn.MaxPool2D(2),
                nn.Activation('relu')
            )
            
            self.fc_loc = nn.HybridSequential()
            self.fc_loc.add(nn.Flatten(),
                            nn.Dense(32),
                            nn.Activation('relu'))
            
            self.rotate = nn.Dense(1, weight_initializer='zeros', bias_initializer='zeros')
            self.log_scale_x = nn.Dense(1, weight_initializer='zeros', bias_initializer='zeros')
            self.log_scale_y = nn.Dense(1, weight_initializer='zeros', bias_initializer='zeros')
            self.translate_x = nn.Dense(1, weight_initializer='zeros', bias_initializer='zeros')
            self.translate_y = nn.Dense(1, weight_initializer='zeros', bias_initializer='zeros')
            self.log_distortion_x = nn.Dense(1, weight_initializer='zeros', bias_initializer='zeros')
            self.log_distortion_y = nn.Dense(1, weight_initializer='zeros', bias_initializer='zeros')
            
        self.data_shape=(28, 28)
        
        self.use_clip = use_clip
        if use_clip:
            ss = np.array(self.data_shape) * 1.0
            self.range = {'rotate': None, 'translate_x': (-1, 1), 'translate_y': (-1, 1),
                          'scale_x': (1/ss[1], ss[1]), 'scale_y': (1/ss[0], ss[0]),
                          'distortion_x': (1/ss[1], ss[1]), 'distortion_y': (1/ss[0], ss[0])}
            for key in self.range:
                if key + '_range' in kwargs:
                    self.range[key] = kwargs[key + '_range']
                    
        self.collect_transform = collect_transform
    
    # log make 2.0 and 0.5 have same offset from 1, |log2 - log1| == |log0.5 - log1|
    # translate [-1, 1], rotate angle [-pi, pi] / pi, scale [MIN, MAX]
    # use params.get can make param pass to hybrid_forward function from forward function
    def hybrid_forward(self, F, x):
        xs = self.localization(x)
        lf = self.fc_loc(xs)
        rotate, scale_x, scale_y = self.rotate(lf), F.exp(self.log_scale_x(lf)), F.exp(self.log_scale_y(lf))
        translate_x, translate_y = self.translate_x(lf), self.translate_y(lf)
        distortion_x, distortion_y  = F.exp(self.log_distortion_x(lf)), F.exp(self.log_distortion_y(lf))
        
        if self.use_clip:
            if self.range['rotate'] is not None:
                rotate = rotate.clip(*self.range['rotate'])
            scale_x, scale_y = scale_x.clip(*self.range['scale_x']), scale_y.clip(*self.range['scale_y'])
            translate_x, translate_y = translate_x.clip(*self.range['translate_x']), translate_y.clip(*self.range['translate_y'])
            distortion_x, distortion_y = distortion_x.clip(*self.range['distortion_x']), distortion_y.clip(*self.range['distortion_y'])
        
        cos_a, sin_a = F.cos(rotate * pi), F.sin(rotate * pi)
        theata = F.concat(cos_a/scale_x, -sin_a/distortion_y, translate_x,
                          sin_a/distortion_x, cos_a/scale_y, translate_y, dim=1)
        
        x = F.SpatialTransformer(x, theata, transform_type ='affine',
                                         sampler_type='bilinear', target_shape=self.data_shape)
        
        if self.collect_transform:
            self.transforms = F.concat(rotate, scale_x, scale_y, translate_x, translate_y, distortion_x, 
                                            distortion_y, dim=1).asnumpy()
        return x
 
"""
    2. generator
"""
def measure(output, label):
    res = output.argmax(axis=1).reshape(label.shape) == label
    acc = nd.mean(res).asscalar()
    wrong_idx = np.array(range(output.shape[0]))[(1 - res).asnumpy().astype('bool').reshape((-1,))]
    return acc, wrong_idx

def generator(data, label, transform_net, net, loss_f, trainer, max_iter, start='random', labelmap=None, labeloutmap=None):
    """
    Parameters
    ------------------------------------------------------------------------------------------------------------
    data:
    label:
    transform_net:
    net: trained base network
    loss_f: loss function, if labelmap and labeloutput is None, it will max loss, or will min loss
    trainer: 
    max_iter:
    start: tansform net start stat
    labelmap:
    labeloutmap:
    
    Example:
    -------------------------------------------------------------------------------------------------------------
    batch_size = 256
    train_data, valid_data = data_loader(batch_size)

    # define trained net
    net = Net_mnist(ctx, prefix='net0_')
    net.load_params('../../params/mnist_mxnet/%s.param' % 'stn_test_3', ctx=ctx)

    # define transform net
    ranges={'translate_x_range': (-0.2, 0.2), 'translate_y_range': (-0.2, 0.2), 'scale_x_range': (1.0/1.2, 1.2),
            'scale_y_range': (1.0/1.2, 1.2), 'rotate_range': (-0.2, 0.2),
            'distortion_x_range': (1.0, 1.0), 'distortion_y_range': (1.0, 1.0)}
    transform_net = Theata2(batch_size, True, True, **ranges)
    transform_net.initialize(ctx=ctx)

    # define loss
    loss_f = SCELoss(sparse_label=True)

    # begin generator
    for data, label in train_data:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        trainer = gluon.Trainer(transform_net.collect_params(), 'adam', 
                                {'learning_rate': 0.1, 'beta1': 0.9, 'beta2':0.999, 'epsilon':1e-8,'wd': 1e-4})
        # transform_net.reset('zeros')
        fdata, dbg_info = generator(data, label, transform_net, net, loss_f, trainer, max_iter=10, start='random')
        show_generator_result(data, fdata, dbg_info)
        break
    """
    if start == 'continue':
        pass
    else: # 'zeros' or 'random'
        transform_net.reset(start)
    if labelmap is not None:
        maplabel = labelmap(label)
        
    accs, wrong_idxs, transforms, losses = [], [], [], []
    dpcontrol = DPControl(net); bncontrol = BNControl(net)
    dpcontrol.save(); bncontrol.save()
    for i in range(max_iter):
        with autograd.record():
            fdata = transform_net(data)
            output = net(fdata)
            if labelmap is None and labeloutmap is None:
                loss = -loss_f(output, label)
            else:
                if labeloutmap is not None:
                    maplabel = labeloutmap(output, label)
                loss = loss_f(output, maplabel)
        loss.backward()
        losses.append(nd.mean(loss).asscalar())
        trainer.step(data.shape[0])
        
        acc, wrong_idx = measure(output, label)
        accs.append(acc); wrong_idxs.append(wrong_idx)
        
        transforms.append(transform_net.transforms)
    dpcontrol.load(); bncontrol.load()
    return fdata, (transforms, losses, accs, wrong_idxs)

def show_generator_result(data=None, fdata=None, dbg_info=tuple()):
    """
        pring log of generator
    """
    transforms, losses, accs, wrong_idxs = dbg_info
    print("loss(%d iter):\n" % len(losses), losses) 
    print("acc(%d iter) :\n" % len(accs), accs)
    
    transforms = np.array(transforms)
    sign = np.sign(np.mean(transforms, axis=(0, 1)))
    print("rotate, scale_x, scale_y, translate_x, translate_y, distortion_x, distortion_y:\n",
          sign * np.mean(np.abs(transforms), axis=(0, 1)))
    
    plt.figure(figsize=(12, 4))
    plt.title("loss and acc")
    plt.subplot(1, 2, 1)
    plt.plot(range(len(losses)), losses)
    plt.subplot(1, 2, 2)
    plt.plot(range(len(accs)), accs)
    plt.show()

    if data is not None:
        show_images(data.asnumpy()[wrong_idxs[-1][:10]]*255, figsize=(12, 8), MN=(1, 10))
    if fdata is not None:
        show_images(fdata.asnumpy()[wrong_idxs[-1][:10]]*255, figsize=(12, 8), MN=(1, 10))

class DDDALayer(nn.HybridBlock):
    """
    Example:
    -------------------------------------------------------------------------------------------------
    net = Net_mnist(ctx, prefix='net0_')
    net.load_params('../../params/mnist_mxnet/%s.param' % 'stn_test_3', ctx=ctx)
    ddda = DDDALayer()
    for data, label in train_data:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        fdata = ddda(data, label, net)
        output = net(fdata)
        show_generator_result(data, fdata, ddda.dbg_info)
        break
    """
    def __init__(self, ctx, labelmap=None, labeloutmap=None, loss_f=SCELoss(), max_iter=10, 
                 transform_net_cls=Theata2, tnet_args=[-1, True, True], # -1 means infer by first forawrd
                 tnet_kwargs={'translate_x_range': (-0.2, 0.2), 'translate_y_range': (-0.2, 0.2),
                              'scale_x_range': (1.0/1.2, 1.2), 'scale_y_range': (1.0/1.2, 1.2), 'rotate_range': (-0.2, 0.2),
                              'distortion_x_range': (1.0, 1.0), 'distortion_y_range': (1.0, 1.0)},
                 trainer_args=['adam', {'learning_rate': 0.1, 'beta1': 0.9, 'beta2':0.999, 'epsilon':1e-8,'wd': 1e-4}], 
                 **kwargs):
        super(DDDALayer, self).__init__(**kwargs)
        
        if tnet_args[0] > 0:
            self.transform_net = transform_net_cls(*tnet_args, **tnet_kwargs)
            self.transform_net.initialize(ctx=ctx)
        else:
            self.transform_net = None
            self.transform_net_cls = transform_net_cls
            self.tnet_args = tnet_args
            self.tnet_kwargs = tnet_kwargs
            self.ctx=ctx
        
        self.trainer_args = trainer_args
        self.labelmap = labelmap
        self.labeloutmap = labeloutmap
        self.max_iter = max_iter
        self.loss_f = loss_f
    
    def hybrid_forward(self, F, data, label, net):
        if self.transform_net is None:
            self.tnet_args[0] = data.shape[0] # batch_size
            self.transform_net = self.transform_net_cls(*self.tnet_args, **self.tnet_kwargs)
            self.transform_net.initialize(ctx=self.ctx)
            
        trainer = gluon.Trainer(self.transform_net.collect_params(), *self.trainer_args)
        fdata, dbg_info = generator(data, label, self.transform_net, net, self.loss_f, trainer, max_iter=self.max_iter, 
                                    start='random', labelmap=self.labelmap, labeloutmap=self.labeloutmap)
        
        self.dbg_info = dbg_info
        return fdata
