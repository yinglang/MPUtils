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

"""
    mnist stn net
    1. stn
    2. stn2 (separate rotate, scale, translate, distortion)
    3. theata (no localization stn, cann't get image info)
    4. theata2 (no localization stn, cann't get image info; separete rotate, scale, translate, distortion)
"""

@mx.init.register
class STInit(mx.init.Initializer):
    def __init__(self):
        super(STInit, self).__init__()
    def _init_weight(self, _, arr):
        arr[:] = nd.array([1, 0, 0, 
                           0, 1, 0], dtype=np.float32)

class STN_mnist(nn.HybridBlock):
    def __init__(self, ctx):
        super(STN_mnist, self).__init__()
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
                            nn.Activation('relu'),
                            nn.Dense(6, weight_initializer='zeros', bias_initializer='stinit'))
            
        self.initialize(ctx=ctx)
        self.data_shape=(28, 28)
    
    def hybrid_forward(self, F, x):
        xs = self.localization(x)
        # print(xs.shape)
        theata = self.fc_loc(xs)
        
        x = F.SpatialTransformer(x, theata, transform_type ='affine', sampler_type='bilinear', target_shape=self.data_shape)
        return x

    
from math import pi
class STN_mnist2(nn.HybridBlock):
    def __init__(self, ctx, use_clip=False, collect_transform=False, **kwargs):
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
        self.initialize(ctx=ctx)
        
        self.use_clip = use_clip
        if use_clip:
            ss = np.array(self.data_shape) * 1.0
            self.range = {'rotate': (-1, 1), 'translate_x': (-1, 1), 'translate_y': (-1, 1),
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
            rotate, scale_x = rotate.clip(*self.range['rotate']), scale_x.clip(*self.range['scale_x'])
            scale_y, translate_x = scale_y.clip(*self.range['scale_y']), translate_x.clip(*self.range['translate_x'])
            translate_y, distortion_x = translate_y.clip(*self.range['translate_y']), distortion_x.clip(*self.range['distortion_x'])
            distortion_y = distortion_y.clip(*self.range['distortion_y'])
        
        cos_a, sin_a = F.cos(rotate * pi), F.sin(rotate * pi)
        theata = F.concat(cos_a/scale_x, -sin_a/distortion_y, translate_x,
                          sin_a/distortion_x, cos_a/scale_y, translate_y, dim=1)
        
        x = F.SpatialTransformer(x, theata, transform_type ='affine',
                                         sampler_type='bilinear', target_shape=self.data_shape)
        
        if self.collect_transform:
            self.transforms = F.concat(rotate, scale_x, scale_y, translate_x, translate_y, distortion_x, 
                                            distortion_y, dim=1).asnumpy()
        return x
    
from math import pi, cos, sin
class Theata(nn.HybridBlock):
    def __init__(self, ctx, in_channels=1, **kwargs):
        super(Theata, self).__init__(**kwargs)
        with self.name_scope():
            pass
        if in_channels != 0:
            self.in_channels = in_channels
        # parameter visit: ?? gluon.Parameter
        self.theata = self.params.get('theata', shape=(in_channels, 6), init='stinit', allow_deferred_init=True)
        self.data_shape=(28, 28)
        self.initialize(ctx=ctx)
    
    # translate [-1, 1], rotate angle [-pi, pi] / pi, scale [MIN, MAX]
    def set(self, rotate, scale_x, scale_y, translate_x, translate_y):
        a, s, t = rotate * pi, (scale_x, scale_y), (translate_x, translate_y)
        self.theata.data()[:, :] = nd.array([cos(a)/s[0], -sin(a), t[0], sin(a), cos(a)/s[1], t[1]])
        
    # use params.get can make param pass to hybrid_forward function from forward function
    def hybrid_forward(self, F, x, theata):
        if theata.shape[0] == 1:
            xs = []
            for i in range(x.shape[0]):
                y = F.SpatialTransformer(x[i:i+1], theata, transform_type ='affine',
                                         sampler_type='bilinear', target_shape=self.data_shape)
                xs.append(y)
            x = nd.concat(*xs, dim=0)
        else:
            x = F.SpatialTransformer(x, theata[:x.shape[0], :], transform_type ='affine',
                                         sampler_type='bilinear', target_shape=self.data_shape)
        return x
    
class Theata2(nn.HybridBlock):
    def __init__(self, ctx, in_channels=1, use_clip=False, collect_transform=False, **kwargs):
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
        self.initialize(ctx=ctx)
        
        self.use_clip = use_clip
        if use_clip:
            ss = np.array(self.data_shape) * 1.0
            self.range = {'rotate': (-1, 1), 'translate_x': (-1, 1), 'translate_y': (-1, 1),
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
                values[k] = nd.random.uniform(*self.range[k], shape=shape)
            self.set(**values)
        
    # use params.get can make param pass to hybrid_forward function from forward function
    def hybrid_forward(self, F, x, rotate, log_scale_x, log_scale_y, translate_x, translate_y, 
                       log_distortion_x, log_distortion_y):
        scale_x, scale_y = F.exp(log_scale_x), F.exp(log_scale_y)
        distortion_x, distortion_y  = F.exp(log_distortion_x), F.exp(log_distortion_y)
        
        if self.use_clip:
            rotate, scale_x = rotate.clip(*self.range['rotate']), scale_x.clip(*self.range['scale_x'])
            scale_y, translate_x = scale_y.clip(*self.range['scale_y']), translate_x.clip(*self.range['translate_x'])
            translate_y, distortion_x = translate_y.clip(*self.range['translate_y']), distortion_x.clip(*self.range['distortion_x'])
            distortion_y = distortion_y.clip(*self.range['distortion_y'])
        
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
    
"""
    cnn classify net for mnist
"""    
class Net_mnist(nn.HybridBlock):
    def __init__(self, ctx=mx.gpu(0), **kwargs):
        super(Net_mnist, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential()
            self.features.add(
                nn.Conv2D(10, kernel_size=5),
                nn.MaxPool2D(2),
                nn.Activation('relu'),
                nn.Conv2D(20, kernel_size=5),
                nn.Dropout(0.5),
                nn.MaxPool2D(2),
                nn.Activation('relu')
            )
            
            self.classifier = nn.HybridSequential()
            self.classifier.add(
                nn.Dense(50, activation='relu'),
                nn.Dropout(0.5),
                nn.Dense(10)
            )
        
        self.features.initialize(ctx=ctx)
        self.classifier.initialize(ctx=ctx)
    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
"""
    stn wrapper
""" 
class STNWrapper(nn.HybridBlock):
    def __init__(self, NetCls, STNCls, netargs=[], stnargs=[], netkwargs={}, stnkwargs={}, **kwargs):
        super(STNWrapper, self).__init__(**kwargs)
        with self.name_scope():
            self.stn = STNCls(*stnargs, **stnkwargs)
            self.net = NetCls(*netargs, **netkwargs)
            
    def hybrid_forward(self, F, x):
        x = self.stn(x)
        return self.net(x)