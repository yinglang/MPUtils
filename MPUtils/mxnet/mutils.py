# insert last dir
import os
import sys
#_ = os.path.abspath(os.path.abspath(__file__) + '/../../')
#if _ not in sys.path:
#    sys.path.insert(0, _)

from ..__normal_utils import *
from . import log_parse

"""
all about:
    1. data prepare
    2. train
    3. data visualize
"""

"""
    data_prepared
"""
from gluoncv.data.batchify import Tuple, Stack, Pad
from mxnet.gluon.data import DataLoader
from mxnet import autograd, nd, gluon
def tri_data_loader(batch_size, transform_train, transform_valid, num_workers=0,
                train_dataset=None, valid_dataset=None, valid_train=False, 
                batchify_fns={'valid': None, 'train': None, 'valid_train':None}):
    """
            batchify_fns: dict(), like in ssd
                       {'train': Tuple(Stack(), Stack(), Stack()),
                        'valid': Tuple(Stack(), Pad(pad_val=-1)),
                        'valid_train': Tuple(Stack(), Stack(), Stack())}
    """
    # 1. valid_data
    valid_data = DataLoader(valid_dataset.transform(transform_valid), batch_size, 
                        shuffle=False, batchify_fn=batchify_fns['valid'], last_batch='keep',
                        num_workers=num_workers)
    
    # 2. train_data
    train_data = DataLoader(train_dataset.transform(transform_train), batch_size,
                             shuffle=True, batchify_fn=batchify_fns['train'], last_batch='rollover',
                             num_workers=num_workers)
    
    # 3. valid_train_data
    if valid_train == False:
        return train_data, valid_data, train_dataset.classes
    else:
        valid_train_data = DataLoader(train_dataset.transform(transform_valid), batch_size, 
                            shuffle=False, batchify_fn=batchify_fns['valid_train'], last_batch='keep',
                            num_workers=num_workers)
        return train_data, valid_data, train_dataset.classes, valid_train_data

"""
 data visulize
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mxnet import nd, gluon, autograd
import mxnet as mx
import os
import shutil
import cv2 as cv
from math import sqrt, ceil, floor

def merge_data(data, **kwargs):
    imgs = list(data.transpose((0, 2, 3, 1)).asnumpy())
    return merge_images(imgs, **kwargs)

def show_merge_data(data, rgb_mean=0, std=1, MN=None, clip=False, **kwargs):
    data = inv_normalize(data, rgb_mean, std=std)
    img = merge_data(data, MN=MN)
    if clip:
        img = img.clip(0, 1)
    plt.imshow(img, **kwargs)
    plt.axis('off')
    plt.show()

def inv_normalize(data, rgb_mean, std):
    return (data.transpose((0, 2, 3, 1)) * std + rgb_mean).transpose((0, 3, 1, 2))

def show_data(data, rgb_mean=0, std=1, **kwargs):
    data = inv_normalize(data, rgb_mean, std=std)
    show_images(data.asnumpy(), **kwargs)

def try_asnumpy(data):
    try:
        data = data.asnumpy() # if is <class 'mxnet.ndarray.ndarray.NDArray'>
    except BaseException:
        pass
    return data

def tonumpy(*args):
    res = []
    for arg in args:
        res.append(arg.asnumpy())
    return res
    
"""
    train
"""
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def accuracy(output, label):
    return nd.sum(output.argmax(axis=1)==label.reshape((output.shape[0],))).asscalar()

def one_hot_accuracy(output, label):
    output = output.reshape((output.shape[0], -1))
    label = label.reshape((label.shape[0], -1))
    return nd.sum(output.argmax(axis=1)==label.argmax(axis=1)).asscalar()

def _get_batch(batch, ctx):
    """return data and label on ctx"""
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.label[0]
    else:
        data, label = batch
    return data.as_in_context(ctx), label.as_in_context(ctx)

def evaluate_accuracy(data_iterator, net, ctx=mx.cpu(), loss_f=None):
    global one_hot_accuracy, accuracy
    _accuracy = one_hot_accuracy if hasattr(loss_f, '_sparse_label') and (not loss_f._sparse_label) else accuracy
    acc = 0.
    loss = 0.
    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()
    image_count = 0
    for i, batch in enumerate(data_iterator):
        data, label = _get_batch(batch, ctx)
        output = net(data)
        acc += _accuracy(output, label)
        if loss_f is not None:
            loss += nd.sum(loss_f(output, label)).asscalar()
        image_count += data.shape[0]
    if loss_f is not None:
        return acc/ image_count, loss/image_count
    return acc / image_count


import datetime
import sys
from random import random
from mxnet import gluon, nd

def abs_mean(W):
    return nd.mean(nd.abs(W)).asscalar()

def in_list(e, l):
    for i in l:
        if i == e:
            return True
    else:
        return False
    
def bak_file_if_exists(path):
    tmp = path[:]
    if not os.path.exists(path): return
    path = path + '.bak'
    while os.path.exists(path):
        path += '.bak'
    shutil.copy(tmp, path)
    
    
class TrainPipeline(object):
    def __init__(self, net, train_data, valid_data, start_epoch, num_epochs, policy=None, ctx=mx.cpu(), w_key=[], trainers=None, 
                 output_file=None, verbose=False, loss_f=gluon.loss.SoftmaxCrossEntropyLoss(), mixup_alpha=None,
                 back_grad_args=None, log=['train_loss', 'train_acc', 'valid_loss', 'valid_acc', 'time'], verbose_iter=1, **kwargs):
        """
            policy: {'lr':xx, 'wd':xx, 'lr_period':xx, 'lr_decay':xx}
            trainers: gluon.Trainer, one of policy or trainers must be specified.
        """
        self.net, self.train_data, self.valid_data, self.start_epoch = net, train_data, valid_data, start_epoch
        self.num_epochs, self.policy, self.ctx, self.w_key, self.trainers = num_epochs, policy, ctx, w_key, trainers
        self.output_file, self.verbose, self.loss_f, self.mixup_alpha = output_file, verbose, loss_f, mixup_alpha
        self.back_grad_args = back_grad_args
        self.log, self.verbose_iter = log, verbose_iter
        
    def initialize(self):
        """
            invoke before train
            1. reset output file
            2. set prev_time for cal cost time
            3. init trainers
            4. verbose set True print test_acc in valid_data
        """
        if self.output_file is None:
            self.output_file = sys.stdout
            self.stdout = sys.stdout
        else:
            bak_file_if_exists(self.output_file)
            self.output_file = open(self.output_file, "w")
            self.stdout = sys.stdout
            sys.stdout = self.output_file
       
        self.prev_time = datetime.datetime.now()  
        self.local = {} # to record some local var for passing between functions

        if self.verbose and self.valid_data is not None:
            print(" # {}".format(evaluate_accuracy(self.valid_data, self.net, self.ctx)))
            
        if self.trainers is None:
            if self.policy is None:
                raise ValueError('traniers or poly must be specified. but they are both None.')
            self.trainers = [gluon.Trainer(self.net.collect_params(), 'sgd', 
                                  {'learning_rate': self.policy['lr'], 'momentum': 0.9, 'wd': self.policy['wd']})]
        else:
            if self.policy is not None and ('lr' in self.policy or 'wd' in self.policy):
                print(" # [Warning]: trainers has been specified, policy will be ignore.")
        self.result = {'valid_acc': [], 'train_acc': [], 'train_loss': [], 'valid_loss': []}
        
        global one_hot_accuracy, accuracy
        self._accuracy = one_hot_accuracy if hasattr(self.loss_f, '_sparse_label') and (not self.loss_f._sparse_label) else accuracy
    
    def after_epoch(self, epoch, train_loss, train_acc, others=[]):
        """
            invoke after every epoch of train
            1. cal and print cost time the epoch
            2. print acc/loss info
            3. print lr
            4. update lr
        """
        train_loss /= len(self.train_data)
        train_acc /= self.local['image_count']
        if train_acc < 1e-6:
            train_acc = evaluate_accuracy(self.train_data, self.net, self.ctx)
        self.result['train_acc'].append(train_acc)
        self.result['train_loss'].append(train_loss)

        epoch_str = ""
        if self.valid_data is not None and 'valid_acc' in self.log:
            if 'valid_loss' in self.log:
                valid_acc, valid_loss = evaluate_accuracy(self.valid_data, self.net, self.ctx, self.loss_f)
                epoch_str += ("epoch %d, loss %.5f, train_acc %.4f, valid_loss %.5f, valid_acc %.4f" 
                         % (epoch, train_loss, train_acc, valid_loss, valid_acc))
            else:
                valid_acc = evaluate_accuracy(self.valid_data, self.net, self.ctx)
                epoch_str += ("epoch %d, loss %.5f, train_acc %.4f, valid_acc %.4f" 
                         % (epoch, train_loss, train_acc, valid_acc))
            self.result['valid_acc'].append(valid_acc)
            self.result['valid_loss'].append(valid_loss)
            
        else:
            epoch_str += ("epoch %d, loss %.5f, train_acc %.4f"
                        % (epoch, train_loss, train_acc))
        self.output_file.write(epoch_str + ", ")
        
        if self.policy is not None and in_list(epoch+1, self.policy['lr_period']):
            for trainer in self.trainers:
                trainer.set_learning_rate(trainer.learning_rate * self.policy['lr_decay'])
                
        self.add_after_epoch(epoch)    # excute for user simple define
        
        # log info
        self.cur_time = datetime.datetime.now()
        h, remainder = divmod((self.cur_time - self.prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        self.prev_time = self.cur_time
  
        self.output_file.write(time_str + ",lr " + str([trainer.learning_rate for trainer in self.trainers]) + "\n")
        self.output_file.flush()  # to disk only when flush or close
    
    def after_iter(self, i, _loss, _acc):
        """
            invoke after every iteration
            1. print iter losss and acc
            2. print weight and grad for every iter
        """
        if self.verbose and i % self.verbose_iter == 0:
            print(" # iter " + str(i), end=' ')
            print(("loss %.5f" % _loss) + (" acc %.5f" % _acc), end=' ')
            print("w (", end=' ')
            for k in self.w_key:
                w = self.net.collect_params()[k]
                print("%.5f, " % abs_mean(w.data()), end=' ')
            print(") g (", end=' ')
            for k in self.w_key:
                w = self.net.collect_params()[k]
                print("%.5f, " % abs_mean(w.grad()), end=' ')
            print(")")
            
        self.add_after_iter(i)      # excute for user simple define
        
    def add_after_iter(self, i):
        pass
    def add_after_epoch(self, e):
        pass
        
    def run(self, log_period=1):
        self.initialize()
        
        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):
            train_loss, train_acc, i, self.local['image_count'] = 0., 0., 0, 0.
            for data, label in self.train_data:
                with autograd.record():
                    data, label = data.as_in_context(self.ctx), label.as_in_context(self.ctx)
                    output = self.net(data)
                    loss = self.loss_f(output, label)
                loss.backward()
                for trainer in self.trainers:
                    trainer.step(data.shape[0])
                
                _loss = nd.mean(loss).asscalar() 
                _acc = self._accuracy(output, label)
                train_loss += _loss
                train_acc += _acc
                
                self.after_iter(i, _loss, _acc)
                i += 1
                self.local['image_count'] += data.shape[0]
            self.after_epoch(epoch, train_loss, train_acc)
        
        if self.output_file == sys.stdout:
            sys.stdout = self.stdout
            if self.output_file != self.stdout: self.output_file.close()
        return self.result
