from mxnet.gluon.loss import SoftmaxCrossEntropyLoss as SCELoss
from BNControl import *
from optimizer import *
from constraint import *
from labelmap import *
from LossCalculator import *
from logger import *

# insert ../ directory
import os
import sys
_ = os.path.abspath(os.path.abspath(__file__) + '/../../')
if _ not in sys.path:
    sys.path.insert(0, _)
    
from mutils import show_images, inv_normalize
import mxnet as mx
from mxnet import gluon, nd, autograd
from mxnet.gluon import loss, nn
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss as SCELoss
import os
def mkdir_if_not_exist(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    
def WFlip(data): # axis=3
    data[:, :, :, :] = data[:, :, :, ::-1]

def generate_backgrad_data(net, data, label, ctx=mx.gpu(0), bn_control=None, 
                           max_iters=60, sgd=SGD(lr=0.1), # optimizer args
                           iter_log_period=None, show_clip=False, record_detail=False, logger=None,# log args
                           labelmap=None, labeloutputmap=None, loss_f=SCELoss(), # loss_args
                           threshold=None, flip=False, constraint=None):#constraint_agrs
    """
        param:
            net: base net model
            data: data will be changed.recomand data context in cpu.
            label: data's label, recomand label context in cpu.
            ctx: backgrad context, if ctx is gpu and data/label in gpu(or said they in same), then backgradwill change the data iteself, 
            max_iters: max_iters for backgrad
            lr: lr for backgrad
            iter_log_period: output log period, None means never output.
            show_clip: log show backgrad image is clip?
            loss_f: loss function for backgrad
            bn_control: bn_control be fore backgrad, None means never use BN Control.
            sgd: backgrad optimizer method.
            trheshold: returned data's color clip trheshold.
            labelmap, labeloutputmap: decide use what label to backgrad generate adversal smaple. use -loss when all None. only one canbe specified.

        data is better in cpu, if data in ctx(global var), the returned backgrad_data is shallow copy of data.
    """        
    if bn_control is not None:
        bn_control.store()

    sparse_label = label.copy()
    if labelmap is not None: label = labelmap(sparse_label)
    data, label, sparse_label_ctx = data.as_in_context(ctx), label.as_in_context(ctx), sparse_label.as_in_context(ctx)
    if logger is None: logger = LogRecorder()
    if record_detail:
        origin_data = data.copy()
    if constraint is None:
        constraint = Constraint(threshold)
        
    for iters in range(1, max_iters+1):
        if flip and iters % 2 == 0: WFlip(data)
        with autograd.record():
            data.attach_grad()
            output = net(data)
            if labeloutputmap is not None: label = labeloutputmap(output, sparse_label).as_in_context(ctx)
            loss = loss_f(output, label)
            if labeloutputmap is None and labelmap is None:
                loss = -loss
        loss.backward()
        mean_loss = nd.mean(loss).asscalar()     # reduce will make memory release
        if flip and iters % 2 == 0: 
            WFlip(data)
            WFlip(data.grad)
                
        sgd(data)
        
        constraint(data, iters, max_iters)
        
        if record_detail:
            logger(output, sparse_label_ctx, origin_data, data, loss)
        logger.print_log(iter_log_period, iters, mean_loss, record_detail, data, show_clip)

    if bn_control is not None:
        bn_control.load()

    logger.asnumpy()
    sgd.clear()
    return data, (logger, )
    
def generate_backgrad_data_def_loss(net, data, label, ctx=mx.gpu(0), bn_control=None, 
                           max_iters=60, sgd=SGD(lr=0.1), # optimizer args
                           iter_log_period=None, show_clip=False, record_detail=False, logger=None,# log args
                           labelmap=None, labeloutputmap=None, loss_f=LossCalculator(), # loss_args
                           threshold=None, flip=False, constraint=None):#constraint_agrs
    """
        param:
            net: base net model
            data: data will be changed.recomand data context in cpu.
            label: data's label, recomand label context in cpu.
            ctx: backgrad context, if ctx is gpu and data/label in gpu(or said they in same), then backgradwill change the data iteself, 
            max_iters: max_iters for backgrad
            lr: lr for backgrad
            iter_log_period: output log period, None means never output.
            show_clip: log show backgrad image is clip?
            loss_f: loss function for backgrad
            bn_control: bn_control be fore backgrad, None means never use BN Control.
            sgd: backgrad optimizer method.
            trheshold: returned data's color clip trheshold.
            labelmap, labeloutputmap: decide use what label to backgrad generate adversal smaple. use -loss when all None. only one canbe specified.

        data is better in cpu, if data in ctx(global var), the returned backgrad_data is shallow copy of data.
    """        
    if bn_control is not None:
        bn_control.store()

    sparse_label = label.copy()
    if labelmap is not None: label = labelmap(sparse_label)
    data, label, sparse_label_ctx = data.as_in_context(ctx), label.as_in_context(ctx), sparse_label.as_in_context(ctx)
    if logger is None: logger = LogRecorder()
    if record_detail:
        origin_data = data.copy()
    if constraint is None:
        constraint = Constraint(threshold)
        
    data.attach_grad()
    for iters in range(1, max_iters+1):
        if flip and iters % 2 == 0:  WFlip(data)
        with autograd.record():
            output ,loss, loss_args = loss_f(net, data, label, labelmap, labeloutputmap, sparse_label, ctx)
        loss.backward()
        mean_loss = nd.mean(loss).asscalar()     # reduce will make memory release
        if flip and iters % 2 == 0: 
            WFlip(data)
            WFlip(data.grad)
            
        sgd(data)
        
        constraint(data, iters, max_iters)
        
        if record_detail:
            logger(output, sparse_label_ctx, origin_data, data, loss)
        logger.print_log(iter_log_period, iters, mean_loss, record_detail, data, show_clip, loss_args=loss_args)

        
    if bn_control is not None:
        bn_control.load()

    logger.asnumpy()
    sgd.clear()
    return data, (logger, )

def adversarial_acc(net, data_iter, bn_control, generator=generate_backgrad_data, **kwargs):
    bn_control.store()
    i= 0
    acc, mse, loss, result_score, label_score, image_count = None, None, None, None, None, 0
    for data, label in data_iter:
        data, (logger,) = generator(net, data, label, bn_control=None, **kwargs)
        if acc is None: 
            acc, mse, loss = logger.acc, logger.sse, logger.losses
            result_score, label_score = logger.log_result_score, logger.log_label_score
        else: 
            acc, mse, loss = acc + logger.acc, mse + logger.sse, loss + logger.losses
            result_score, label_score = result_score + logger.log_result_score, label_score + logger.log_label_score
        image_count += data.shape[0]
        #if i > 3 : break
        i += 1
    acc, mse, loss = acc / image_count, mse / image_count, loss / image_count
    result_score, label_score = np.exp(result_score / image_count), np.exp(label_score / image_count)
    bn_control.load()
    return acc, mse, loss, result_score, label_score

"""
    generate and save adversarial sample for a dataset
"""
class LoggerSum(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def __call__(self, **kwargs):
        for key in kwargs:
            if self.kwargs[key] is None:
                self.kwargs[key] = kwargs[key]
            else:
                self.kwargs[key] += kwargs[key]
        
        
from PIL import Image
def save_nparray_as_img(array, save_path):
    img = Image.fromarray(array)
    img.save(save_path)

def turn_name(idx, fmt=5):
    idx = str(idx)
    return "0" * (fmt-len(idx)) + idx

def turn_dataset_image_name(data_dir, fmt=3):
    for l in os.listdir(data_dir):
        nl = turn_name(int(l), fmt)
        os.rename(data_dir + "/" + l, data_dir + '/' + nl)
        path = os.path.join(data_dir, nl)
        for iname in os.listdir(path):
            idx, ext = os.path.splitext(iname)
            niname = turn_name(int(idx), fmt) + ext
            os.rename(path + '/' + iname, path + '/' + niname)
            
def turn_dataset_2_jpeg(data_dir, out_dir):
    mkdir_if_not_exist(out_dir)
    mse, n = 0., 0
    for l in os.listdir(data_dir):
        mkdir_if_not_exist(out_dir + "/" + l)
        path = os.path.join(data_dir, l)
        for iname in os.listdir(path):
            idx, ext = os.path.splitext(iname)
            a = np.load(path + '/' + iname)
            b = np.round(a).astype('uint8')
            mse += np.sum((a-b)**2)
            save_nparray_as_img(b, out_dir + '/' + l + '/' + idx + '.jpg')
            n += 1
    return mse/n

def save_data(data, label, save_prefix, label_idx, fmt=4, ext='.npy'):
    data = data.transpose((0, 2, 3, 1)).asnumpy()
    label = label.reshape((-1, )).asnumpy()
    for img, l in zip(data, label):
        l = int(l)
        np.save(save_prefix + "/" + turn_name(l, fmt) + "/" + turn_name(label_idx[l], fmt), img)
        label_idx[l] += 1

def dataset_adverarial(data_iter, mean, std, output_path, net, ctx, bn_control, generator, fmt=4, **kwargs):
    mkdir_if_not_exist(output_path)
    bn_control.store()
    i, image_count = 0, 0
    logger_sum = LoggerSum(acc=None, mse=None, loss=None, result_score=None, label_score=None)
    label_idx = []
    for data, label in data_iter:
        for l in label.asnumpy().reshape((-1,)):
            mkdir_if_not_exist(os.path.join(output_path, turn_name(int(l), fmt)))
            num_label = (nd.max(label) + 1).astype('int').asscalar()
            if len(label_idx) < num_label:
                label_idx = label_idx + [0] * (num_label - len(label_idx))
        adversarial_data, o = generator(net, data, label, bn_control=None, ctx=ctx, record_detail=True, **kwargs)
        adversarial_data = inv_normalize(adversarial_data.as_in_context(mean.context), mean=mean, std=std, clip=False, asnumpy=False)
        save_data(adversarial_data, label, output_path, label_idx, fmt)
        
        logger = o[0]
        logger_sum(acc=logger.acc, mse=logger.sse, loss=logger.losses, result_score=logger.log_result_score, 
                   label_score=logger.log_label_score)
        image_count += data.shape[0]
        #if i > 3 : break
        i += 1
    for key in logger_sum.kwargs:
        logger_sum.kwargs[key] /= image_count
    logger_sum.kwargs['label_score'] = np.exp(logger_sum.kwargs['label_score'])
    logger_sum.kwargs['result_score'] = np.exp(logger_sum.kwargs['result_score'])
    bn_control.load()
    return logger_sum.kwargs
    