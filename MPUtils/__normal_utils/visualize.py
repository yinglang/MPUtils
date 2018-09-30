"""
 data visulize
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import shutil
from .bbox import inv_normalize_box

import cv2 as cv
from math import sqrt, ceil, floor
def merge_images(imgs, MN=None, shape=None, space=1):
    """
        imgs: list of numpy array , can be different size, normalize to [0, 1]
        space: the space of dirrent image
    """
    if len(imgs) == 0: return None
    if len(imgs) == 1: return imgs[0]
    if shape is None:
        s = shape = imgs[0].shape
    if MN is None:
        M = int(floor(sqrt(len(imgs))))
        N = int(ceil(len(imgs) * 1.0 / M))
    else:
        M, N = MN
    ms = (s[0] * M + space * (M-1), s[1] * N + space*(N-1)) + s[2:]
    merged_img = np.ones(shape=ms)
    for m in range(M):
        for n in range(N):
            i = m * N + n
            if i >= len(imgs): break
            ms, ns = m*(s[0]+space), n*(s[1]+space)
            merged_img[ms:ms+s[0], ns:ns+s[1], :] = cv.resize(imgs[i], shape[:2]).reshape((s[0], s[1], -1))
    return merged_img

import numpy as np
import matplotlib.pyplot as plt
def box_to_rect(box, color, linewidth=1):
    """convert an anchor box to a matplotlib rectangle"""
    return plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                  fill=False, edgecolor=color, linewidth=linewidth)

import warnings

def draw_bbox(fig, bboxes, color=(0, 0, 0), linewidth=1, fontsize=5, normalized_label=True, wh=None, show_text=False):
    """
        draw boxes on fig
        
    argumnet:
        bboxes: [[x1, y1, x2, y2, (cid), (score) ...]], 
        normalized_label: if label xmin, xmax, ymin, ymax is normaled to 0~1, set it to True and wh must given, else set to False.
        wh: (image width, height) needed when normalized_label set to True
        show_text: if boxes have cid or (cid, score) dim, can set to True to visualize it.
    """
    if np.max(bboxes) <= 1.:
        if normalized_label==False: warnings.warn("[draw_bbox]:the label boxes' max value less than 1.0, may be it is noramlized box," + 
                      "maybe you need set normalized_label==True and specified wh", UserWarning)
    else:
        if normalized_label==True: warnings.warn("[draw_bbox]:the label boxes' max value bigger than 1.0, may be it isn't noramlized box," + 
                      "maybe you need set normalized_label==False.", UserWarning)
    
    if normalized_label: 
        assert wh != None, "wh must be specified when normalized_label is True. maybe you need setnormalized_label=False "
        bboxes = inv_normalize_box(bboxes, wh[0], wh[1])
    for box in bboxes:
        # [x1, y1, x2, y2, (cid), (score) ...]
        if len(box) >= 5 and box[4] < 0: continue  # have cid or not
        rect = box_to_rect(box[:4], color, linewidth)
        fig.add_patch(rect)
        if show_text:
            text = str(int(box[4]))
            if len(box) >= 6: text += " {.3f}".format(box[5])
            fig.text(box[0], box[1], text , 
                bbox=dict(facecolor=(1, 1, 1), alpha=0.5), fontsize=fontsize, color=(0, 0, 0))   
    
def show_images(images, labels=None, rgb_mean=np.array([0, 0, 0]), std=np.array([1, 1, 1]),
                MN=None, color=(0, 1, 0), linewidth=1, figsize=(8, 4), show_text=False, fontsize=5, 
                xlabels=None, ylabels=None, clip=True, normalized_label=True, bboxes_list=[], bboxes_colors=[]):
    """
    advise to set dpi to 120
        import matplotlib as mpl
        mpl.rcParams['figure.dpi'] = 120
    
    images: numpy array type, shape is (n, 3, h, w), or (n, 2, h, w), pixel value range 0~1, float type
    labels: boxes, shape is (n, m, k), m is number of box, k(k>=5) means every box is [xmin, ymin, xmax, ymax, cid, ...]
    rgb_mean: if images has sub rgb_mean, shuold specified.
    MN: is subplot's row and col, defalut is (-1, 5), -1 mean row is adaptive, and col is 5
    normalized_label: if label xmin, xmax, ymin, ymax is normaled to 0~1, set it to True, else set to False.
    """  
    if MN is None:
        M, N = (images.shape[0] + 4) // 5, 5
    else:
        M, N = MN
    _, figs = plt.subplots(M, N, figsize=figsize)
    
    images = (images.transpose((0, 2, 3, 1)) * std) + rgb_mean
    h, w = images.shape[1], images.shape[2]
    
    wh = (w, h) if normalized_label else None
    for i in range(M):
        for j in range(N):
            if M == 1 and N == 1: fig = figs
            else: fig = figs[i][j] if M > 1 and N > 1 else figs[j]
            if N * i + j < images.shape[0]:
                image = (images[N * i + j])
                if clip:
                    image = image.clip(0, 1)
                fig.imshow(image)
                
                if xlabels is not None: 
                    fig.set_xlabel(xlabels[N * i + j], fontsize=fontsize)
                if ylabels is not None: 
                    fig.set_ylabel(ylabels[N * i + j], fontsize=fontsize)
                
                fig.set_xticks([])
                fig.set_yticks([])
#                 fig.axes.get_xaxis().set_visible(False)
#                 fig.axes.get_yaxis().set_visible(False)
                
                if labels is not None:
                    draw_bbox(fig, labels[N * i + j], color, linewidth, fontsize, normalized_label, wh, show_text)
                for bboxes, box_color in zip(bboxes_list, bboxes_colors):
                    draw_bbox(fig, bboxes, box_color, linewidth, fontsize, normalized_label, wh, show_text)
            else:
                fig.set_visible(False)
    return figs

def show_image(image, label=None, rgb_mean=np.array([0, 0, 0]), std=np.array([1, 1, 1]),
                MN=None, color=(0, 1, 0), linewidth=1, figsize=(8, 4), show_text=False, fontsize=5, 
               xlabels=None, ylabels=None, clip=True, normalized_label=True, bboxes_list=[], bboxes_colors=[]):
    """
    advise to set dpi to 120
        import matplotlib as mpl
        mpl.rcParams['figure.dpi'] = 120
    
        image: numpy array type, shape is (h, w, 3), or (h, w, 2), pixel value range 0~1, float type
        label: boxes, shape is (m, k), m is number of box, k(k>=5) means every box is 
                [xmin, ymin, xmax, ymax, (cid, (score)) ...], '(x)'means x is optinal
        rgb_mean: if images has sub rgb_mean, shuold specified.
        MN: is subplot's row and col, defalut is (-1, 5), -1 mean row is adaptive, and col is 5
        normalized_label: if label xmin, xmax, ymin, ymax is normaled to 0~1, set it to True, else set to False.
    
        bboxes_list: list(boxes), other boxes need to draw
        bboxes_colors: list((r, g, b)), color to bboxes in bboxes_list.
    """
    si = tuple([1] + list(image.shape))
    image = image.reshape(si).transpose((0, 3, 1, 2)).astype(np.float)
    if label is not None: 
        sl = tuple([1] + list(label.shape))
        label = label.reshape(sl).astype(np.float)
    show_images(image, label, rgb_mean, std, MN, color, linewidth, figsize, 
                show_text, fontsize, xlabels, ylabels, clip, normalized_label,
               bboxes_list, bboxes_colors)