import os  
from PIL import Image  
import imageio
import numpy as np
import cv2 as cv

def create_number_image(num, size=(200, 200)):
    image = np.zeros(shape=(200, 200, 3))
    text = str(num)
    text = "0" * (3-len(text)) + text
    x, y = 10, 120
    font_size = 3
    color=(0, 1, 0)
    thickness = 8
    cv.putText(image, text, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
    image = cv.resize(image, size)  
    return image
  
def analyseImage(path):  
    ''''' 
    Pre-process pass over the image to determine the mode (full or additive). 
    Necessary as assessing single frames isn't reliable. Need to know the mode  
    before processing all frames. 
    '''  
    im = Image.open(path)  
    results = {  
        'size': im.size,  
        'mode': 'full',  
    }  
    try:  
        while True:  
            if im.tile:  
                tile = im.tile[0]  
                update_region = tile[1]  
                update_region_dimensions = update_region[2:]  
                if update_region_dimensions != im.size:  
                    results['mode'] = 'partial'  
                    break  
            im.seek(im.tell() + 1)  
    except EOFError:  
        pass  
    return results  
  
def saveGiFToImage(path):  
    ''''' 
    Iterate the GIF, extracting each frame. 
    '''  
    mode = analyseImage(path)['mode']  
      
    im = Image.open(path)  
  
    i = 0  
    p = im.getpalette()  
    last_frame = im.convert('RGBA')  
      
    try:  
        while True:  
            #print "saving %s (%s) frame %d, %s %s" % (path, mode, i, im.size, im.tile)  
              
            ''''' 
            If the GIF uses local colour tables, each frame will have its own palette. 
            If not, we need to apply the global palette to the new frame. 
            '''  
            if not im.getpalette():  
                im.putpalette(p)  
              
            new_frame = Image.new('RGBA', im.size)  
              
            ''''' 
            Is this file a "partial"-mode GIF where frames update a region of a different size to the entire image? 
            If so, we need to construct the new frame by pasting it on top of the preceding frames. 
            '''  
            if mode == 'partial':  
                new_frame.paste(last_frame)  
              
            new_frame.paste(im, (0,0), im.convert('RGBA'))  
            #new_frame.save('%s-%d.png' % (''.join(os.path.basename(path).split('.')[:-1]), i), 'PNG')  
  
            i += 1  
            last_frame = new_frame  
            im.seek(im.tell() + 1)  
    except EOFError:  
        pass  

def get_images_from_gif(gif_path, mode='RGBA'):
    gmode = analyseImage(gif_path)['mode']  
    im = Image.open(gif_path)  
    last_frame = im.convert(mode)
    frames = [np.array(last_frame)]
    p = im.getpalette()  
    try:  
        while True:  
            if not im.getpalette():  
                im.putpalette(p)    
            new_frame = Image.new(mode, im.size)  
            if gmode == 'partial':  
                new_frame.paste(last_frame)  
            new_frame.paste(im, (0,0), im.convert(mode))    
            last_frame = new_frame  
            im.seek(im.tell() + 1)  
            frames.append(np.array(new_frame))
    except EOFError:  
        pass  
    return frames

def write_gif(filename, images, end_image_path=None, duration=0.5, **kwargs):
    if end_image_path is not None:
        end_img = Image.open(end_image_path)
        w, h = images[-1].shape[1], images[-1].shape[0]
        end_img = np.array(end_img.resize((h, w)))
        images = images + [end_img]
    imageio.mimsave(filename, images, 'GIF', duration=duration, **kwargs)
    
def add_end_img_to_gif(gifpath, end_image_path):
    imgs = get_images_from_gif(gifpath)
    write_gif(gifpath, imgs, end_image_path)
