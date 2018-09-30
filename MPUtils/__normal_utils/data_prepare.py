import os

"""
    generate lst file from dataset, for LstDataset
"""
def write_line(anno_path, img_path, name):
    return img_path + " " + anno_path + '\n'

def generate_list(dataset_root, lst_file, filter=None, write_line=write_line, image_suffixes=['.jpg'], 
                  sub_dir={'annotations':'annotations', 'images':'images'}):
    """
        traversal all annotation file in (dataset_root + "/" + sub_dir['annotations']),
        filte them use given (filter), 
        write the result of (write_line) to (lst_file) line by line.
        
        param:
            dataset_root: str, dataset root dir path, such as "caltech/train".
            lst_file: str, generated lst file path, such as "caltech/lst/train.lst"
            filter: function, return True will keep, False will ignore.
                    def filter(anno_path, image_path):
                        # anno_path: abs path of annotation file
                        # image_path: abs path of image file
                        # they are a path pair, such as filter('/home/user/xx.txt', '/home/user/xx.jpg')
                        return len(open(anno_path).readlines()) > 0 # not empty filter
            write_line: function, returned result will be writed to lst file as a line.
                    def write_line(anno_path, img_path, name):
                        # anno_path, img_path same as filter argument
                        # name is no suffix image name and annotation name
                        # such as write_line('/home/user/xx.txt', '/home/user/xx.jpg', 'xx')
            image_suffixes: list(str), image suffix must be one of them.
            sub_dir: dict('str':'str'), default is {'annotations':'annotations', 'images':'images'}
                    such as data dir structure
                    ----dataset_root
                    --------labels (all annotatons in here) 
                    --------imageset (all images here)
                    then sub_dir={'annotations':'labels', 'images':'imageset'}
    """
    def get_image_path_name(anno_name, image_dir, image_suffix):
        idx = anno_name.rfind('.')
        if idx == -1: idx = len(anno_name) # if hvae no '.', mean it have no suffix.
        name = anno_name[:idx]
        for suffix in image_suffixes:
            imgpath = os.path.join(image_dir, name + suffix)
            if os.path.exists(imgpath):
                return os.path.abspath(imgpath), name
        raise ValueError('image name (' + image_dir + '/' + name + ') with suffix in' 
                         + str(image_suffixes) + 'not exists, but annotation file exsit,' +
                         'please check image_suffixes or image_name')
        
    image_dir = dataset_root + '/' + sub_dir['images'] + "/"
    anno_dir = dataset_root + '/' + sub_dir['annotations'] + '/'
    
    list_f = open(lst_file, 'w')
    for txt in sorted(os.listdir(anno_dir)):
        anno_path = os.path.abspath(os.path.join(anno_dir, txt))
        img_path, name = get_image_path_name(txt, image_dir, image_suffixes)
        if filter is None or filter(anno_path, img_path):  # passed filter
            list_f.write(write_line(anno_path, img_path, name))  
    list_f.close()


class LabelParser(object):
    """
    base class for Label parse, for LstDataset
    """
    @property
    def classes(self):
        """
            return class_name list, such as ['person', 'people', 'person?', 'person-fa']
        """
        raise NotImplementedError()
    def __call__(self, label_strs, image, image_label_path_pair, *args, **kwargs):
        """
            from annotation file content 'label_strs' and image to get label
            return list(list) or np.array, cannot be nd.array
        """
        raise NotImplementedError()


import numpy as np
def valid_box_filter(box):
    if box[0] >= box[2] or box[1] >= box[3]:
        return False
    return True

class CaltechLabelParser(LabelParser):
    def __init__(self, filter=valid_box_filter):
        self.label_dict={'person': 0, 'people': 1, 'person?': 2, 'person-fa': 3}
        self.class_names = [''] * len(self.label_dict)
        for key in self.label_dict:
            self.class_names[self.label_dict[key]] = key
        self.filter = filter
            
    @property
    def classes(self):
        return self.class_names
    
    def __call__(self, lines, image, path_pair, *args, **kwargs):
        """
        input line format
            % Each object struct has the following fields:
            %  lbl  - a string label describing object type (eg: 'pedestrian')
            %  bb   - [l t w h]: bb indicating predicted object extent
            %  occ  - 0/1 value indicating if bb is occluded
            %  bbv  - [l t w h]: bb indicating visible region (may be [0 0 0 0])
            %  ign  - 0/1 value indicating bb was marked as ignore
            %  ang  - [0-360] orientation of bb in degrees
        output line format:
            [xmin, ymin, xmax, ymax, cid, difficult]
        """
        labels = []
        for line in lines:
            if line.strip()[0] == '%': continue
            lbl, l, t, w, h, occ, vl, vt, vw, vh, ign, ang = line.split(' ')
            cid = self.label_dict[lbl]
            xmin, ymin = float(l), float(t)
            xmax, ymax = xmin + float(w), ymin + float(h)
            if self.filter([xmin, ymin, xmax, ymax]):
                difficult = 0
                labels.append([xmin, ymin, xmax, ymax, cid, difficult])
        return np.array(labels)
