import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.rcParams['figure.dpi'] = 120
#plt.figure(figsize=(12, 4))

"""
most use print_log, parse_log, show_logs_multi_key, update
"""

def print_log(log_file, log_period=1, ignore_sharp=True):
    lines = open(log_file).readlines()
    i = 0
    for line in lines:
        if ignore_sharp and len(line.strip()) > 0 and line.strip()[0] == "#":
            continue
        if i % log_period == 0:
            print(line, end=' ')
        i += 1
    print(lines[-1], end=' ')

def find_tuple(line, end=None):
    idx1 = line.find("(")
    idx2 = line[idx1+1:].find(")") + idx1+1
    w_strs = line[idx1+1:idx2].split(",")
    if end is not None:
        w_strs = w_strs[:end]
    w = []
    for i in range(len(w_strs)):
        w_strs[i] = w_strs[i].strip()
        if len(w_strs[i])<=0: continue
        w.append(float(w_strs[i]))
    return w, idx1, idx2
    
def parse_sharp(line):
    line  = line[1:]
    w, _, idx2 = find_tuple(line)
    g, _, _ = find_tuple(line[idx2+1:], -1)
    return w, g
    

def parse_log(log_file, begin_line=0, to_float_k=["train_acc", "valid_acc", "loss", "epoch"]):
    obj = {'weight':[], 'grad':[]}
    with open(log_file) as f:
        lines = f.readlines()[begin_line:]
        for line in lines:
            line = line.strip()
            if line[0:1] == "#": 
                if line[2:13] == 'epoch_round': pass
                elif line[:6] == "#     ":
                    k, v = line[6:].split(":")
                    k = k.strip()
                    if ("b"+k) in obj: obj["b"+k].append(find_tuple(v)[0])
                    else: obj["b"+k] = [find_tuple(v)[0]]
                else:
                    w, g = parse_sharp(line)
                    #print 'w, g', w, g
                    if len(w) > 0 and len(g) > 0:
                        obj['weight'].append(w)
                        obj['grad'].append(g)
                continue
            vs = []
            for d in line.split(","):
                d = d.strip()
                i = d.rfind(' ')
                k = d[:i].strip()
                v = d[i+1:].strip()
                obj[k] = obj.get(k, [])
                obj[k].append(v)
        if len(lines) > 0:
            for k in to_float_k:
                if k in obj:
                    obj[k] = to_float(obj[k])
    obj['weight'] = obj['weight'][1:]
    obj['grad'] = obj['grad'][1:]
    return obj

def to_float(str_list):
    fa = []
    for s in str_list:
        fa.append(float(s))
    return fa

def plot(data, key, x_range=None):
    x_range = (0, len(data[key])) if x_range is None else x_range
    if x_range[1] == -1:  x_range = (x_range[0], len(data[key]))
    plt.plot(range(*x_range), data[key][x_range[0]:x_range[1]], label=key)
    plt.legend(loc="upper left")
    plt.xlabel("epoch")
    plt.ylabel(key)
    
dataset = {}
data_init = {"train_acc": []}

def update(log_file, x_range=None):
    dataset[log_file] = dataset.get(log_file, {"train_acc": []})
    data = dataset[log_file]
    
    def get_begin_line(log_file, data):
        epochs = len(data["train_acc"])
        if epochs == 0: return 0, 0
        l = 0
        with open(log_file) as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip()
                if line[0:1] != "#":
                    l += 1
                if l >= epochs:
                    return i+1, l
    
    # update date
    begin_line, _  = get_begin_line(log_file, data)
    _data = parse_log(log_file, begin_line=begin_line)
    for k in _data:
        data[k] = data.get(k, [])
        data[k].extend(_data[k])
    if not ("valid_acc" in data) or not ("train_acc" in data) or not ("loss" in data):
        return
    
    # plot
    plt.figure(figsize=(12, 4))   # (w, h)
    plt.subplot(1, 2, 1)
    plot(data, 'train_acc', x_range)
    plot(data, 'valid_acc', x_range)
    #plt.show()
    plt.subplot(1, 2, 2)
    plot(data, 'loss', x_range)
    plt.show()
    #print "lr", sorted(to_float(set(data['lr'])), reverse=True)
    
    # weight and grad
    data = dataset[log_file]["weight"]
    if len(data) == 0: return
    data = np.array(data)
    grad = dataset[log_file]["grad"]
    if len(grad) == 0: return
    grad = np.array(grad)
    print(np.mean(data, axis=0), np.var(data, axis=0))
    print(np.mean(grad, axis=0), np.var(grad, axis=0))

def parse_logs(logs, begin_line=0, to_float_k=None):
    if to_float_k is None: to_float_k=["train_acc", "valid_acc", "loss", "epoch", "valid_loss"]
    dataset = {}
    for log_file in logs:
        dataset[log_file] = parse_log(log_file, begin_line, to_float_k)
    return dataset

def show_log(log, key, as_epoch=True, x_range=(0, -1), idx=None, datasets=None, name=None, begin_line=0, to_float_k=None):
    if datasets is None: datasets = parse_logs([log], begin_line, to_float_k)
    data = datasets[log][key]
    if idx is not None:
        data = np.array(data)[:, idx]
    name = log if name is None else name
    plt.plot(datasets[log]['epoch'][x_range[0]:x_range[1]], data[x_range[0]:x_range[1]], label=name)
    
def show_logs(logs, key, as_epoch=True, x_range=(0, -1), idx=None, datasets=None, names=None, begin_line=0, to_float_k=None):
    if datasets is None: datasets = parse_logs(logs, begin_line, to_float_k)
    for log in logs:
        if key in datasets[log]:
            name = None if names is None else names[log]
            show_log(log, key, as_epoch, x_range, idx, datasets, name, begin_line, to_float_k)
    plt.legend()
    
def show_weight_logs(logs, key, idx, datasets=None, names=None, begin_line=0, to_float_k=None):
    if datasets is None: datasets = parse_logs(logs, begin_line, to_float_k)
    idx = 0
    for log in logs:
        w = np.array(datasets[log][key])
        name = log if names is None else names[log]
        plt.plot(range(len(w)), w[:, idx], label=name)
    plt.legend()
    
def show_logs_multi_key(logs, keys, as_epochs=True, x_ranges=(0, -1), idxs=None, MN=None, datasets=None, names=None, begin_line=0, to_float_k=None, **kwargs):
    mpl.rcParams['figure.dpi'] = 120
    #plt.figure(figsize=(12, 4))
    
    if datasets is None: datasets = parse_logs(logs)
    
    length = len(keys)
    as_epochs = as_epochs if isinstance(as_epochs, list) else [as_epochs] * length
    x_ranges = x_ranges if isinstance(x_ranges, list) else [x_ranges] * length
    idxs = idxs if isinstance(idxs, list) else [idxs] * length
    if MN == None:
        MN = (1, length)
    MN = str(MN[0]) + str(MN[1])
    for i, (key, as_epoch, x_range, idx) in enumerate(zip(keys, as_epochs, x_ranges, idxs)):
        plt.subplot(int(MN + str(i+1)))
        plt.title(key)
        sub_key = key.split('_')
        if sub_key[0] in ['weight', 'grad']:
            show_weight_logs(logs, sub_key[0], int(sub_key[1]), datasets, names)
        show_logs(logs, key, as_epoch, x_range, idx, datasets, names, begin_line, to_float_k, **kwargs)
    return datasets
