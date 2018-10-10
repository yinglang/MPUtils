import shutil
import os
from ..__normal_utils.bbox import xyxy2xywh
from ..__normal_utils.general_tools import mkdir_if_not_exist
import numpy as np
import warnings

def is_empty_dir(path):
    return (not os.path.exists(path)) or len(os.listdir(path)) == 0

def write_result(cids, scores, bboxes, ids, root_dir, person_label, mode='xyxy'):
    """
    write result to file as caltech toolbox format

        cids: np.array, will reshape to ((-1,))
        scores: np.array, will reshape to ((-1,))
        bboes: np.array, will reshape to ((-1, 4)), [[x1, y1, x2, y2]]
        ids: np.array, will reshape to ((-1,)
        mode: 'xyxy' means bboxes is [[xmin, ymin, xmax, ymax]], 'xywh' means bboxes is [[xmin, ymin, w, h]]
    """
    cids, scores, ids = cids.reshape((-1,)), scores.reshape((-1,)), ids.reshape((-1,))
    bboxes = bboxes.reshape((-1, 4))
    if mode.lower() == 'xyxy': bboxes = xyxy2xywh(bboxes)
    setid, vid, iid = ids.astype(np.int)
    setdir = os.path.join(root_dir, 'set' + str(int(setid)).zfill(2))
    mkdir_if_not_exist(setdir)
    vfile = open(os.path.join(setdir, 'V' + str(int(vid)).zfill(3) + '.txt'), 'a')
    iid = str(iid+1)
    for cid, score, bbox in zip(cids, scores, bboxes):
        if cid != person_label: continue
        bbox = [str(b) for b in bbox]
        vfile.write(','.join([iid] + bbox + [str(score)]) + "\n")
    vfile.close()
    
def run_matlab_eval_code(matlab_code='../../dataset/CaltechPedestrians/code3.2.1/MydbEval.m', verbose=False, is_test=True, input_result_dir='.'):
    """
        matlab_code: the matlab code wait to run, eval file generate by write_result and write result to '${matlab_code_dir}/temp_results'
        input_result_dir: the dir of caltech result generate by 'write_result' func
        is_test: use test set gt, if False will use train set gt.
    """
    def split_path(matlab_code):
        idx = matlab_code.rfind('/')
        if idx == -1:
            idx = matlab_code.rfind('\\')
        if idx == -1:
            matlab_code_dir, matlab_code_script = '.', matlab_code[idx+1:]
        else:
            matlab_code_dir, matlab_code_script = matlab_code[:idx], matlab_code[idx+1:]
        return matlab_code_dir, matlab_code_script
    
    matlab_code_dir, matlab_code_script = split_path(matlab_code)
    if input_result_dir[-1] == '/' or input_result_dir[-1] == '\\':
        input_result_dir = input_result_dir[:-1]
    det_dir, algo_name = split_path(input_result_dir)
    #print(matlab_code_dir, matlab_code_script)
    dataName = 'UsaTest' if is_test else 'UsaTrain'
    cmd = ('cd {} && matlab -nodesktop -nodisplay -nojvm -nosplash -r "{} \'{}\' \'{}\' \'{}\';quit"'.format(
        matlab_code_dir, matlab_code_script[:-2], dataName, os.path.abspath(det_dir)+'/', algo_name))
    if verbose: print(cmd)
    os.system(cmd)

    l = len(dataName)
    result = {'Pr':{}, 'Roc': {}}
    for txt in os.listdir(matlab_code_dir + '/temp_results'): # UsaTestPrAll.txt
        if txt[-4:] != '.txt': continue
        f = open('{}/temp_results/{}'.format(matlab_code_dir, txt))
        v = float(f.readline().split()[1])
        if txt[l:l+2] == 'Pr':
            result['Pr'][txt[l+2:-4]] = v
        elif txt[l:l+3] == 'Roc':
            result['Roc'][txt[l+3:-4]] = v
        f.close()
    return result
        
def validate_matlab(net, val_data2, ctx, out_dir, person_label, clear_out_dir=False, verbose=False, is_test=True):
    """
        net: net, output (cids, scores, bboxes)
        val_data2: dataloader, iter output (data, label, Iids(setid, vid, iid))
        out_dir: temp output dir
        clear_out_dir: if not clear, new result will append to old result, get error result. 
                    when out_dir is not empty must set to True.
        is_test: use test set gt, if False will use train set gt.
        
    """
    if not is_empty_dir(out_dir):
        if not clear_out_dir:
            warnings.warn('out_dir {} is not empty, it will use cahce last time generate to eval maybe need set clear_out_dir=True to clear it.'.format(out_dir), UserWarning)
            #raise ValueError('out_dir {} is not empty, maybe need set clear_out_dir=True to clear it.'.format(out_dir))
        else: shutil.rmtree(out_dir)   # clear out_dir
    
    if clear_out_dir or is_empty_dir(out_dir):
        net.collect_params().reset_ctx(ctx)
        net.set_nms(nms_thresh=0.45, nms_topk=400)
        for i, (x, y, ids) in enumerate(val_data2):
            x, y = x.as_in_context(ctx), y.as_in_context(ctx)
            det_bboxes, det_ids, det_scores = [], [], []
            gt_bboxes, gt_ids, gt_difficults = [], [], []
        
            # get prediction results
            cid, scores, bboxes = net(x)
            cid, scores, bboxes, ids = cid.asnumpy(), scores.asnumpy(), bboxes.asnumpy(), ids.asnumpy()
            #print(cid.shape, scores.shape, bboxes.shape, ids.shape)
            for icid, iscores, ibboxes, iids in zip(cid, scores, bboxes, ids):
                write_result(icid, iscores, ibboxes, iids, out_dir, person_label)
            
    result = run_matlab_eval_code(verbose=verbose, is_test=is_test, input_result_dir=out_dir)
    return result
