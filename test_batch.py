# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:11:44 2019

@author: yiyuezhuo
"""

import os
import torch
import cv2
#import time
import numpy as np
from scipy.ndimage import convolve

#from torch.utils.data import Dataset, DataLoader


#from data import BaseTransform
#from ssd import build_ssd

'''
from data import VOC_CLASSES
labelmap = VOC_CLASSES

use_cuda = True
batch_size = 30

if use_cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
'''

'''
def load_net(cache_path):
    num_classes = len(VOC_CLASSES) + 1 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(cache_path))
    net.eval()
    return net


class BasicDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.listdir = os.listdir(self.root)
        #self.set_listdir = set(self.listdir)
        self.transform = transform
    def __len__(self):
        return len(self.listdir)
    def __getitem__(self, idx):
        path = os.path.join(self.root, self.listdir[idx])
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        x = torch.from_numpy(self.transform(img)[0]).permute(2, 0, 1)
        scale = torch.Tensor([img.shape[1], img.shape[0],
                     img.shape[1], img.shape[0]])

        return x, scale, self.listdir[idx]
    def diff(self, predict_dir, verbose=False):
        check_set = set([os.path.splitext(fname)[0] for fname in self.listdir])
        exist_set = set([os.path.splitext(fname)[0] for fname in os.listdir(predict_dir)])
        self.listdir = ['{}.jpg'.format(name) for name in check_set - exist_set]
        if verbose:
            print('check_set:{} exist_set:{} resid:{} remove:{}'.format(
                    len(check_set),len(exist_set),len(self.listdir),len(check_set)-len(self.listdir)))
    
def batch_predict(net, dataloader, threshold = 0.6, predict_dir='predict_cache', 
                  verbose=False, force=False):
    os.makedirs(predict_dir, exist_ok=True)
    
    if not force:
        dataloader.dataset.diff(predict_dir, verbose=verbose)
    
    if verbose:
        batch_processed = 0
    
    for x_batch, scale_batch, name_list in dataloader:
        if use_cuda:
            x_batch = x_batch.cuda()
        
        with torch.no_grad():
            y_batch = net(x_batch)
        detections = y_batch.data
        
        for batch_idx in range(y_batch.size(0)):
            scale = scale_batch[batch_idx]
            name = name_list[batch_idx]
            pname, ext = os.path.splitext(name)
            
            for i in range(detections.size(1)):
                j = 0
                det_list = []
                while detections[batch_idx, i, j, 0] >= threshold:
                    score = detections[batch_idx, i, j, 0]
                    label_name = labelmap[i-1]
                    pt = (detections[batch_idx, i, j, 1:]*scale).cpu().numpy()
                    
                    det = {'score': score.cpu(),
                                'label_name': label_name,
                                'pt': pt}
                    det_list.append(det)
                    
                    
                    #coords = (pt[0], pt[1], pt[2], pt[3])

                    j += 1
                    
                torch.save(det_list, '{}/{}.cache'.format(predict_dir, pname))
        if verbose:
            batch_processed += 1
            print('Processing batch {}/{}'.format(batch_processed, len(dataloader)))
            pass
'''

def draw_predict_single(predict_name, predict_dir, img_dir, verbose=False):
    name,ext = os.path.splitext(predict_name)
    img_name = '{}.{}'.format(name, 'jpg')
    
    predict_path = os.path.join(predict_dir, predict_name)
    img_path = os.path.join(img_dir, img_name)
    #target_path = os.path.join(output_dir, img_name)
    
    det_list = torch.load(predict_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    for det in det_list:
        score, label_name, pt = det['score'], det['label_name'], det['pt']
        if verbose:
            print(det)
        
        cv2.rectangle(img, (pt[0], pt[1]), (pt[2], pt[3]), (255, 0, 0), 3)
        text = label_name + ':' + '{:.4f}'.format(score.item())
        img = cv2.putText(img, text, (pt[0], pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          (255, 255, 255), 2)
    return img

def draw_predict(predict_dir, img_dir, output_dir, verbose=False):
    os.makedirs(output_dir, exist_ok=True)
    
    for predict_name in os.listdir(predict_dir):
        img = draw_predict_single(predict_name, predict_dir, img_dir, verbose=verbose)
        
        name,ext = os.path.splitext(predict_name)
        img_name = '{}.{}'.format(name, 'jpg')
        target_path = os.path.join(output_dir, img_name)
        
        cv2.imwrite(target_path, img)
        if verbose:
            print('{}=>{}'.format(name, target_path))
    

def box_distance(box1, box2, key='pt'):
    box1_center = np.array([box1[key][::2].mean(),box1[key][1::2].mean()])
    box2_center = np.array([box2[key][::2].mean(),box2[key][1::2].mean()])
    #print(box1_center)
    return np.linalg.norm(box1_center - box2_center)

def load_predict(cache_root):
    predict_cache = []
    predict_cache_map = {}
    id2key = {}
    key2id = {}
    
    for i,name in enumerate(os.listdir(cache_root)):
        det_list = torch.load('{}/{}'.format(cache_root, name))
        predict_cache.append(det_list)
        id2key[i] = name
        key2id[name] = i
        predict_cache_map[name] = det_list
        
    return {'predict_cache': predict_cache,
            'predict_cache_map':predict_cache_map,
            'id2key': id2key,
            'key2id': key2id}
    
class ChainingFail(Exception):
    pass

class DummyPointer:
    '''
    Helper to prevent render recursive content in Jupyter notebook
    '''
    def __init__(self, value):
        self.value = value
    
def smooth_predict(predict_cache, pixel_threshold = 60, verbose=False,
                    K_size = 60, K_delta_size=40, chain_smoother = None,
                    jump_tol = 0):
    #predict_cache = loaded['predict_cache']
    #predict_cache_map = loaded['predict_cache_map']
    #id2key = loaded['id2key']
    #key2id = loaded['key2id']
    
    chain_list = []
    #loaded['chain_list'] = chain_list
    
    '''
    debug_set = set()
    '''
    
    for i,pred in enumerate(predict_cache):
        for box in pred:
            if 'matched' in box:
                continue
            box['matched'] = True
            box['head'] = True
            box['frame'] = i
            chain = [box]
            
            '''
            if id(box) in debug_set:
                raise Exception
            debug_set.add(id(box))
            '''
            
            box_head = box
            jump_count = 0
            for d,pred_test in enumerate(predict_cache[i+1:]):
                try:
                    if len(pred_test) == 0:
                        #break
                        raise ChainingFail
                    distance_list = []
                    alt_list = []
                    for box_test in pred_test:
                        if 'matched' not in box_test:
                            distance = box_distance(box_head, box_test)
                            if distance > pixel_threshold:
                                continue
                            distance_list.append(distance)
                            alt_list.append(box_test)
                    if len(distance_list) == 0:
                        #break
                        raise ChainingFail
                    idx = np.argmin(distance_list)
                    
                    alt_list[idx]['matched'] = True
                    alt_list[idx]['frame'] = i+d+1
                    box_head = alt_list[idx]
                    
                    chain.append(box_head)
                    
                    jump_count = 0
                except ChainingFail:
                    if jump_count >= jump_tol:
                        jump_count = 0
                        break
                    jump_count += 1
                
                '''
                if id(box_head) in debug_set:
                    raise Exception
                debug_set.add(id(box_head))
                '''
            
            chain_list.append(chain)
            
    if verbose:                    
        print('{} chain detected'.format(len(chain_list)))
    
    K = np.ones(K_size)/K_size
    K_delta = np.ones(K_delta_size)/K_delta_size
    
    for chain in chain_list:
        pt_conv_list = []
        for pt_idx in range(4):
            if chain_smoother is None:
                pt_conv = convolve([box['pt'][pt_idx] for box in chain], K, mode ='nearest')
            else:
                pt_conv = chain_smoother(np.array([box['pt'][pt_idx] for box in chain]))
            pt_conv_list.append(pt_conv)
        for j,smoothed_pt in enumerate(np.array(pt_conv_list).T):
            chain[j]['pt_smoothed'] = smoothed_pt
            if j>0:
                chain[j-1]['pt_smoothed_next'] = smoothed_pt
                chain[j-1]['pt_next'] = chain[j]['pt']
    
    # compute normed delta and smooth it            
    
    for chain in chain_list:
        for box in chain:
            if 'pt_smoothed_next' in box:
                pt_smoothed_next_center = np.array([box['pt_smoothed_next'][::2].mean(), box['pt_smoothed_next'][1::2].mean()])
                pt_smoothed_cemter = np.array([box['pt_smoothed'][::2].mean(), box['pt_smoothed'][1::2].mean()])                
                box['delta'] = pt_smoothed_next_center - pt_smoothed_cemter
    
    for chain in chain_list:
        delta_conv_list = []
        for delta_idx in range(2):
            delta_conv = convolve([box['delta'][delta_idx] for box in chain if 'delta' in box], K_delta, mode ='nearest')
            delta_conv_list.append(delta_conv)
        for i, delta_smoothed in enumerate(np.array(delta_conv_list).T):
            chain[i]['delta_smoothed'] = delta_smoothed
            
    # add back-reference to chain from element(box)
    
    for chain in chain_list:
        for box in chain:
            box['chain'] = DummyPointer(chain)

    return {'predict_cache': predict_cache, 'chain_list':chain_list}

def collect_predict(cache_root_or_loaded, pixel_threshold = 60, verbose=False,
                    K_size = 60, K_delta_size=40, chain_smoother = None,
                    jump_tol = 0):
    if isinstance(cache_root_or_loaded, str):
        loaded = load_predict(cache_root_or_loaded)
    else:
        loaded = cache_root_or_loaded

    smoothed = smooth_predict(loaded['predict_cache'], pixel_threshold = pixel_threshold, verbose=verbose,
                    K_size = K_size, K_delta_size=K_delta_size, chain_smoother = chain_smoother,
                    jump_tol = jump_tol)
    loaded['chain_list'] = smoothed['chain_list']
    
    return loaded

        
def test_arrow(cache_name, img_path, predict_cache_map, cvt=False, 
               delta_smoothed = True, resize_factor=None):
    #cache_name = '{}.cache'.format(name)
    #img_path = 'images/{}.jpg'.format(name)
    
    vis = cv2.imread(img_path)
    if cvt:
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    
    #pt = predict_cache_map[cache_name][0]['pt']
    
    if len(predict_cache_map[cache_name]) == 0:
        return vis
    
    for i in range(len(predict_cache_map[cache_name])):
        pt = predict_cache_map[cache_name][i]['pt_smoothed']
        
        # draw rectangle
        cv2.rectangle(vis, (pt[0], pt[1]), (pt[2], pt[3]), (255, 0, 0), 3)
        
        # draw arrow
        pt_center = np.array([(pt[0]+pt[2])/2, (pt[1]+pt[3])/2])
        if delta_smoothed:
            if 'delta_smoothed' not in predict_cache_map[cache_name][i]:
                continue
            delta = predict_cache_map[cache_name][i]['delta_smoothed']
        else:
            if 'pt_smoothed_next' not in predict_cache_map[cache_name][i]:
                continue
            pt_smoothed_next = predict_cache_map[cache_name][i]['pt_smoothed_next']
            pt_smoothed_next_center = np.array([(pt_smoothed_next[0]+pt_smoothed_next[2])/2, (pt_smoothed_next[1]+pt_smoothed_next[3])/2])
            delta = pt_smoothed_next_center - pt_center
            
        delta /= np.linalg.norm(delta) / 200

        pt_smoothed_next_center_adjusted = pt_center + delta
        cv2.arrowedLine(vis, tuple(pt_center.astype(int)), tuple(pt_smoothed_next_center_adjusted.astype(int)), (0, 0, 255), 10)
    
    if resize_factor is not None:
        vis = cv2.resize(vis, (vis.shape[1]//resize_factor, vis.shape[0]//resize_factor))
    
    return vis

def interpolation_box(box_begin, box_end):
    '''
    The function define `interpolation`, `pt_smoothed`, `delta_smoothed`, `pt_smoothed_next`, `frame`
    but not `chain` etc.
    '''
    new_box_list = []
    for frame in range(box_begin['frame']+1, box_end['frame']):
        p = (frame - box_begin['frame'])/(box_end['frame'] - box_begin['frame'])
        box = {'interpolation': True, 'frame': frame}
        box['pt_smoothed'] =  box_begin['pt_smoothed'] * (1-p) +  box_end['pt_smoothed'] * p
        if 'delta_smoothed' in box_begin and 'delta_smoothed' in box_end:
            box['delta_smoothed'] = box_begin['delta_smoothed'] * (1-p) +  box_end['delta_smoothed'] * p
        new_box_list.append(box)
    box_list = [box_begin] + new_box_list + [box_end]
    for box,box_next in zip(box_list[:-1],box_list[1:]):
        box['pt_smoothed_next'] = box_next['pt_smoothed']
    
    return new_box_list
        
def interpolation_chain(chain):
    new_chain = [chain[0]]
    chain[0]['chain'] = DummyPointer(new_chain)
    for i,(box, box_next) in enumerate(zip(chain[:-1],chain[1:])):
        if box['frame']+1 != box_next['frame']:
            for new_box in interpolation_box(box, box_next):
                new_chain.append(new_box)
                new_box['chain'] = DummyPointer(new_chain)
        new_chain.append(box_next)
        box_next['chain'] = DummyPointer(new_chain)
    return new_chain

def interpolation_chain_list(chain_list):
    new_chain_list = []
    for chain in chain_list:
        new_chain_list.append(interpolation_chain(chain))
    
    return new_chain_list

def interpolation_collected(collected):
    new_chain_list = interpolation_chain_list(collected['chain_list'])
    new_predict_cache = [[] for _ in range(len(collected['predict_cache']))]
    for chain in new_chain_list:
        # Note that every boxes occur and only occur once in a chain
        for box in chain:
            new_predict_cache[box['frame']].append(box)
    new_predict_cache_map = {}
    for key,_id in collected['key2id'].items():
        new_predict_cache_map[key] = new_predict_cache[_id]
    return {'predict_cache':new_predict_cache,
            'predict_cache_map':new_predict_cache_map,
            'id2key':collected['id2key'].copy(),
            'key2id':collected['key2id'].copy(), 
            'chain_list': new_chain_list}
        
def cache_to_arrowed(cache_root, img_root, target_root, pixel_threshold = 60, verbose=False,
                    K_size = 60, K_delta_size=40, cvt=False, 
               delta_smoothed = True, resize_factor=None, chain_smoother = None,
               jump_tol = 0, interpolation=False):
    
    collected = collect_predict(cache_root, pixel_threshold = pixel_threshold, verbose=verbose,
                    K_size = K_size, K_delta_size=K_delta_size, chain_smoother = chain_smoother,
                    jump_tol = jump_tol)
    if interpolation:
        collected = interpolation_collected(collected)
        
    predict_cache_map = collected['predict_cache_map']
    
    for i,cache_name in enumerate(predict_cache_map):
        name, ext = os.path.splitext(cache_name)
        img_path = '{}/{}.jpg'.format(img_root, name)
        img_processed = test_arrow(cache_name, img_path, predict_cache_map, cvt=cvt, 
               delta_smoothed = delta_smoothed, resize_factor=resize_factor)
        target_path = '{}/{}.jpg'.format(target_root, name)
        cv2.imwrite(target_path, img_processed)
        if verbose:
            if i % 100 == 0:
                print('write {}/{}'.format(i,len(predict_cache_map)))
                
def adaptive_chain_smoother(arr, F=3):
    n_K = max(len(arr) //F, 1)
    K = np.ones(n_K)/n_K
    return convolve(arr, K, mode ='nearest')


if __name__ == '__main__':

    '''
    dataset_video = BasicDataset('video_data', base_transform)
    dataloader_video2 = DataLoader(dataset_video, batch_size=30, shuffle = False, num_workers=2)
    
    batch_predict(net, dataloader_images, verbose=True)
    draw_predict('predict_cache', 'test_data', 'test_data_output2', verbose=True)
    
    cache_to_arrowed('output074_cache', 'output074_frames', 'output074_processed_int', verbose=True, resize_factor=4, chain_smoother=adaptive_chain_smoother,jump_tol=10,interpolation=True)
    
    smoother = lambda arr: adaptive_chain_smoother(arr, F=6)
    cache_to_arrowed('output074_cache', 'output074_frames', 'output074_processed_int', 
                     verbose=True, resize_factor=4, chain_smoother=smoother,
                     jump_tol=10,interpolation=True)
    '''