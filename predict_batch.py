# -*- coding: utf-8 -*-
"""
Created on Sat May 25 14:42:56 2019

@author: yiyuezhuo
"""

import os
import torch
import cv2
#import time
#import numpy as np
#from scipy.ndimage import convolve

from torch.utils.data import Dataset, DataLoader


from data import BaseTransform
from ssd import build_ssd

from data import VOC_CLASSES
labelmap = VOC_CLASSES

use_cuda = True
batch_size = 30

if use_cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    
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
                    '''
                    cv2.rectangle(img, (pt[0], pt[1]), (pt[2], pt[3]), (255, 0, 0), 3)
                    text = label_name + ':' + '{:.4f}'.format(score.item())
                    img = cv2.putText(img, text, (pt[0], pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                      (255, 255, 255), 2)
                    pred_num += 1
                    '''
                    j += 1
                    
                torch.save(det_list, '{}/{}.cache'.format(predict_dir, pname))
        if verbose:
            batch_processed += 1
            print('Processing batch {}/{}'.format(batch_processed, len(dataloader)))
            pass


if __name__ == '__main__':
    net = load_net('weights/ssd300_COCO_6000.pth')
    base_transform = BaseTransform(net.size, (104, 117, 123))
    
    dataset = BasicDataset('test_data', base_transform)
    dataloader = DataLoader(dataset, batch_size, shuffle = False)
    
    dataset_video = BasicDataset('video_data', base_transform)
    dataloader_video = DataLoader(dataset_video, batch_size, shuffle = False)
    
    dataset_images = BasicDataset('images', base_transform)
    dataloader_images = DataLoader(dataset_images, batch_size, shuffle = False)

    
    #dataset_video = BasicDataset('video_data', base_transform)
    #dataloader_video2 = DataLoader(dataset_video, batch_size=30, shuffle = False, num_workers=2)
    
    #batch_predict(net, dataloader_images, verbose=True)
    #draw_predict('predict_cache', 'test_data', 'test_data_output2', verbose=True)
    
    #cache_to_arrowed('output074_cache', 'output074_frames', 'output074_processed_int', verbose=True, resize_factor=4, chain_smoother=adaptive_chain_smoother,jump_tol=10,interpolation=True)
    
    '''
    dataset_images = BasicDataset('output087_frames', base_transform)
    dataloader_images = DataLoader(dataset_images, batch_size, shuffle = False)
    batch_predict(net, dataloader_images, verbose=True, predict_dir='output087_cache')
    '''
    '''
    For 216 images:
        batch_size = 30 -> 52.1s 
        batch_size = 5 -> 56.5s 
        batch_size = 1 -> 56s 
        batch_size = 30 num_workers=2 -> \infty (Is there a BUG??? https://github.com/pytorch/pytorch/issues/12831)
    
    it seems like the batch_size is useless
    '''
    
    '''
    t1 = time.time()
    batch_predict(net, dataloader_video2)
    t2 = time.time()
    print(t2-t1)
    '''    