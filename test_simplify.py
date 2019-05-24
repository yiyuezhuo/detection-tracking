# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:56:22 2019

@author: yiyuezhuo
"""

from __future__ import print_function
#import sys
import os
#import argparse
import torch
#import torch.nn as nn
#import torch.backends.cudnn as cudnn
#import torchvision.transforms as transforms
#from torch.autograd import Variable
#from data import VOC_ROOT, VOC_CLASSES
#from PIL import Image
#from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from data import BaseTransform
#import torch.utils.data as data
from ssd import build_ssd
import cv2

from data import VOC_CLASSES
labelmap = VOC_CLASSES



use_cuda = True

if use_cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')



#base_transform = BaseTransform(net.size, (104, 117, 123))    
    
    
def test_basic(net, img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    transform = BaseTransform(net.size, (104, 117, 123))
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    #x = Variable(x.unsqueeze(0))
    x = x.unsqueeze(0)
    
    if use_cuda:
        x = x.cuda()

    with torch.no_grad():
        y = net(x)
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0],
                         img.shape[1], img.shape[0]])
    pred_num = 0
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            score = detections[0, i, j, 0]
            label_name = labelmap[i-1]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            #coords = (pt[0], pt[1], pt[2], pt[3])
            cv2.rectangle(img, (pt[0], pt[1]), (pt[2], pt[3]), (255, 0, 0), 3)
            text = label_name + ':' + '{:.4f}'.format(score.item())
            img = cv2.putText(img, text, (pt[0], pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              (255, 255, 255), 2)
            pred_num += 1
            j += 1
    #cv2.imwrite('eval/' + img_id + '.jpg', img)
    return img

def load_net(cache_path):
    num_classes = len(VOC_CLASSES) + 1 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(cache_path))
    net.eval()
    return net

def batch_predict(net, dir_path, output_suffix = 'output'):
    output_dir_path = '{}_{}'.format(dir_path, output_suffix)
    os.makedirs(output_dir_path, exist_ok=True)
    
    for fname in os.listdir(dir_path):
        img_path = os.path.join(dir_path, fname)
        target_path = os.path.join(output_dir_path, fname)
        img_output = test_basic(net, img_path)
        cv2.imwrite(target_path, img_output)
        


if __name__ == '__main__':
    net = load_net('weights/ssd300_COCO_6000.pth')
    img = test_basic(net, 'test_data/images1154.jpg')
    cv2.imwrite('test_data_output/images1154.jpg' ,img)
