# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:23:28 2019

@author: yiyuezhuo
"""

import torch
from torchvision import models
from pprint import pprint

model=models.resnet18(num_classes=3)

pretrained_weight = torch.load('weights/resnet18-5c106cde.pth')

model_key_set = set(model.state_dict().keys())
pretrained_weight_key_set = set(pretrained_weight.keys())

print('Missing part:')
pprint(model_key_set - pretrained_weight_key_set)
print('Extra part:')
pprint(pretrained_weight_key_set - model_key_set)

print('Fc shape:',model.state_dict()['fc.weight'].shape,pretrained_weight['fc.weight'].shape)

#model.load_state_dict(pretrained_weight)

print("Try load weight directly...")

try:
    model.load_state_dict(pretrained_weight)
except RuntimeError as e:
    print(e)
    
    
del pretrained_weight['fc.bias']
del pretrained_weight['fc.weight']

print('Try load weight without wrong shape weight...')

try:
    model.load_state_dict(pretrained_weight)
except RuntimeError as e:
    print(e)
    
pretrained_weight = torch.load('weights/resnet18-5c106cde.pth')

print('Try load weight direct but with strict=False...')


try:
    model.load_state_dict(pretrained_weight, strict=False)
except RuntimeError as e:
    print(e)
    
del pretrained_weight['fc.bias']
del pretrained_weight['fc.weight']

print('Try load weight without wrong shape weight but with strict=False')

imcompatible_keys = model.load_state_dict(pretrained_weight, strict=False)
print(imcompatible_keys)