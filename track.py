# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:38:00 2019

@author: yiyuezhuo
"""

import sys
import os
import argparse
import torch

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from torch.autograd import Variable
#from data import VOC_ROOT, VOC_CLASSES as labelmap
from PIL import Image
import torch.utils.data as data
import cv2

from data import VOCAnnotationTransform, VOCDetection, BaseTransform#, VOC_CLASSES
from data import VOC_CLASSES,VOC_ROOT
from ssd import build_ssd


#VOC_ROOT = '"../dataset/video/"'
#VOC_CLASSES = ('boat',)

