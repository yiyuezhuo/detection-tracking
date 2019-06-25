# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:07:01 2019

@author: yiyuezhuo
"""

from PIL import Image

write_map = {}

FONT_HERSHEY_SIMPLEX = 0

class TraceMat:
    def __init__(self):
        self.path = None
        self.command_list = []
        self.shape = None    

def imread(path):
    mat = TraceMat()
    mat.path = path
    im = Image.open(path) # Image will not read entire image into memory
    mat.shape = im.size # (width, height)
    return mat

def cvtColor(*args, **kwargs):
    raise NotImplementedError
    
def rectangle(mat, pt1, pt2, color, thickness):
    command = ('rectangle', pt1, pt2, color, thickness)
    mat.command_list.append(command)
    return mat
    
def arrowedLine(mat, pt1, pt2, color, thickness):
    command = ('arrowedLine', pt1, pt2, color, thickness)
    mat.command_list.append(command)
    return mat
    
def resize(mat, size):
    mat.command_list.append(('resize', size))
    mat.shape = size
    return mat
    
def putText(mat, text, org, fontFace, fontScale, color, thickness, lineType):
    command = ('putText', text, org, fontFace, fontScale, color, thickness, lineType)
    mat.command_list.append(command)
    return mat

def imwrite(path, mat):
    write_map[path] = mat
    return mat

