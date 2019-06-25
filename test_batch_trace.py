# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:01:45 2019

@author: yiyuezhuo

In this module, we will create a fake cv2 module to
change the behaviour of cv2 related function and draw_chinese
"""

import cv2_trace
import test_batch

import json
import os
import numpy as np

def draw_chinese(mat, desc, org_desc, fill):
    command = ('draw_chinese', desc, org_desc, fill)
    mat.command_list.append(command)
    return mat

test_batch.cv2 = cv2_trace
test_batch.draw_chinese = draw_chinese

from test_batch import adaptive_chain_smoother, cache_to_arrowed

print('Verify trace:', adaptive_chain_smoother.__globals__['cv2'],
      adaptive_chain_smoother.__globals__['draw_chinese'])

def clean(obj):
    '''
    Convert strange numpy.float32 value to python float so that we can use JSON
    ''' 
    if isinstance(obj, dict):
        cleaned_obj = {}
        for key,value in obj.items():
            cleaned_obj[key] = clean(value)
        return cleaned_obj
    elif isinstance(obj, (tuple, list)):
        cleaned_obj = []
        for value in obj:
            cleaned_obj.append(clean(value))
        if isinstance(obj, tuple):
            cleaned_obj = tuple(cleaned_obj)
        return cleaned_obj
    elif isinstance(obj, (int, float, str)):
        return obj
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int16, np.int32, np.int64)):
        return int(obj)
    else:
        raise NotImplementedError("Unknown type {}".format(obj))


def transform_command_list(command_list):
    pass

def flush_trace(trace_name = 'trace.json', rewrite=False, verbose=True):
    if not rewrite and os.path.exists(trace_name):
        with open(trace_name, 'r', encoding='utf8') as f:
            obj_map = json.load(f)
        if verbose:
            print('read {} object size: {}'.format(trace_name, len(obj_map)) )

    else:
        obj_map = {}
        
    for key,mat in cv2_trace.write_map.items():
        obj = dict(path = mat.path, 
                   command_list = mat.command_list,
                   shape = mat.shape)
        obj_map[key] = clean(obj)
    
    with open(trace_name, 'w', encoding='utf8') as f:
        json.dump(obj_map, f)
        if verbose:
            print('write {} object size: {}'.format(trace_name, len(obj_map)) )

direc_map = {'无':'none','左':'left','右':'right'}

def transform_trace(write_map):
    '''
    Here we will drop path(to read original image) and keep shape once,
    and transform command_list to a more compact format.
    
    trace = transform_trace(cv2_trace.write_map)
    '''
    root_map = {}
    for write_name, write_value in write_map.items():
        root_name, frame_name = os.path.split(write_name)
        if root_name not in root_map:
            root_map[root_name] = {'shape': write_value.shape, 
                                'frames':{}}
            
        command_list = []
        legend_list = []
        for command in write_value.command_list:
            if command[0] in {'rectangle','arrowedLine','putText'}:
                command_list.append(clean(command))
            elif command[0] == 'draw_chinese':
                _,desc, org_desc, fill = command
                desc_list = desc.split(' ')
                idx = int(desc_list[0].replace('船只','')[:-1])
                coord = eval(desc_list[1].split(':')[1])
                width = eval(desc_list[2].split(':')[1])
                height = eval(desc_list[3].split(':')[1])
                direc = direc_map[desc_list[4].split(':')[1][1:-1]]
                speed = desc_list[5].split(':')[1]
                legend_list.append([idx, coord, width, height, direc, speed])
            else:
                raise Exception("Unknown command {}".format(command[0]))
        frame = {'command_list': command_list,
                 'legend_list': legend_list,
                 'frame_name': frame_name}
        root_map[root_name]['frames'][frame_name] = frame
    
    for root_name, root_value in root_map.items():
        frame_list = []
        length = len(root_value['frames'])
        for i in range(length):
            frame_name = 'thumb{0:05}.jpg'.format(i+1)
            frame_list.append(root_value['frames'][frame_name])
        root_value['frames'] = frame_list
    
    return root_map
        

if __name__ == '__main__':
    '''
    smoother = lambda arr: adaptive_chain_smoother(arr, F=6)
    cache_to_arrowed('hiv00801_cache', r'E:\ship_detect_demo\hiv00801_frames', 'hiv00801_processed_int', 
                     verbose=True, chain_smoother=smoother,
                     jump_tol=120,pixel_threshold = 100,chain_length_threshold=200,
                     interpolation=True, legend=True)
    
    cache_to_arrowed('hiv00803_cache', r'E:\ship_detect_demo\hiv00803_frames', 'hiv00803_processed_int', 
                     verbose=True, chain_smoother=smoother,
                     jump_tol=60,chain_length_threshold=200,
                     interpolation=True, legend=True)
    
    trace = transform_trace(cv2_trace.write_map)
    
    for key, value in trace.items():
        target_path = '{}.json'.format(key)
        with open(target_path, 'w') as f:
            json.dump(value, f)
            print('write: {}'.format(target_path))
    
    #flush_trace()
    '''
