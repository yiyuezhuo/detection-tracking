# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:45:40 2019

@author: yiyuezhuo
"""

import os

def rename_dir(dir_path, dir_list = None):
    if dir_list is None:
        dir_list = os.listdir(dir_path)
    len_dir_list = len(dir_list)
    base = str(len(str(len_dir_list)) + 1)
    pattern = 'img%0'+base+'d.jpg'
    for i,fname in enumerate(dir_list):
        path = os.path.join(dir_path, fname)
        target_path = os.path.join(dir_path, pattern % i)
        os.rename(path, target_path)


def generate_video_from_dir(dir_path, frame_rate=24, dir_list = None):
    '''
    Warning: Side effect - change name of images dir_path
    
    dir_list: prevent wrong order of such fname structure: 1.jpg 2.jpg ... 10.jpg
    '''
    
    '''
    dir_list = os.listdir(dir_path)
    len_dir_list = len(dir_list)
    base = str(len(str(len_dir_list)) + 1)
    pattern = 'img%0'+base+'d.jpg'
    for i,fname in enumerate(dir_list):
        path = os.path.join(dir_path, fname)
        target_path = os.path.join(dir_path, pattern % i)
        os.rename(path, target_path)
    '''
    rename_dir(dir_path, dir_list = dir_list)
    
    if dir_list is None:
        dir_list = os.listdir(dir_path)
    len_dir_list = len(dir_list)
    base = str(len(str(len_dir_list)) + 1)
    
    command_frame_rate = ' -framerate '+str(frame_rate)
    #command_pattern = ' -i img%0' + base + 'd.jpg'
    command_pattern = ' -i '+dir_path+'/img%0' + base + 'd.jpg'
    command_output = ' '+dir_path+'_output.mp4'
    command = 'ffmpeg' + command_frame_rate + command_pattern + command_output
    print('command: '+command)
    #os.system(command)
    # For some reason, calling it directly doest't work.
    
#\