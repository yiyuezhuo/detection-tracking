# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 22:21:58 2019

@author: yiyuezhuo
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 21:57:16 2019

@author: yiyuezhuo
"""

import torch
from torchvision import models, transforms, datasets
from torch.utils.data.dataset import Dataset
#from torch.utils.data import Dataloader

#import imageio
from PIL import Image
import os
import numpy as np

model_path = 'weights/resnet18_350.pth'

batch_size = 15
workers = 0
pin_memory = True
_device = 'cuda'

model=models.resnet18(num_classes=3)
model.load_state_dict(torch.load(model_path))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

base_transform = transforms.Compose([ # Is it suitable?
                transforms.Resize(256), 
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
])

device = torch.device(_device)

model.to(device)
model.eval()


class CollectedDataset(Dataset):
    def __init__(self, collected, img_root):
        predict_cache_map = collected['predict_cache_map']
        
        self.box_list = []
        self.img_path_list = []
        
        for i,cache_name in enumerate(predict_cache_map):
            name, ext = os.path.splitext(cache_name)
            img_path = '{}/{}.jpg'.format(img_root, name)
            
            for box in predict_cache_map[cache_name]:
                if 'pt' in box:
                    self.box_list.append(box)
                    self.img_path_list.append(img_path)
                    box['predict'] = None
                else:
                    box['predict'] = None # maybe require extra label
                
        self.cache_name = None
        self.cache_img = None
    
    def __len__(self):
        return len(self.box_list)
    
    def __getitem__(self, index):
        box = self.box_list[index]
        img_path = self.img_path_list[index]
        
        if img_path != self.cache_name:
            #self.cache_img = imageio.imread(img_path)
            self.cache_img = Image.open(img_path)
            self.cache_name = img_path
        
        pt = box['pt']
        #crop = self.cache_img[int(pt[1]):int(pt[3]), int(pt[0]):int(pt[2])]
        crop = self.cache_img.crop(pt)
        
        return base_transform(crop), index
    
    def set_box(self, index, predict):
        self.box_list[index]['predict'] = int(predict)
    
def classify_crop(collected, img_root, disp_it=100):
    dataset = CollectedDataset(collected, img_root)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        num_workers=workers, pin_memory=pin_memory)
    
    total_iteration = len(dataset) // batch_size
    
    with torch.no_grad():
                
        for it_idx, (image, idxs) in enumerate(data_loader):
            image = image.to(device)
            output = model(image) # batch_size x num_class
            values, indices = torch.max(output, 1)
            
            for idx, predict in zip(idxs, indices):
                dataset.set_box(idx, predict)
                
            if it_idx % disp_it ==0:
                print('Iteration {}/{}'.format(it_idx, total_iteration))
                
    return collected # It's not required since we have changed their state, the bare reference is not very interesing

def vote_crop(collected):
    chain_list = collected['chain_list']
    for chain in chain_list:
        vote = [0,0,0]
        for box in chain:
            if box['predict'] is not None:
                vote[box['predict']] += 1
        idx = np.argmax(vote)
        for box in chain:
            box['predict'] = idx
    return collected

def classify_transform(collected, img_root, disp_it=100):
    collected = classify_crop(collected, img_root, disp_it=disp_it)
    collected = vote_crop(collected)
    return collected

if __name__ == '__main__':
    '''
    DEBUG code
    '''
    import matplotlib.pyplot as plt
    from test_batch import adaptive_chain_smoother,collect_predict,interpolation_collected,choose_chain_collected,supplement_delta_smoothed,chain_list_desc
    
    smoother = lambda arr: adaptive_chain_smoother(arr, F=6)
    
    cache_root = 'hiv00801_cache'
    img_root = r'E:\ship_detect_demo\hiv00801_frames'
    
    pixel_threshold = 60
    verbose=False
    K_size = 60
    K_delta_size=40
    cvt=False
    delta_smoothed = True
    resize_factor=None
    chain_smoother = None
    jump_tol = 0
    interpolation=False
    chain_length_threshold = 0
    legend=False
    
    verbose=True
    chain_smoother=smoother
    jump_tol=120
    pixel_threshold = 100
    chain_length_threshold=200
    interpolation=True
    legend=True
    
    collected = collect_predict(cache_root, pixel_threshold = pixel_threshold, verbose=verbose,
                    K_size = K_size, K_delta_size=K_delta_size, chain_smoother = chain_smoother,
                    jump_tol = jump_tol)
    if interpolation:
        collected = interpolation_collected(collected)
        
    collected = choose_chain_collected(collected, chain_length_threshold)
    
    for chain in collected['chain_list']:
        supplement_delta_smoothed(chain)
    
    chain_list_desc(collected['chain_list'])
    
    dataset = CollectedDataset(collected, img_root)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        num_workers=workers, pin_memory=pin_memory)
    
    for i in range(5):
        img_tensor, idx = dataset[i]
        assert i==idx
        img = img_tensor.permute(1,2,0)
        img_n = img*torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        
        plt.imshow(img_n)
        plt.show()
        
    print('dataset size:', len(dataset))
    
    classify_crop(collected, img_root)
    
    for i in range(190,210):
        img_tensor, idx = dataset[i]
        assert i==idx
        img = img_tensor.permute(1,2,0)
        img_n = img*torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        
        plt.imshow(img_n)
        plt.show()
        print(dataset.box_list[i]['predict'])
        
    collected = vote_crop(collected)
    
    for chain in collected['chain_list']:
        predict = chain[0]['predict']
        for box in chain:
            if box['predict'] != predict:
                raise Exception