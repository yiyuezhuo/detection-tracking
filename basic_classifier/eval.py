# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:34:42 2019

@author: yiyuezhuo
"""

import torch
from torchvision import models, transforms, datasets

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser("Evaluate the model",
                                     usage = 'python dataset weights/resnet18_350.pth')
    parser.add_argument('data_root')
    parser.add_argument('model_path')
    parser.add_argument('--batch_size', default=15, type=int)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--pin_memory', default=True, type=bool)
    parser.add_argument('--device', default='cuda', help='device: cuda or cpu')


    args = parser.parse_args()
    print(args)
    
    model=models.resnet18(num_classes=3)
    model.load_state_dict(torch.load(args.model_path))
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    # TODO: replace it with a more realistic one
    dataset = datasets.ImageFolder(
        args.data_root,
        transforms.Compose([ # Is it suitable?
                transforms.Resize(256), 
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
    ]))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=args.pin_memory)
    
    device = torch.device(args.device)
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        
        total_true = 0
        total = 0
        
        for image, target in data_loader:
            image = image.to(device)
            target = target.to(device)
            output = model(image) # batch_size x num_class
            values, indices = torch.max(output, 1)
            
            count_true = torch.sum(indices == target)
            count_total = indices.shape[0]
            
            total_true += count_true
            total += count_total
        
        print('true: {} total: {} acc: {}'.format(total_true, total, float(total_true)/float(total)))
    
