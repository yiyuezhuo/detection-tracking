# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:40:50 2019

@author: yiyuezhuo
"""

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import os


#model=models.resnet18(num_classes=3)
#pretrained_weight = torch.load('weights/resnet18-5c106cde.pth')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser("Train a resnet18 model to classify boat",
                                     usage='python train.py dataset --pretrained weights/resnet18-5c106cde.pth')
    
    parser.add_argument('data_root')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--pretrained', default=None)
    parser.add_argument('--batch_size', default=15, type=int)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--pin_memory', default=True, type=bool)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--device', default='cuda', help='device: cuda or cpu')
    parser.add_argument('--epoch', default=10000, type=int)
    parser.add_argument('--save_epoch', default=10, type=int)
    parser.add_argument('--cache_name', default='resnet18')
    parser.add_argument('--cache_root', default='checkpoints')

    
    args = parser.parse_args()
    
    print(args)
    
    model=models.resnet18(num_classes=3)
    
    assert not (args.resume and args.pretrained)
    
    if not args.resume and not args.pretrained:
        print("Training start from scratch")
    
    if args.resume:
        print('Resume from: {}'.format(args.resume))
        model.load_state_dict(torch.load(args.resume), strict=True)
    
    if args.pretrained:
        print('Use pretrained model: {}'.format(args.pretrained))
        pretrained_weight = torch.load(args.pretrained)
        del pretrained_weight['fc.weight']
        del pretrained_weight['fc.bias']
        missing_keys = model.load_state_dict(pretrained_weight, strict=False)
        print(missing_keys)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    dataset = datasets.ImageFolder(
        args.data_root,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
    ]))
    
    print('dataset size:', len(dataset))
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=args.pin_memory)
    
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    device = torch.device(args.device)
    model.to(device)
    
    model.train()
    
    for epoch_idx in range(args.epoch):
        loss_sum = 0.0
        for it_idx,(image, target) in enumerate(data_loader):
            image = image.to(device)
            target = target.to(device)
            
            output = model(image)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item()
            
            #print('Iteration: {} loss: {}'.format(it_idx, loss.item()))
            
        print('Epoch: {} loss: {}'.format(epoch_idx, loss_sum))
        
        if epoch_idx % args.save_epoch==0:
            os.makedirs(args.cache_root, exist_ok=True)
            cache_name = '{}_{}.pth'.format(args.cache_name, epoch_idx)
            cache_path = os.path.join(args.cache_root, cache_name)
            torch.save(model.state_dict(), cache_path)
            print('Saving {}'.format(cache_path))
            
    