# -*- coding: UTF-8 -*-
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from utils.dataset import BSDS500_TEST
from models.hed import HED

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

##########   DATASET   ###########
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])
data = BSDS500_TEST(transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=data,
                                           batch_size=1,
                                           shuffle=True,
                                           num_workers=2)

torch.cuda.set_device(1)

net = HED()
net.cuda()

net.load_state_dict(torch.load('checkpoints/hed_1.pth'))

print(net)

########### testing   ###########
net.eval()
to_pil_image = transforms.ToPILImage()
for i, image in enumerate(test_loader):
    ########### fDx ###########

    imgA = image[0]
    name = image[1]
    img = Variable(imgA, volatile = True).cuda()

    # train with fake
    output = net(img)
    
    side_output1 = output[0].cpu()
    side_output2 = output[1].cpu()
    side_output3 = output[2].cpu()
    side_output4 = output[3].cpu()
    side_output5 = output[4].cpu()
    final_output = output[5].cpu()

    print(i) 
    fn = name[0]

    save_dir = 'test/'
    save_side1 = save_dir + '1/'
    save_side2 = save_dir + '2/'
    save_side3 = save_dir + '3/'
    save_side4 = save_dir + '4/'
    save_side5 = save_dir + '5/'
    save_fuse = save_dir + 'fuse/'

    if not os.path.exists(save_side1):
        os.makedirs(save_side1)
    if not os.path.exists(save_side2):
        os.makedirs(save_side2)
    if not os.path.exists(save_side3):
        os.makedirs(save_side3)
    if not os.path.exists(save_side4):
        os.makedirs(save_side4)
    if not os.path.exists(save_side5):
        os.makedirs(save_side5)
    if not os.path.exists(save_fuse):
        os.makedirs(save_fuse)
    
    side1 = to_pil_image(side_output1.data[0])
    side2 = to_pil_image(side_output2.data[0]) 
    side3 = to_pil_image(side_output3.data[0])
    side4 = to_pil_image(side_output4.data[0])
    side5 = to_pil_image(side_output5.data[0])
    fuse = to_pil_image(final_output.data[0])   
    
    side1.save(save_side1 + fn)
    side2.save(save_side2 + fn)
    side3.save(save_side3 + fn)
    side4.save(save_side4 + fn)
    side5.save(save_side5 + fn)
    fuse.save(save_fuse + fn)

