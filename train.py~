# -*- coding: UTF-8 -*-
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from utils.dataset import BSDS500
from models.hed import HED
import torch.nn.init as init
import torch.nn.functional as F
import utils.utils as utils


parser = argparse.ArgumentParser(description='train hed model')
parser.add_argument('--batchSize', type=int, default=1, help='with batchSize=1 equivalent to instance normalization.')
parser.add_argument('--niter', type=int, default=15, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
parser.add_argument('--lr_decay', type=float, help='learning rate decay', default=0.1)
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--outf', default='checkpoints/', help='folder to output images and model checkpoints')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

lr_decay_epoch = {3, 5, 8 , 10, 12}
lr_decay = opt.lr_decay
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

##########   DATASET   ###########
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])
target_transform = transforms.Compose([
                transforms.ToTensor()
            ])

###########   DATASET   ###########
data = BSDS500(transform=transform, target_transform=target_transform)
train_loader = torch.utils.data.DataLoader(dataset=data,
                                           batch_size=1,
                                           shuffle=True,
                                           num_workers=2)

torch.cuda.set_device(0)

###########   MODEL   ###########

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    print (classname)
    if classname.find('Conv2d') != -1:
        init.xavier_uniform(m.weight.data)
        init.constant(m.bias.data, 0.1)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

net = HED()
net.cuda()

net.apply(weights_init)

print ('load the weight from vgg')
pretrained_dict = torch.load('vgg16.pth')
model_dict = net.state_dict()
# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
# 3. load the new state dict
net.load_state_dict(model_dict)
print ('copy the weight sucessfully')
print(net)

###########   LOSS & OPTIMIZER   ##########
def bce2d(input, target):
    n, c, h, w = input.size()
    # assert(max(target) == 1)
    log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_trans = target_t.clone()
    pos_index = (target_t >0)
    neg_index = (target_t ==0)
    target_trans[pos_index] = 1
    target_trans[neg_index] = 0
    pos_index = pos_index.data.cpu().numpy().astype(bool)
    neg_index = neg_index.data.cpu().numpy().astype(bool)
    weight = torch.Tensor(log_p.size()).fill_(0)
    weight = weight.numpy()
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num*1.0 / sum_num
    weight[neg_index] = pos_num*1.0 / sum_num
    
    weight = torch.from_numpy(weight)
    weight = weight.cuda()
    loss = F.binary_cross_entropy(log_p, target_t, weight, size_average=True)
    return loss
# criterion = nn.BCELoss()



lr = opt.lr
# optimizer = torch.optim.Adam(net.parameters(),lr=lr, betas=(opt.beta1, 0.999))
# optimizer = torch.optim.SGD(net.parameters(),lr=lr, momentum=opt.momentum, weight_decay=0.0002)
## different learning rate
conv5_params = list(map(id, net.conv5.parameters()))
fuse_params = list(map(id, net.fuse.parameters()))

base_params = filter(lambda p: id(p) not in conv5_params+fuse_params,
                     net.parameters())

optimizer = torch.optim.SGD([
            {'params': base_params},
            {'params': net.conv5.parameters(), 'lr': lr * 100},
            {'params': net.fuse.parameters(), 'lr': lr * 0.001}
            ], lr=lr, momentum=0.9)
###

########### Training   ###########
net.train()
for epoch in range(1,opt.niter+1):
    for i, image in enumerate(train_loader):
        ########### fDx ###########
        net.zero_grad()

        imgA = image[0]
        imgB = image[1]
        img = Variable(imgA).cuda()
        gt = Variable(imgB).cuda()
        output = net(img)

        side_output1 = output[0]
        side_output2 = output[1]
        side_output3 = output[2]
        side_output4 = output[3]
        side_output5 = output[4]
        final_output = output[5]

        loss_side1 = bce2d(side_output1, gt)
        loss_side2 = bce2d(side_output2, gt)
        loss_side3 = bce2d(side_output3, gt)
        loss_side4 = bce2d(side_output4, gt)
        loss_side5 = bce2d(side_output5, gt)
        final_loss = bce2d(final_output, gt)

        loss = (loss_side1 + loss_side2 + loss_side3 + loss_side4 + loss_side5 + final_loss) 
        loss.backward()
        optimizer.step()

        ########### Logging ##########
        if(i % 10 == 0):
            print('[%d/%d][%d/%d] Loss_1: %.4f Loss_2: %.4f Loss_3: %.4f Loss_4: %.4f Loss_5: %.4f Loss_all: %.4f lr= %.8f'
                      % (epoch, opt.niter, i, len(train_loader),
                         loss_side1.data[0], loss_side2.data[0], loss_side3.data[0], loss_side4.data[0], loss_side5.data[0], final_loss.data[0], lr))
        if(i % 100 == 0):
            vutils.save_image(final_output.data,
                       'tmp/samples_i_%d_%03d.png' % (epoch, i),
                       normalize=True)

    if epoch in lr_decay_epoch:
            lr *= lr_decay
            optimizer = torch.optim.SGD([
                {'params': base_params},
                {'params': net.conv5.parameters(), 'lr': lr * 100},
                {'params': net.fuse.parameters(), 'lr': lr * 0.001}
                ], lr=lr, momentum=0.9)
            # optimizer = torch.optim.Adam(net.parameters(),lr=lr, betas=(opt.beta1, 0.999))

    torch.save(net.state_dict(), '%s/hed_%d.pth' % (opt.outf, epoch))
