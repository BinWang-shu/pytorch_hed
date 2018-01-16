import numpy as np
from scipy.misc import imread, imresize, imsave
import torch
import copy
from our_vgg import vgg


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def deprocess(img):
    img = img.add_(1).div_(2)

    return img

def transfer_weights(model_from, model_to):
    wf = copy.deepcopy(model_from.state_dict())
    wt = model_to.state_dict()
    for k in wt.keys():
        if not k in wf:
            wf[k] = wt[k]
    model_to.load_state_dict(wf)

def convert_vgg(vgg16):
	net = vgg()

	vgg_items = net.state_dict().items()
	vgg16_items = vgg16.items()

	pretrain_model = {}
	j = 0
	for k, v in net.state_dict().iteritems():
	    v = vgg16_items[j][1]
	    k = vgg_items[j][0]
	    pretrain_model[k] = v
	    j += 1
	return pretrain_model
