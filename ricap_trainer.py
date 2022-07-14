import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import numpy as np

class ricap_dataset:
    def __init__(self, masked_data, beta_of_ricap = 0.3):
        beta = beta_of_ricap
        images = masked_data.train_data
        targets = masked_data.train_noisy_labels

        # size of image
        I_x, I_y = 32, 32

        # generate boundary position (w, h)
        w = int(np.round(I_x * np.random.beta(beta, beta)))
        h = int(np.round(I_y * np.random.beta(beta, beta)))
        w_ = [w, I_x - w, w, I_x - w]
        h_ = [h, h, I_y - h, I_y - h]

        # select four images
        cropped_images = {}
        c_ = {}
        W_ = {}
        for k in range(4):
            index = torch.randperm(len(images))
            x_k = np.random.randint(0, I_x - w_[k] + 1)
            y_k = np.random.randint(0, I_y - h_[k] + 1)
            cropped_images[k] = images[index][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
            c_[k] = targets[index]
            W_[k] = (w_[k] * h_[k]) / (I_x * I_y)

        # patch cropped images
        patched_images = torch.cat(
            (torch.cat((cropped_images[0], cropped_images[1]), 2),
             torch.cat((cropped_images[2], cropped_images[3]), 2)),
            3)

        self.targets = (c_, W_)
        self.patched_images = patched_images

        #return patched_images, targets

    def __getitem__(self, index):
        img, target = self.patched_images[index], self.targets[index]
        return img, target, index