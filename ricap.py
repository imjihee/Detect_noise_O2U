from typing import List, Tuple

import numpy as np
import torch
import yacs.config
import torch
import torch.nn as nn
from time import sleep
from PIL import Image
import pdb

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torchvision

def ricap(
    batch: Tuple[torch.Tensor, torch.Tensor], beta: float
) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[float]]]:
    data, targets = batch
    image_h, image_w = data.shape[2:]
    ratio = np.random.beta(beta, beta, size=2)
    w0, h0 = np.round(np.array([image_w, image_h]) * ratio).astype(np.int)
    w1, h1 = image_w - w0, image_h - h0
    ws = [w0, w1, w0, w1]
    hs = [h0, h0, h1, h1]

    patches = []
    labels = []
    label_weights = []
    for w, h in zip(ws, hs):
        indices = torch.randperm(data.size(0))
        x0 = np.random.randint(0, image_w - w + 1)
        y0 = np.random.randint(0, image_h - h + 1)
        patches.append(data[indices, :, y0:y0 + h, x0:x0 + w])
        labels.append(targets[indices])
        label_weights.append(h * w / (image_h * image_w))

    data = torch.cat(
        [torch.cat(patches[:2], dim=3),
         torch.cat(patches[2:], dim=3)], dim=2)
    targets = (labels, label_weights)

    return data, targets

class RICAPCollactor:
    def __init__(self):
        self.beta = 0.3

    def __call__(
            self, batch: List[Tuple[torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[float]]]:
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = ricap(batch, self.beta)
        return batch

class RICAPloss:
    def __init__(self):
        self.loss_func = nn.CrossEntropyLoss(reduction='mean').cuda()

    def __call__(
            self, predictions: torch.Tensor,
            targets: Tuple[List[torch.Tensor], List[float]]) -> torch.Tensor:
        target_list, weights = targets
        return sum([
            weight * self.loss_func(predictions, targets)
            for targets, weight in zip(target_list, weights)
        ])


"""---------------------------------------------------------------------"""

class ricap_dataset:
    def __init__(self, masked_data, beta_of_ricap = 0.3):
        beta = beta_of_ricap
        images = torch.tensor(np.transpose(np.array(masked_data.train_data), (0,3,1,2)))
        labels = torch.tensor(np.array(masked_data.train_noisy_labels))

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
            #print(np.shape(images), np.shape(images[1])) #(45000, 3, 32, 32)
            cropped_images[k] = images[index][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]

            c_[k] = labels[index]
            W_[k] = (w_[k] * h_[k]) / (I_x * I_y)

        # patch cropped images
        self.patched_images = torch.cat(
            (torch.cat((cropped_images[0], cropped_images[1]), 2),
             torch.cat((cropped_images[2], cropped_images[3]), 2)),
            3)

        self.targets = (c_, W_)
        #print("len targets!", len(self.targets)) #2 (labels and weights)
        #print(self.targets)

        #return patched_images, targets

    def __getitem__(self, index):
        img = self.patched_images[index]
        img = np.array(img)
        
        target = [0]*10
        weight = self.targets[1]
        for k in range(4):
            idx = self.targets[0][k][index]
            target[idx] += weight[k]
        #print(img.shape, target) #torch.Size([3, 32, 32])

        #topil = ToPILImage()
        #test = topil(img)
        #test.save("test.png")
        target = torch.tensor(target)
        print(target)
        return img, target, index
        
    def __len__(self):
        return len(self.targets)

def ricap_criterion(logits, labels):
    loss_func = nn.CrossEntropyLoss(reduction='mean').cuda()
    loss = 0
    print(labels)
    for logit in logits:
        for i in range(10):
            #temp = [0]*10
            #temp[i]=1
            #temp = torch.cuda.FloatTensor(temp)
            
            logit = logit.reshape(1,10).to('cuda')
            target = torch.tensor(i).reshape(1).to('cuda')
            #print("!!!!", logit.size(), temp.size())
            #print("label!!", labels, labels.shape)
            loss += loss_func(logit, target)
    return loss
