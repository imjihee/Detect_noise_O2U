from typing import List, Tuple

import numpy as np
import torch
import yacs.config
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

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