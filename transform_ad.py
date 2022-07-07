import yacs.config
import numpy as np
import PIL.Image
import PIL, PIL.ImageOps
import random
import torchvision
import albumentations #https://pypi.org/project/albumentations/
#https://github.com/imjihee/DeepAA-bulryang/blob/master/augmentation.py

class CenterCrop:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.CenterCrop(
            config.dataset.image_size)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)


class Normalize:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image: PIL.Image.Image) -> np.ndarray:
        image = np.asarray(image).astype(np.float32) / 255.
        image = (image - self.mean) / self.std
        return image


class RandomCrop:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.RandomCrop(
            config.dataset.image_size,
            padding=config.augmentation.random_crop.padding,
            fill=config.augmentation.random_crop.fill,
            padding_mode=config.augmentation.random_crop.padding_mode)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)


class RandomResizeCrop:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.RandomResizedCrop(
            config.dataset.image_size)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)


class RandomHorizontalFlip:
    def __init__(self):
        self.transform = torchvision.transforms.RandomHorizontalFlip(
            0.5)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image: #PIL->PIL
        #print("RHF input:", data)
        temp =self.transform(data)
        print("RHF return:", temp)
        return temp

class Resize:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.Resize(config.tta.resize)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)

"""
Translations (with torchvision.transforms functions)
"""

class TranslateX:
    def __init__(self, p):
        self.v = ((random.random() - 0.5) * 0.8) * 32
        self.p = p
    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        if random.random() < self.p:
            data = data.transform(data.size, PIL.Image.AFFINE, (1, 0, self.v, 0, 1, 0))
        return data

class TranslateY:
    def __init__(self, p):
        self.v = ((random.random() - 0.5) * 0.8) * 32
        self.p = p
    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        if random.random() < self.p:
            data = data.transform(data.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, self.v))
        return data

class Flip:
    def __init__(self, p):
        self.p = p
    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        if random.random() < self.p:
            data = PIL.ImageOps.mirror(data)
        return data

class Rotate:
    def __init__(self, p):
        self.p = p
    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        if random.random() < self.p:
            if random.random() > 0.5:
                data = data.rotate(30)
            else:
                data = data.rotate(-30)
        return data

class Posterize:
    def __init__(self, p):
        self.p = p
    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        if random.random() < self.p:
            if random.random()>0.5:
                data = PIL.ImageOps.posterize(data, 4)
            else:
                data = PIL.ImageOps.posterize(data, 5)
        return data

class AutoContrast:
    def __init__(self, p):
        self.p = p
    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        if random.random() < self.p:
            data = PIL.ImageOps.autocontrast(data)
        return data

"""
class :
    def __init__(self, p):
        self.p = p
    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        if random.random() > self.p:
            data = 
        return data
        
img.save('-.png')
"""