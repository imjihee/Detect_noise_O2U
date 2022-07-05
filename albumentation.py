from typing import Tuple, Union
import yacs.config
import numpy as np
import PIL.Image
import torch
import torchvision
import albumentations #https://pypi.org/project/albumentations/

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
albumentations
"""
class ShiftScaleRotate:
    def __init__(self):
        self.transform = albumentations.ShiftScaleRotate(p = 0.5)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        #print("SSR input:", data)
        temp =self.transform(image = data)
        #self.transform(data) 
        print("SSR return:", temp)
        return temp['image']


class RandomRotate90:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = albumentations.RandomRotate90(p = 0.2)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        temp =self.transform(image = data)
        return temp['image']


class RandomGridShuffle:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = albumentations.RandomGridShuffle(
            (5, 5),
            p = 0.2
        )

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        temp =self.transform(image = data)
        return temp['image']


class Transpose:
    def __init__(self):
        self.transform = albumentations.Transpose(p = 0.2)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        temp =self.transform(image = data)
        return temp['image']


class ColorJitter:
    def __init__(self):
        self.transform = albumentations.ColorJitter(
        p = 0.2  )

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        temp =self.transform(image = data)
        return temp['image']


class Sharpen:
    def __init__(self):
        self.transform = albumentations.Sharpen(
                alpha = (1,1), lightness = (0.5, 1.0),p = 0.5)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        temp =self.transform(image = data)
        return temp['image']

class VerticalFlip:
    def __init__(self):
        self.transform = albumentations.VerticalFlip(p = 0.2)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        temp =self.transform(image = data)
        return temp['image']

class ToSepia:
    def __init__(self):
        self.transform = albumentations.ToSepia(p = 0.2)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        temp =self.transform(image = data)
        return temp['image']

class ChannelShuffle:
    def __init__(self):
        self.transform = albumentations.ChannelShuffle(p = 0.2)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        temp =self.transform(image = data)
        return temp['image']

class ToTensor:
    def __call__(
        self, data: Union[np.ndarray, Tuple[np.ndarray, ...]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if isinstance(data, tuple):
            return tuple([self._to_tensor(image) for image in data])
        else:
            return self._to_tensor(data)

    @staticmethod
    def _to_tensor(data: np.ndarray) -> torch.Tensor:
        if len(data.shape) == 3:
            return torch.from_numpy(data.transpose(2, 0, 1).astype(np.float32))
        else:
            return torch.from_numpy(data[None, :, :].astype(np.float32))
