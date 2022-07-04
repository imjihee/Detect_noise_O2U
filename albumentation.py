from typing import Tuple, Union

import numpy as np
import PIL.Image
import torch
import albumentations #https://pypi.org/project/albumentations/

class Normalize:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.transform = albumentations.Normalize(
                mean = np.array(mean),
                std = np.array(std)
        )
        # self.mean = np.array(mean)
        # self.std = np.array(std)

    def __call__(self, **data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(image=data['image'])


class Resize:
    def __init__(self, args):
        self.transform = albumentations.Resize(args.resize, args.resize)

    def __call__(self, **data: PIL.Image.Image) -> PIL.Image.Image:
        
        return self.transform(image=data['image'])

class Blur:
    def __init__(self, args):
        self.transform = albumentations.Blur(
            args.blur_limit,
            p= 1)

    def __call__(self, **data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(image=data['image'])

class GridDistortion:
    def __init__(self, args):
        self.transform = albumentations.GridDistortion(
                    num_steps = 15,
                    distort_limit= (-0.8, 0.8),
                    p = 1
        )

    def __call__(self, **data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(image=data['image'])

class ElasticTransform:
    def __init__(self, args):
        self.transform = albumentations.ElasticTransform(alpha=100, sigma=8, p = 1)

    def __call__(self, **data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(image=data['image'])

class ColorJitter:
    def __init__(self, args):
        self.transform = albumentations.ColorJitter(
        p = 1  )

    def __call__(self, **data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(image=data['image'])

class ShiftScaleRotate:
    def __init__(self, args):
        self.transform = albumentations.ShiftScaleRotate(p = 0.5)

    def __call__(self, **data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(image=data['image'])

class Transpose:
    def __init__(self, args):
        self.transform = albumentations.Transpose(p = 1)

    def __call__(self, **data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(image=data['image'])

class RandomRotate90:
    def __init__(self, args):
        self.transform = albumentations.RandomRotate90(p = 1)

    def __call__(self, **data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(image=data['image'])

class Sharpen:
    def __init__(self, args):
        self.transform = albumentations.Sharpen(
                alpha = (1,1), lightness = (0.5, 1.0),p = 1)

    def __call__(self, **data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(image=data['image'])

class MedianBlur:
    def __init__(self, args):
        self.transform = albumentations.MedianBlur(p = 1)

    def __call__(self, **data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(image=data['image'])

class MultiplicativeNoise:
    def __init__(self, args):
        self.transform = albumentations.MultiplicativeNoise(
            args.multiplicative_noise_multiplier,
            args.multiplicative_noise_per_channel,
            args.multiplicative_noise_elementwise,
            p = 1
        )

    def __call__(self, **data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(image=data['image'])

class JpegCompression:
    def __init__(self, args):
        self.transform = albumentations.JpegCompression(
            args.jpegcompression_quality_lower,
            args.jpegcompression_quality_upper,
            p=1
        )

    def __call__(self, **data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(image=data['image'])

class RandomGridShuffle:
    def __init__(self, args):
        self.transform = albumentations.RandomGridShuffle(
            args.randomgridshuffle_grid,
            p = 1
        )

    def __call__(self, **data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(image=data['image'])

class ToTensor:
    def __call__(
        self, **data: Union[np.ndarray, Tuple[np.ndarray, ...]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if isinstance(data['image'], tuple):
            return tuple([self._to_tensor(image) for image in data])
        else:
            return self._to_tensor(data)

    @staticmethod
    def _to_tensor(data: np.ndarray) -> torch.Tensor:
        if len(data.shape) == 3:
            return torch.from_numpy(data.transpose(2, 0, 1).astype(np.float32))
        else:
            return torch.from_numpy(data[None, :, :].astype(np.float32))

class CenterCrop:
    def __init__(self, args):
        self.transform = albumentations.CenterCrop(
            args.center_crop, args.center_crop, 
            p = 1
        )

    def __call__(self, **data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(image=data['image'])
