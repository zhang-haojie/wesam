import os
import cv2
import torch
# import kornia as K
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union
from imagecorruptions import corrupt, get_corruption_names

weak_transforms = A.Compose(
    [A.Flip(), A.HorizontalFlip(), A.VerticalFlip()],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
    # keypoint_params=A.KeypointParams(format='xy')
)

spatial_transforms = A.Compose(
    [
        A.Flip(),
        A.HorizontalFlip(), 
        A.VerticalFlip(),
        A.Rotate(limit=45),
        # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45),
        # A.ElasticTransform(),
        # A.Affine(scale=(0.75, 1.25), translate_percent=(0.7, 1.3), rotate=(-30, 30)),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
    # keypoint_params=A.KeypointParams(format='xy')
)

strong_transforms = A.Compose(
    [
        A.Posterize(),
        A.Equalize(),
        A.Sharpen(),
        A.Solarize(),
        A.RandomBrightnessContrast(),
        A.RandomShadow(),
    ]
)


def vflip(image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(image, np.ndarray):  # [H, W, 3] or [H, W]
        return cv2.flip(image, 0)
    elif isinstance(image, torch.Tensor):  # [H, W]
        return torch.flip(image, dims=[0])


def hflip(image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(image, np.ndarray):  # [H, W, 3] or [H, W]
        return cv2.flip(image, 1)
    elif isinstance(image, torch.Tensor):  # [H, W]
        return torch.flip(image, dims=[1])


def rot90(
    image: Union[np.ndarray, torch.Tensor], factor: int = 2
) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(image, np.ndarray):  # [H, W, 3]  or [H, W]
        image = np.rot90(image, k=factor)
        return np.ascontiguousarray(image)
    elif isinstance(image, torch.Tensor):  # [H, W]
        return torch.rot90(image, k=factor, dims=[0, 1])


def corrupt_image(image, filename):
    file_name = os.path.basename(os.path.abspath(filename))
    file_path = os.path.dirname(os.path.abspath(filename))
    for corruption in get_corruption_names():
        corrupted = corrupt(image, severity=5, corruption_name=corruption)
        corrupt_path = file_path.replace(
            "val2017", os.path.join("corruption", corruption)
        )
        if not os.path.exists(corrupt_path):
            os.makedirs(corrupt_path, exist_ok=True)
        cv2.imwrite(os.path.join(corrupt_path, file_name), corrupted)

