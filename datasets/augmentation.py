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
