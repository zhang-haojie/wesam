import os
import cv2
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from skimage.draw import polygon2mask
from datasets.tools import ResizeAndPad, soft_transform, collate_fn, collate_fn_


class SADataset(Dataset):
    def __init__(self, cfg, root_dir, transform=None, training=False):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = []
        self.sa_list = [
            sa for sa in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, sa))
        ]

        for sa in self.sa_list:
            sa_dir = os.path.join(root_dir, sa)
            self.image_list.extend(
                [
                    os.path.join(sa_dir, f)
                    for f in os.listdir(sa_dir)
                    if f.endswith(".jpg")
                ]
            )

        self.image_list = random.sample(self.image_list, 200)
        self.if_self_training = training

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.cfg.get_prompt:
            image_info = {}
            height, width, _ = image.shape
            image_info["file_path"] = image_path
            image_info["height"] = height
            image_info["width"] = width
            return idx, image_info, image

        json_path = image_path.replace(".jpg", ".json")
        with open(json_path, "r") as f:
            annotations = json.load(f)

        bboxes = []
        masks = []
        categories = []
        for anno in annotations["annotations"]:
            x, y, w, h = anno["bbox"]
            bboxes.append([x, y, x + w, y + h])
            mask = mask_utils.decode(anno["segmentation"])
            masks.append(mask)
            categories.append("0")

        if self.if_self_training:
            image_weak, bboxes_weak, masks_weak, image_strong = soft_transform(image, bboxes, masks, categories)

            if self.transform:
                image_weak, masks_weak, bboxes_weak = self.transform(image_weak, masks_weak, np.array(bboxes_weak))
                image_strong = self.transform.transform_image(image_strong)

            bboxes_weak = np.stack(bboxes_weak, axis=0)
            masks_weak = np.stack(masks_weak, axis=0)
            return image_weak, image_strong, torch.tensor(bboxes_weak), torch.tensor(masks_weak).float()
        else:
            if self.transform:
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            return image, torch.tensor(bboxes), torch.tensor(masks).float()


class SADatasetwithCoarse(SADataset):
    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        json_path = image_path.replace(".jpg", ".json")
        with open(json_path, "r") as f:
            annotations = json.load(f)

        bboxes = []
        masks = []
        categories = []
        for anno in annotations["annotations"]:
            mask = mask_utils.decode(anno["segmentation"])

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            num_vertices = 0.05 * cv2.arcLength(contours[0], True)
            num_vertices = num_vertices if num_vertices > 5 else 5
            approx = cv2.approxPolyDP(contours[0], num_vertices, True)  # [x, y]
            approx = approx.squeeze(1)
            coordinates = np.array(approx)
            x_max, x_min = max(coordinates[:, 0]), min(coordinates[:, 0])
            y_max, y_min = max(coordinates[:, 1]), min(coordinates[:, 1])
            coarse_mask = polygon2mask(mask.shape, coordinates).astype(mask.dtype) 
            if x_min == x_max or y_min == y_max:
                x, y, w, h = cv2.boundingRect(mask)
                bboxes.append([x, y, x + w, y + h])
            else:
                bboxes.append([x_min, y_min, x_max, y_max])

            masks.append(mask)
            categories.append("0")

        if self.if_self_training:
            image_weak, bboxes_weak, masks_weak, image_strong = soft_transform(image, bboxes, masks, categories)

            if self.transform:
                image_weak, masks_weak, bboxes_weak = self.transform(image_weak, masks_weak, np.array(bboxes_weak))
                image_strong = self.transform.transform_image(image_strong)

            bboxes_weak = np.stack(bboxes_weak, axis=0)
            masks_weak = np.stack(masks_weak, axis=0)
            return image_weak, image_strong, torch.tensor(bboxes_weak), torch.tensor(masks_weak).float()
        else:
            if self.transform:
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            return image, torch.tensor(bboxes), torch.tensor(masks).float()


def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    train = SADataset(
        cfg,
        root_dir=cfg.datasets.sa.root_dir,
        transform=transform,
        training=True,
        if_self_training=cfg.augment,
    )
    val = SADataset(
        cfg,
        root_dir=cfg.datasets.sa.root_dir,
        transform=transform,
    )
    train_dataloader = DataLoader(
        train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    return train_dataloader, val_dataloader


def load_datasets_coarse(cfg, img_size):
    transform = ResizeAndPad(img_size)
    train = SADatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.sa.root_dir,
        transform=transform,
        training=True,
        if_self_training=cfg.augment,
    )
    val = SADatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.sa.root_dir,
        transform=transform,
    )
    train_dataloader = DataLoader(
        train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    return train_dataloader, val_dataloader


def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    train = SADataset(
        cfg,
        root_dir=cfg.datasets.sa.root_dir,
        transform=transform,
        training=True,
        if_self_training=cfg.augment,
    )
    train_dataloader = DataLoader(
        train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_,
    )
    return train_dataloader
