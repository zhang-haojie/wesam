import os
import cv2
import random
import glob
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage.draw import polygon2mask

from datasets.tools import ResizeAndPad, soft_transform, collate_fn, decode_mask, collate_fn_


class PolypDataset(Dataset):
    def __init__(self, cfg, root_dir, annotation_file, transform=None, split=False, training=False, if_self_training=False):
        self.cfg = cfg
        self.root_dir = root_dir
        self.transform = transform
        with open(annotation_file, "r") as ann_file:
            anns = json.load(ann_file)
        all_images = list(anns.keys())

        if split:
            train_images = []
            eval_images = []
            while all_images:
                for _ in range(5):
                    if all_images:
                        train_images.append(all_images.pop(0))
                if all_images:
                    eval_images.append(all_images.pop(0))

            if training:
                random.shuffle(train_images)
                images = train_images
            else:
                random.shuffle(eval_images)
                images = eval_images
        else:
            images = all_images

        self.images = images
        self.anns = anns
        self.if_self_training = if_self_training

    def __len__(self):
        return len(self.images)

    def find_points_outside_bbox(self, mask, bboxes):
        points_outside_bbox = np.where(mask != 0)
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            points_outside_bbox = (points_outside_bbox[0][(points_outside_bbox[0] < y_min) | (points_outside_bbox[0] >= y_max)],
                                points_outside_bbox[1][(points_outside_bbox[1] < x_min) | (points_outside_bbox[1] >= x_max)])
        return points_outside_bbox

    def __getitem__(self, idx):
        name = self.images[idx]
        image_path = os.path.join(self.root_dir, "images", name + ".jpg")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.cfg.get_prompt:
            image_info = {}
            height, width, _ = image.shape
            image_info["file_path"] = image_path
            image_info["height"] = height
            image_info["width"] = width
            return idx, image_info, image

        gt_path = image_path.replace("images", "masks")
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_mask[gt_mask > 0] = 255

        masks = []
        bboxes = []
        categories = []
        anns = self.anns[name]
        ann_bboxes = anns["bbox"]

        for i, bbox in enumerate(ann_bboxes):
            x_min = bbox["xmin"]
            x_max = bbox["xmax"]
            y_min = bbox["ymin"]
            y_max = bbox["ymax"]
            gt_mask[y_min:y_max, x_min:x_max][gt_mask[y_min:y_max, x_min:x_max] > 0] = i + 1
            bboxes.append([x_min, y_min, x_max, y_max])
            categories.append(bbox["label"])

        gt_mask[gt_mask > i + 1] = 0
        gt_masks = decode_mask(torch.tensor(gt_mask[None, :, :])).numpy().astype(np.uint8)
        assert gt_masks.sum() == (gt_mask > 0).sum()
        assert len(ann_bboxes) == gt_masks.shape[0]
        masks = [mask for mask in gt_masks]

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


class PolypDatasetwithCoarse(PolypDataset):

    def __getitem__(self, idx):
        name = self.images[idx]
        image_path = os.path.join(self.root_dir, "images", name + ".jpg")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gt_path = image_path.replace("images", "masks")
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_mask[gt_mask > 0] = 255

        masks = []
        bboxes = []
        categories = []
        anns = self.anns[name]
        ann_bboxes = anns["bbox"]

        for i, bbox in enumerate(ann_bboxes):
            x_min = bbox["xmin"]
            x_max = bbox["xmax"]
            y_min = bbox["ymin"]
            y_max = bbox["ymax"]
            gt_mask[y_min:y_max, x_min:x_max][gt_mask[y_min:y_max, x_min:x_max] > 0] = i + 1
            # bboxes.append([x_min, y_min, x_max, y_max])
            categories.append(bbox["label"])

        gt_mask[gt_mask > i + 1] = 0
        gt_masks = decode_mask(torch.tensor(gt_mask[None, :, :])).numpy().astype(np.uint8)
        assert gt_masks.sum() == (gt_mask > 0).sum()
        assert len(ann_bboxes) == gt_masks.shape[0]

        for mask in gt_masks:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            num_vertices = 0.05 * cv2.arcLength(contours[0], True)
            num_vertices = num_vertices if num_vertices > 3 else 3
            approx = cv2.approxPolyDP(contours[0], num_vertices, True)  # [x, y]
            approx = approx.squeeze(1)

            coordinates = np.array(approx)
            x_max, x_min = max(coordinates[:, 0]), min(coordinates[:, 0])
            y_max, y_min = max(coordinates[:, 1]), min(coordinates[:, 1])
            if x_min == x_max or y_min == y_max:
                x, y, w, h = cv2.boundingRect(mask)
                bboxes.append([x, y, x + w, y + h])
            else:
                bboxes.append([x_min, y_min, x_max, y_max])

            coarse_mask = polygon2mask(mask.shape, coordinates).astype(mask.dtype)

            masks.append(mask)

        masks = [mask for mask in gt_masks]

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
    val = PolypDataset(
        cfg,
        root_dir=cfg.datasets.Polyp.root_dir,
        annotation_file=cfg.datasets.Polyp.annotation_file,
        transform=transform,
        split=cfg.split,
    )
    train = PolypDataset(
        cfg,
        root_dir=cfg.datasets.Polyp.root_dir,
        annotation_file=cfg.datasets.Polyp.annotation_file,
        transform=transform,
        split=cfg.split,
        training=True,
        if_self_training=cfg.augment,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    train_dataloader = DataLoader(
        train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    return train_dataloader, val_dataloader


def load_datasets_coarse(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = PolypDatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.Polyp.root_dir,
        annotation_file=cfg.datasets.Polyp.annotation_file,
        transform=transform,
        split=cfg.split,
    )
    train = PolypDatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.Polyp.root_dir,
        annotation_file=cfg.datasets.Polyp.annotation_file,
        transform=transform,
        split=cfg.split,
        training=True,
        if_self_training=cfg.augment,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    train_dataloader = DataLoader(
        train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    return train_dataloader, val_dataloader


def load_datasets_prompt(cfg, img_size):
    transform = ResizeAndPad(img_size)
    train = PolypDataset(
        cfg,
        root_dir=cfg.datasets.Polyp.root_dir,
        annotation_file=cfg.datasets.Polyp.annotation_file,
        transform=transform,
        split=cfg.split,
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
