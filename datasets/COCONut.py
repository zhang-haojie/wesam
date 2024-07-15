import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pycocotools.coco import COCO
from skimage.draw import polygon2mask
from datasets.tools import ResizeAndPad, soft_transform, collate_fn, collate_fn_


class COCONutDataset(Dataset):
    def __init__(self, cfg, root_dir, annotation_file, transform=None, split=False, training=False, if_self_training=False):
        self.cfg = cfg
        self.root_dir = root_dir
        self.transform = transform
        self.coconut = COCO(annotation_file)
        all_image_ids = sorted(list(self.coconut.imgs.keys()))

        if split:
            train_image_ids = []
            eval_image_ids = []
            while all_image_ids:
                for _ in range(6):
                    if all_image_ids:
                        train_image_ids.append(all_image_ids.pop(0))
                if all_image_ids:
                    eval_image_ids.append(all_image_ids.pop(0))

            if training:
                random.shuffle(train_image_ids)
                image_ids = train_image_ids
            else:
                random.shuffle(eval_image_ids)
                image_ids = eval_image_ids
        else:
            image_ids = all_image_ids

        # Filter out image_ids without any annotations
        self.image_ids = [
            image_id
            for image_id in image_ids
            if len(self.coconut.getAnnIds(imgIds=image_id)) > 0
        ]

        self.if_self_training = if_self_training

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coconut.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info["file_name"])
        if self.cfg.corrupt in self.cfg.corruptions:
            image_path = image_path.replace("val2017", os.path.join("corruption", self.cfg.corrupt))
        image = cv2.imread(image_path)
        # corrupt_image(image, image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.cfg.get_prompt:
            image_info["file_path"] = image_path
            return image_id, image_info, image

        ann_ids = self.coconut.getAnnIds(imgIds=image_id)
        anns = self.coconut.loadAnns(ann_ids)
        bboxes = []
        masks = []
        categories = []
        for ann in anns:
            if len(masks) > 150:
                break
            x, y, w, h = ann["bbox"]
            mask = self.coconut.annToMask(ann)
            if w == 0 or h ==0 and np.count_nonzero(mask) < 100:
                continue
            masks.append(mask)
            bboxes.append([x, y, x + w, y + h])
            categories.append(ann["category_id"])

        if self.if_self_training:
            image_weak, bboxes_weak, masks_weak, image_strong = soft_transform(image, bboxes, masks, categories)
            # image_origin = image_weak

            # image_weak = cv2.cvtColor(image_weak, cv2.COLOR_RGB2BGR)
            # image_strong = cv2.cvtColor(image_strong, cv2.COLOR_RGB2BGR)
            # cv2.imwrite('image_weak.jpg', image_weak)
            # cv2.imwrite('image_strong.jpg', image_strong)

            if self.transform:
                image_weak, masks_weak, bboxes_weak = self.transform(image_weak, masks_weak, np.array(bboxes_weak))
                image_strong = self.transform.transform_image(image_strong)

            bboxes_weak = np.stack(bboxes_weak, axis=0)
            masks_weak = np.stack(masks_weak, axis=0)
            return image_weak, image_strong, torch.tensor(bboxes_weak), torch.tensor(masks_weak).float()

        elif self.cfg.visual:
            origin_image = image
            origin_bboxes = bboxes
            origin_masks = masks
            if self.transform:
                padding, image, masks, bboxes = self.transform(image, masks, np.array(bboxes), True)

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            origin_bboxes = np.stack(origin_bboxes, axis=0)
            origin_masks = np.stack(origin_masks, axis=0)
            return image_id, padding, origin_image, origin_bboxes, origin_masks, image, torch.tensor(bboxes), torch.tensor(masks).float()

        else:
            if self.transform:
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            return image, torch.tensor(bboxes), torch.tensor(masks).float()


class COCONutDatasetwithCoarse(COCONutDataset):

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coconut.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info["file_name"])
        if self.cfg.corrupt in self.cfg.corruptions:
            image_path = image_path.replace("val2017", os.path.join("corruption", self.cfg.corrupt))
        image = cv2.imread(image_path)
        # corrupt_image(image, image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coconut.getAnnIds(imgIds=image_id)
        anns = self.coconut.loadAnns(ann_ids)

        bboxes = []
        masks = []
        coarse_masks = []
        categories = []
        approxes = []

        for ann in anns:
            mask = self.coconut.annToMask(ann)

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            num_vertices = 0.05 * cv2.arcLength(contours[0], True)
            num_vertices = num_vertices if num_vertices > 3 else 3
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
            coarse_masks.append(coarse_mask)
            approxes.append(approx)
            categories.append(ann["category_id"])

        if self.if_self_training:
            image_weak, bboxes_weak, masks_weak, image_strong = soft_transform(image, bboxes, masks, categories)

            if self.transform:
                image_weak, masks_weak, bboxes_weak = self.transform(image_weak, masks_weak, np.array(bboxes_weak))
                image_strong = self.transform.transform_image(image_strong)

            bboxes_weak = np.stack(bboxes_weak, axis=0)
            masks_weak = np.stack(masks_weak, axis=0)
            return image_weak, image_strong, torch.tensor(bboxes_weak), torch.tensor(masks_weak).float()

        elif self.cfg.visual:
            origin_image = image
            origin_approxes = approxes
            origin_masks = masks
            if self.transform:
                padding, image, masks, bboxes = self.transform(image, masks, np.array(bboxes), self.cfg.visual)

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            origin_masks = np.stack(origin_masks, axis=0)
            return image_id, padding, origin_image, origin_approxes, origin_masks, image, torch.tensor(bboxes), torch.tensor(masks).float()

        else:
            if self.transform:
                _, coarse_masks, _ = self.transform(image, coarse_masks, np.array(bboxes))
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            coarse_masks = np.stack(coarse_masks, axis=0)
            return image, torch.tensor(bboxes), torch.tensor(masks).float()


def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    train = COCONutDataset(
        cfg,
        root_dir=cfg.datasets.coconut.root_dir,
        annotation_file=cfg.datasets.coconut.annotation_file,
        transform=transform,
        split=cfg.split,
        training=True,
        if_self_training=cfg.augment,
    )
    val = COCONutDataset(
        cfg,
        root_dir=cfg.datasets.coconut.root_dir,
        annotation_file=cfg.datasets.coconut.annotation_file,
        transform=transform,
        split=cfg.split,
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
        batch_size=cfg.val_batchsize,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    return train_dataloader, val_dataloader


def load_datasets_coarse(cfg, img_size):
    transform = ResizeAndPad(img_size)
    train = COCONutDatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.coconut.root_dir,
        annotation_file=cfg.datasets.coconut.annotation_file,
        transform=transform,
        split=cfg.split,
        training=True,
        if_self_training=cfg.augment,
    )
    val = COCONutDatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.coconut.root_dir,
        annotation_file=cfg.datasets.coconut.annotation_file,
        transform=transform,
        split=cfg.split,
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
        batch_size=cfg.val_batchsize,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    return train_dataloader, val_dataloader


def load_datasets_visual(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = COCONutDataset(
        cfg,
        root_dir=cfg.datasets.coconut.root_dir,
        annotation_file=cfg.datasets.coconut.annotation_file,
        transform=transform,
        split=cfg.split,
    )
    subset = Subset(val, indices=range(0, 100))
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_,
    )
    return val_dataloader


def load_datasets_visual_coarse(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = COCONutDatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.coconut.root_dir,
        annotation_file=cfg.datasets.coconut.annotation_file,
        transform=transform,
        split=cfg.split,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_,
    )
    return val_dataloader


def load_datasets_prompt(cfg, img_size):
    transform = ResizeAndPad(img_size)
    train = COCONutDataset(
        cfg,
        root_dir=cfg.datasets.coconut.root_dir,
        annotation_file=cfg.datasets.coconut.annotation_file,
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