import os
import cv2
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from skimage.draw import polygon2mask
from datasets.tools import ResizeAndPad, soft_transform, collate_fn, collate_fn_, decode_mask


class COD10KDataset(Dataset):
    def __init__(self, cfg, image_root, gt_root, transform=None, split=False, training=False, if_self_training=False):
        self.cfg = cfg
        self.root_dir = image_root
        self.transform = transform
        all_images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg')]
        all_images = sorted(all_images)

        if split:
            train_images = []
            eval_images = []
            while all_images:
                for _ in range(4):
                    if all_images:  # 1621
                        train_images.append(all_images.pop(0))
                if all_images:  # 405
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
        self.gts = [os.path.join(gt_root, os.path.basename(image_path.replace(".jpg", ".png"))) for image_path in self.images]
        self.filter_files()

        self.if_self_training = if_self_training

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size and not np.all(np.array(gt) == 0):
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(self.rgb_loader(self.images[idx]))
        gt_mask = np.array(self.binary_loader(self.gts[idx]))

        if self.cfg.get_prompt:
            image_info = {}
            height, width, _ = image.shape
            image_info["file_path"] = self.images[idx]
            image_info["height"] = height
            image_info["width"] = width
            return idx, image_info, image

        bboxes = []
        masks = []
        categories = []

        gt_masks = decode_mask(torch.tensor(gt_mask[None, :, :])).numpy().astype(np.uint8)
        assert gt_masks.sum() == (gt_mask > 0).sum()
        for mask in gt_masks:
            if np.all(mask == 0):
                continue
            masks.append(mask)
            x, y, w, h = cv2.boundingRect(mask)
            bboxes.append([x, y, x + w, y + h])
            categories.append("0")
        
        if self.if_self_training:
            image_weak, bboxes_weak, masks_weak, image_strong = soft_transform(image, bboxes, masks, categories)

            if self.transform:
                image_weak, masks_weak, bboxes_weak = self.transform(image_weak, masks_weak, np.array(bboxes_weak))
                image_strong = self.transform.transform_image(image_strong)

            bboxes_weak = np.stack(bboxes_weak, axis=0)
            masks_weak = np.stack(masks_weak, axis=0)
            return image_weak, image_strong, torch.tensor(bboxes_weak), torch.tensor(masks_weak).float()

        elif self.cfg.visual:
            image_name =  os.path.splitext(os.path.basename(self.images[idx]))[0]
            origin_image = image
            origin_bboxes = bboxes
            origin_masks = masks
            if self.transform:
                padding, image, masks, bboxes = self.transform(image, masks, np.array(bboxes), True)

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            origin_bboxes = np.stack(origin_bboxes, axis=0)
            origin_masks = np.stack(origin_masks, axis=0)
            return image_name, padding, origin_image, origin_bboxes, origin_masks, image, torch.tensor(bboxes), torch.tensor(masks).float()

        else:
            if self.transform:
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            return image, torch.tensor(bboxes), torch.tensor(masks).float()


class COD10KDatasetwithCoarse(COD10KDataset):

    def __getitem__(self, idx):
        image = np.array(self.rgb_loader(self.images[idx]))
        gt_mask = np.array(self.binary_loader(self.gts[idx]))

        bboxes = []
        masks = []
        coarse_masks = []
        categories = []
        approxes = []

        gt_masks = decode_mask(torch.tensor(gt_mask[None, :, :])).numpy().astype(np.uint8)
        assert gt_masks.sum() == (gt_mask > 0).sum()
        for mask in gt_masks:
            if np.all(mask == 0.):
                continue
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
            categories.append("0")

        if self.if_self_training:
            image_weak, bboxes_weak, masks_weak, image_strong = soft_transform(image, bboxes, masks, categories)

            if self.transform:
                image_weak, masks_weak, bboxes_weak = self.transform(image_weak, masks_weak, np.array(bboxes_weak))
                image_strong = self.transform.transform_image(image_strong)

            bboxes_weak = np.stack(bboxes_weak, axis=0)
            masks_weak = np.stack(masks_weak, axis=0)
            return image_weak, image_strong, torch.tensor(bboxes_weak), torch.tensor(masks_weak).float()

        elif self.cfg.visual:
            image_name =  os.path.splitext(os.path.basename(self.images[idx]))[0]
            origin_image = image
            origin_approxes = approxes
            origin_masks = masks
            if self.transform:
                padding, image, masks, bboxes = self.transform(image, masks, np.array(bboxes), self.cfg.visual)

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            origin_masks = np.stack(origin_masks, axis=0)
            return image_name, padding, origin_image, origin_approxes, origin_masks, image, torch.tensor(bboxes), torch.tensor(masks).float()

        else:
            if self.transform:
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            return image, torch.tensor(bboxes), torch.tensor(masks).float()


def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = COD10KDataset(
        cfg,
        image_root=cfg.datasets.COD10K.test,
        gt_root=cfg.datasets.COD10K.GT,
        transform=transform,
        split=cfg.split,
    )
    train = COD10KDataset(
        cfg,
        image_root=cfg.datasets.COD10K.test,
        gt_root=cfg.datasets.COD10K.GT,
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
    val = COD10KDatasetwithCoarse(
        cfg,
        image_root=cfg.datasets.COD10K.test,
        gt_root=cfg.datasets.COD10K.GT,
        transform=transform,
        split=cfg.split,
    )
    train = COD10KDatasetwithCoarse(
        cfg,
        image_root=cfg.datasets.COD10K.test,
        gt_root=cfg.datasets.COD10K.GT,
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


def load_datasets_visual(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = COD10KDataset(
        cfg,
        image_root=cfg.datasets.COD10K.test,
        gt_root=cfg.datasets.COD10K.GT,
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


def load_datasets_visual_coarse(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = COD10KDatasetwithCoarse(
        cfg,
        image_root=cfg.datasets.COD10K.test,
        gt_root=cfg.datasets.COD10K.GT,
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
    train = COD10KDataset(
        cfg,
        image_root=cfg.datasets.COD10K.test,
        gt_root=cfg.datasets.COD10K.GT,
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