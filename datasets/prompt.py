import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, ConcatDataset
from pycocotools.coco import COCO
from skimage.draw import polygon2mask
from datasets.tools import ResizeAndPad, soft_transform, soft_transform_all, collate_fn, collate_fn_soft, collate_fn_
from prompt import Prompt
from datasets.coco import COCODataset, COCODatasetwithCoarse


class promptCOCODataset(Dataset):
    def __init__(self, cfg, root_dir, annotation_file, rate=(6, 1), transform=None, training=False, if_self_training=False):
        self.cfg = cfg
        self.root_dir = root_dir
        self.transform = transform
        self.prompt = Prompt(annotation_file)
        all_image_ids = sorted(list(self.prompt.imgs.keys()))

        train_image_ids = []
        eval_image_ids = []
        train_rate, eval_rate = rate
        while all_image_ids:
            for _ in range(train_rate):
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

        # Filter out image_ids without any annotations
        self.image_ids = [
            image_id
            for image_id in image_ids
            if len(self.prompt.getAnnIds(imgIds=image_id)) > 0
        ]

        self.if_self_training = if_self_training

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.prompt.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info["file_name"])
        if self.cfg.corrupt in self.cfg.corruptions:
            image_path = image_path.replace("val2017", os.path.join("corruption", self.cfg.corrupt))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.prompt.getAnnIds(imgIds=image_id)
        anns = self.prompt.loadAnns(ann_ids)
        bboxes = []
        masks = []
        categories = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            bboxes.append([x, y, x + w, y + h])
            mask = self.prompt.annToMask(ann)
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


class promptCOCODatasetwithCoarse(promptCOCODataset):
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.prompt.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info["file_name"])
        if self.cfg.corrupt in self.cfg.corruptions:
            image_path = image_path.replace("val2017", os.path.join("corruption", self.cfg.corrupt))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.prompt.getAnnIds(imgIds=image_id)
        anns = self.prompt.loadAnns(ann_ids)

        bboxes = []
        masks = []
        categories = []
        for ann in anns:
            mask = self.prompt.annToMask(ann)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            num_vertices = 0.05 * cv2.arcLength(contours[0], True)
            num_vertices = num_vertices if num_vertices > 3 else 3
            approx = cv2.approxPolyDP(contours[0], num_vertices, True)  # [x, y]
            approx = approx.squeeze(1)
            coordinates = np.array(approx)
            x_max, x_min = max(coordinates[:, 0]), min(coordinates[:, 0])
            y_max, y_min = max(coordinates[:, 1]), min(coordinates[:, 1])
            # coarse_mask = polygon2mask(mask.shape, coordinates).astype(mask.dtype)

            if x_min == x_max or y_min == y_max:
                x, y, w, h = cv2.boundingRect(mask)
                bboxes.append([x, y, x + w, y + h])
            else:
                x, y, w, h = ann["bbox"]
                bboxes.append([x, y, x + w, y + h])

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


class MixDataset(Dataset):
    def __init__(self, cfg, root_dir, coco_annotation_file, prompt_annotation_file, rate=(6, 1), transform=None, training=False, if_self_training=False):
        self.cfg = cfg
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(coco_annotation_file)
        self.prompt = Prompt(prompt_annotation_file)

        all_image_ids = sorted(list(self.coco.imgs.keys()))

        train_image_ids = []
        eval_image_ids = []
        train_rate, eval_rate = rate
        while all_image_ids:
            for _ in range(train_rate):
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

        # Filter out image_ids without any annotations
        self.image_ids = [
            image_id
            for image_id in image_ids
            if len(self.coco.getAnnIds(imgIds=image_id)) > 0
        ]

        self.if_self_training = if_self_training
        self.memory = 0

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if self.memory % 10 <= 7:
            data = self.coco
            self.memory += 1
        else:
            data = self.prompt
            self.memory += 1

        image_id = self.image_ids[idx]
        image_info = data.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info["file_name"])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = data.getAnnIds(imgIds=image_id)
        anns = data.loadAnns(ann_ids)
        bboxes = []
        masks = []
        categories = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            bboxes.append([x, y, x + w, y + h])
            mask = data.annToMask(ann)
            masks.append(mask)
            categories.append("0")

        if self.if_self_training:
            image_weak, bboxes_weak, masks_weak, image_strong = soft_transform(image, bboxes, masks, categories)
            # image_origin = image_weak

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


class MixDatasetwithCoarse(MixDataset):
    def __getitem__(self, idx):
        if self.memory % 10 <= 4:
            data = self.coco
            self.memory += 1
        else:
            data = self.prompt
            self.memory += 1

        image_id = self.image_ids[idx]
        image_info = data.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info["file_name"])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = data.getAnnIds(imgIds=image_id)
        anns = data.loadAnns(ann_ids)
        bboxes = []
        masks = []
        categories = []
        for ann in anns:
            mask = data.annToMask(ann)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            num_vertices = 0.05 * cv2.arcLength(contours[0], True)
            num_vertices = num_vertices if num_vertices > 3 else 3
            approx = cv2.approxPolyDP(contours[0], num_vertices, True)  # [x, y]
            approx = approx.squeeze(1)
            coordinates = np.array(approx)
            x_max, x_min = max(coordinates[:, 0]), min(coordinates[:, 0])
            y_max, y_min = max(coordinates[:, 1]), min(coordinates[:, 1])
            # coarse_mask = polygon2mask(mask.shape, coordinates).astype(mask.dtype)

            if x_min == x_max or y_min == y_max:
                x, y, w, h = cv2.boundingRect(mask)
                bboxes.append([x, y, x + w, y + h])
            else:
                x, y, w, h = ann["bbox"]
                bboxes.append([x, y, x + w, y + h])

            masks.append(mask)
            categories.append("0")

        if self.if_self_training:
            image_weak, bboxes_weak, masks_weak, image_strong = soft_transform(image, bboxes, masks, categories)
            # image_origin = image_weak

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

    
def load_datasets_soft(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = COCODataset(
        cfg,
        root_dir=cfg.datasets.val.root_dir,
        annotation_file=cfg.datasets.val.annotation_file,
        transform=transform,
    )
    soft_train = promptCOCODataset(
        cfg,
        root_dir=cfg.datasets.prompt.root_dir,
        annotation_file=cfg.datasets.prompt.annotation_file,
        transform=transform,
        training=True,
        if_self_training=True,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    soft_train_dataloader = DataLoader(
        soft_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_soft,
    )
    return soft_train_dataloader, val_dataloader


def load_datasets_soft_coarse(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = COCODatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.val.root_dir,
        annotation_file=cfg.datasets.val.annotation_file,
        transform=transform,
    )
    soft_train = promptCOCODatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.prompt.root_dir,
        annotation_file=cfg.datasets.prompt.annotation_file,
        transform=transform,
        training=True,
        if_self_training=True,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    soft_train_dataloader = DataLoader(
        soft_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_soft,
    )
    return soft_train_dataloader, val_dataloader


def load_datasets_mix(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = COCODataset(
        cfg,
        root_dir=cfg.datasets.val.root_dir,
        annotation_file=cfg.datasets.val.annotation_file,
        transform=transform,
    )
    soft_train = MixDataset(
        cfg,
        root_dir=cfg.datasets.val.root_dir,
        coco_annotation_file=cfg.datasets.val.annotation_file,
        prompt_annotation_file=cfg.datasets.prompt.annotation_file,
        rate=(6, 1),
        transform=transform,
        training=True,
        if_self_training=True,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    soft_train_dataloader = DataLoader(
        soft_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_soft,
    )
    return soft_train_dataloader, val_dataloader


def load_datasets_mix_coarse(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = COCODatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.val.root_dir,
        annotation_file=cfg.datasets.val.annotation_file,
        transform=transform,
    )
    soft_train = MixDatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.val.root_dir,
        coco_annotation_file=cfg.datasets.val.annotation_file,
        prompt_annotation_file=cfg.datasets.prompt.annotation_file,
        rate=(6, 1),
        transform=transform,
        training=True,
        if_self_training=True,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    soft_train_dataloader = DataLoader(
        soft_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_soft,
    )
    return soft_train_dataloader, val_dataloader
