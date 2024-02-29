import os
import cv2
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from skimage.draw import polygon2mask
from datasets.tools import ResizeAndPad, soft_transform, soft_transform_all, collate_fn, collate_fn_soft, collate_fn_, jitter_bbox, encode_mask, decode_mask


class PerSegDataset(Dataset):
    def __init__(self, cfg, image_root, transform=None, training=False, if_self_training=False):
        self.cfg = cfg
        self.root_dir = image_root
        self.transform = transform
        file_paths = []
        for root, directories, files in os.walk(image_root):
            for file in files:
                if file.endswith(".jpg"):
                    file_paths.append(os.path.join(root, file))
        all_images = sorted(file_paths)

        # train_images = []
        # eval_images = []
        # while all_images:
        #     for _ in range(3):
        #         if all_images:
        #             train_images.append(all_images.pop(0))
        #     if all_images:
        #         eval_images.append(all_images.pop(0))

        # if training:
        #     random.shuffle(train_images)
        #     images = train_images
        # else:
        #     random.shuffle(eval_images)
        #     images = eval_images

        self.images = all_images
        self.gts = [image_path.replace("Images", "Annotations").replace(".jpg", ".png") for image_path in self.images]

        self.if_self_training = if_self_training

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gt_path = self.gts[idx]
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        masks = []
        bboxes = []
        categories = []
        gt_masks = decode_mask(torch.tensor(gt_mask[None, :, :])).numpy().astype(np.uint8)
        assert gt_masks.sum() == (gt_mask > 0).sum()

        for mask in gt_masks:
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

        else:
            if self.transform:
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            return image, torch.tensor(bboxes), torch.tensor(masks).float()


class ISICDataset(Dataset):
    def __init__(self, cfg, root_dir, list_file, transform=None, training=False, if_self_training=False):
        self.cfg = cfg
        df = pd.read_csv(os.path.join(list_file), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.root_dir = root_dir
        self.transform = transform

        self.if_self_training = if_self_training

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        image_path = os.path.join(self.root_dir, name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_name = self.label_list[idx]
        gt_path = os.path.join(self.root_dir, label_name)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        masks = []
        bboxes = []
        categories = []
        gt_masks = decode_mask(torch.tensor(gt_mask[None, :, :])).numpy().astype(np.uint8)
        assert gt_masks.sum() == (gt_mask > 0).sum()
        for mask in gt_masks:
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
            file_name = os.path.splitext(os.path.basename(name))[0]
            origin_image = image
            origin_bboxes = bboxes
            origin_masks = masks
            if self.transform:
                padding, image, masks, bboxes = self.transform(image, masks, np.array(bboxes), True)

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            origin_bboxes = np.stack(origin_bboxes, axis=0)
            origin_masks = np.stack(origin_masks, axis=0)
            return file_name, padding, origin_image, origin_bboxes, origin_masks, image, torch.tensor(bboxes), torch.tensor(masks).float()

        else:
            if self.transform:
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            return image, torch.tensor(bboxes), torch.tensor(masks).float()


def load_datasets_soft(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = PerSegDataset(
        cfg,
        image_root=cfg.datasets.Personal.PerSeg,
        transform=transform,
    )
    soft_train = PerSegDataset(
        cfg,
        image_root=cfg.datasets.Personal.PerSeg,
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

