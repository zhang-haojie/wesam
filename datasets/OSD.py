import os
import cv2
import random
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from skimage.draw import polygon2mask
from pathlib import Path
from PIL import Image
from datasets.tools import ResizeAndPad, soft_transform, collate_fn, decode_mask, collate_fn_


class OSDObject(Dataset):
    def __init__(self, cfg, root_dir, transform=None, split=False, training=False, if_self_training=False):
        self.cfg = cfg
        self._osd_object_path = root_dir
        self.transform = transform
        # get all images
        data_path = os.path.join(self._osd_object_path, 'image_color')
        all_image_paths = sorted(glob.glob(data_path + '/*.png'))
        all_image_paths = self.check_empty(all_image_paths)

        if split:
            train_image_paths = []
            eval_image_paths = []
            while all_image_paths:
                for _ in range(6):
                    if all_image_paths:
                        train_image_paths.append(all_image_paths.pop(0))
                if all_image_paths:
                    eval_image_paths.append(all_image_paths.pop(0))

            if training:
                random.shuffle(train_image_paths)
                image_paths = train_image_paths
            else:
                random.shuffle(eval_image_paths)
                image_paths = eval_image_paths
        else:
            image_paths = all_image_paths

        self.image_files = image_paths
        # self.image_files = all_image_paths

        self.if_self_training = if_self_training
        assert os.path.exists(self._osd_object_path), \
                'osd_object path does not exist: {}'.format(self._osd_object_path)

    def process_label(self, foreground_labels):
        """ Process foreground_labels
                - Map the foreground_labels to {0, 1, ..., K-1}

            @param foreground_labels: a [H x W] numpy array of labels

            @return: foreground_labels
        """
        # Find the unique (nonnegative) foreground_labels, map them to {0, ..., K-1}
        unique_nonnegative_indices = np.unique(foreground_labels)
        mapped_labels = foreground_labels.copy()
        for k in range(unique_nonnegative_indices.shape[0]):
            mapped_labels[foreground_labels == unique_nonnegative_indices[k]] = k
        foreground_labels = mapped_labels
        return foreground_labels

    def __len__(self):
        return len(self.image_files)

    def check_empty(self, image_paths):
        new_image_paths = []
        for filename in image_paths:
            labels_filename = str(filename).replace('image_color', 'annotation')
            annotation = Image.open(labels_filename)
            foreground_labels = np.array(annotation)
            # mask table as background
            foreground_labels[foreground_labels == 1] = 0
            if 'table' in labels_filename:
                foreground_labels[foreground_labels == 2] = 0
            gt_mask = self.process_label(foreground_labels)
            if not np.all(gt_mask == 0):
                new_image_paths.append(filename)
        return new_image_paths

    def __getitem__(self, idx):
        filename = self.image_files[idx]

        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.cfg.get_prompt:
            image_info = {}
            height, width, _ = image.shape
            image_info["file_path"] = filename
            image_info["height"] = height
            image_info["width"] = width
            return idx, image_info, image

        labels_filename = filename.replace('image_color', 'annotation')
        annotation = Image.open(labels_filename)
        foreground_labels = np.array(annotation)

        # mask table as background
        foreground_labels[foreground_labels == 1] = 0
        if 'table' in labels_filename:
            foreground_labels[foreground_labels == 2] = 0
        gt_mask = self.process_label(foreground_labels)

        bboxes = []
        masks = []
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


class OSDObjectwithCoarse(OSDObject):

    def __getitem__(self, idx):
        filename = self.image_files[idx]

        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        labels_filename = filename.replace('image_color', 'annotation')
        annotation = Image.open(labels_filename)
        foreground_labels = np.array(annotation)

        # mask table as background
        foreground_labels[foreground_labels == 1] = 0
        if 'table' in labels_filename:
            foreground_labels[foreground_labels == 2] = 0
        gt_mask = self.process_label(foreground_labels)

        bboxes = []
        masks = []
        categories = []
        gt_masks = decode_mask(torch.tensor(gt_mask[None, :, :])).numpy().astype(np.uint8)
        assert gt_masks.sum() == (gt_mask > 0).sum()
        for mask in gt_masks:
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
    train = OSDObject(
        cfg,
        root_dir=cfg.datasets.robot.OSD,
        transform=transform,
        split=cfg.split,
        training=True,
        if_self_training=cfg.augment,
    )
    val = OSDObject(
        cfg,
        root_dir=cfg.datasets.robot.OSD,
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
    train = OSDObjectwithCoarse(
        cfg,
        root_dir=cfg.datasets.robot.OSD,
        transform=transform,
        split=cfg.split,
        training=True,
        if_self_training=cfg.augment,
    )
    val = OSDObjectwithCoarse(
        cfg,
        root_dir=cfg.datasets.robot.OSD,
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


def load_datasets_prompt(cfg, img_size):
    transform = ResizeAndPad(img_size)
    train = OSDObject(
        cfg,
        root_dir=cfg.datasets.robot.OSD,
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
