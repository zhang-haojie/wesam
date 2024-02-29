import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from skimage.draw import polygon2mask
from datasets.tools import ResizeAndPad, soft_transform, soft_transform_all, collate_fn, collate_fn_soft, collate_fn_, jitter_bbox


class COCODataset(Dataset):
    def __init__(self, cfg, root_dir, annotation_file, rate=(6, 1), transform=None, training=False, if_self_training=False):
        self.cfg = cfg
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
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

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info["file_name"])
        if self.cfg.corrupt in self.cfg.corruptions:
            image_path = image_path.replace("val2017", os.path.join("corruption", self.cfg.corrupt))
        image = cv2.imread(image_path)
        # corrupt_image(image, image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        masks = []
        categories = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            bboxes.append([x, y, x + w, y + h])
            mask = self.coco.annToMask(ann)
            masks.append(mask)
            categories.append(ann["category_id"])

        # bboxes = jitter_bbox(bboxes, image.shape, self.cfg.jitter)

        if self.if_self_training:
            image_weak, bboxes_weak, masks_weak, image_strong = soft_transform(image, bboxes, masks, categories)
            # image_origin = image_weak

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


class COCODatasetwithCoarse(COCODataset):

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info["file_name"])
        if self.cfg.corrupt in self.cfg.corruptions:
            image_path = image_path.replace("val2017", os.path.join("corruption", self.cfg.corrupt))
        image = cv2.imread(image_path)
        # corrupt_image(image, image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        # image_ = image.copy()
        bboxes = []
        masks = []
        # keypoints = []
        coarse_masks = []
        categories = []
        approxes = []
        # i = 0
        for ann in anns:
            # x, y, w, h = ann["bbox"]
            # bboxes.append([x, y, x + w, y + h])
            mask = self.coco.annToMask(ann)

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            num_vertices = 0.05 * cv2.arcLength(contours[0], True)
            num_vertices = num_vertices if num_vertices > 3 else 3
            approx = cv2.approxPolyDP(contours[0], num_vertices, True)  # [x, y]
            approx = approx.squeeze(1)

            # cv2.drawContours(image_, [contours[0]], 0, (255, 0, 0), 2)
            # cv2.drawContours(image_, [approx], 0, (255, 0, 0), 2)
            # coordinates = [[y, x] for [x, y] in approx]
            coordinates = np.array(approx)
            x_max, x_min = max(coordinates[:, 0]), min(coordinates[:, 0])
            y_max, y_min = max(coordinates[:, 1]), min(coordinates[:, 1])
            coarse_mask = polygon2mask(mask.shape, coordinates).astype(mask.dtype)            
            if x_min == x_max or y_min == y_max:
                x, y, w, h = cv2.boundingRect(mask)
                bboxes.append([x, y, x + w, y + h])
            else:
                bboxes.append([x_min, y_min, x_max, y_max])
            # if x_min == x_max:
            #     x_min = max(x_min - 1, 0)
            #     x_max = min(x_max + 1, mask.shape[1])
            # if y_min == y_max:
            #     y_min = max(y_min - 1, 0)
            #     y_max = min(y_max + 1, mask.shape[0])

            # cv2.rectangle(image_, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2, 4)
            # cv2.imwrite(f"/home/zhang.haojie/workspace/data/coco/coarse/mask_{i}.jpg", mask.astype(np.uint8)*255)
            # i += 1

            masks.append(mask)
            # keypoints.extend(coordinates)
            # keypoints.append(coordinates)
            coarse_masks.append(coarse_mask)
            # bboxes.append([x_min, y_min, x_max, y_max])
            approxes.append(approx)
            categories.append(ann["category_id"])

        # cv2.imwrite("/home/zhang.haojie/workspace/data/coco/coarse/test.jpg", image_)        
        if self.if_self_training:
            image_weak, bboxes_weak, masks_weak, image_strong = soft_transform(image, bboxes, masks, categories)
            # image_weak, bboxes_weak, masks_weak, keypoints_weak, image_strong = soft_transform_all(image, bboxes, coarse_masks, keypoints, categories)
            # new_keypoints = []
            # for bbox in bboxes_weak:
            #     for point in keypoints_weak:
            #         if  bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]:
            #             new_keypoints.append(point)

            if self.transform:
                image_weak, masks_weak, bboxes_weak = self.transform(image_weak, masks_weak, np.array(bboxes_weak))
                image_strong = self.transform.transform_image(image_strong)
                # points_weak = self.transform.transform_coords(image_weak, new_keypoints)

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
            # origin_approxes = np.stack(origin_approxes, axis=0)
            origin_masks = np.stack(origin_masks, axis=0)
            return image_id, padding, origin_image, origin_approxes, origin_masks, image, torch.tensor(bboxes), torch.tensor(masks).float()

        else:
            if self.transform:
                _, coarse_masks, _ = self.transform(image, coarse_masks, np.array(bboxes))
                # new_keypoints = []
                # for points in keypoints:
                #     new_keypoints.append(self.transform.transform_coords(points, image))
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            coarse_masks = np.stack(coarse_masks, axis=0)
            return image, torch.tensor(bboxes), torch.tensor(masks).float()



def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    train = COCODataset(
        cfg,
        root_dir=cfg.datasets.val.root_dir,
        annotation_file=cfg.datasets.val.annotation_file,
        transform=transform,
        training=True,
    )
    val = COCODataset(
        cfg,
        root_dir=cfg.datasets.val.root_dir,
        annotation_file=cfg.datasets.val.annotation_file,
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
        batch_size=cfg.val_batchsize,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    return train_dataloader, val_dataloader


def load_datasets_soft(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = COCODataset(
        cfg,
        root_dir=cfg.datasets.val.root_dir,
        annotation_file=cfg.datasets.val.annotation_file,
        transform=transform,
    )
    soft_train = COCODataset(
        cfg,
        root_dir=cfg.datasets.val.root_dir,
        annotation_file=cfg.datasets.val.annotation_file,
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


def load_datasets_coarse(cfg, img_size):
    transform = ResizeAndPad(img_size)
    train = COCODatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.val.root_dir,
        annotation_file=cfg.datasets.val.annotation_file,
        transform=transform,
        training=True,
    )
    val = COCODatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.val.root_dir,
        annotation_file=cfg.datasets.val.annotation_file,
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
        batch_size=cfg.val_batchsize,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    return train_dataloader, val_dataloader


def load_datasets_soft_coarse(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = COCODatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.val.root_dir,
        annotation_file=cfg.datasets.val.annotation_file,
        transform=transform,
    )
    soft_train = COCODatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.val.root_dir,
        annotation_file=cfg.datasets.val.annotation_file,
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


def load_datasets_soft_all(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = COCODataset(
        cfg,
        root_dir=cfg.datasets.val.root_dir,
        annotation_file=cfg.datasets.val.annotation_file,
        transform=transform,
    )
    val_coarse = COCODatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.val.root_dir,
        annotation_file=cfg.datasets.val.annotation_file,
        transform=transform,
    )
    soft_train = COCODataset(
        cfg,
        root_dir=cfg.datasets.val.root_dir,
        annotation_file=cfg.datasets.val.annotation_file,
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
    val_coarse_dataloader = DataLoader(
        val_coarse,
        batch_size=cfg.val_batchsize,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    soft_train_dataloader = DataLoader(
        soft_train,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_soft,
    )
    return soft_train_dataloader, val_dataloader, val_coarse_dataloader


def load_datasets_visual(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = COCODataset(
        cfg,
        root_dir=cfg.datasets.val.root_dir,
        annotation_file=cfg.datasets.val.annotation_file,
        transform=transform,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_,
    )
    return val_dataloader


def load_datasets_visual_coarse(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = COCODatasetwithCoarse(
        cfg,
        root_dir=cfg.datasets.val.root_dir,
        annotation_file=cfg.datasets.val.annotation_file,
        transform=transform,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_,
    )
    return val_dataloader
