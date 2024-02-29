import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from tools import ResizeAndPad, soft_transform, collate_fn, collate_fn_soft, collate_fn_
from skimage.draw import polygon2mask


class COCOoneshotDataset(Dataset):
    def __init__(self, cfg, root_dir, annotation_file, cat_id=None, transform=None, training=False, if_self_training=False):
        self.cfg = cfg
        self.root_dir = root_dir
        self.transform = transform
        self.cat_ids = [cat_id]
        self.coco = COCO(annotation_file)
        all_image_ids = sorted(list(self.coco.imgs.keys()))

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

        # Filter out image_ids without any annotations
        self.image_ids = [
            image_id
            for image_id in image_ids
            if len(self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids)) > 0
        ]

        self.if_self_training = if_self_training

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info["file_name"])
        image = cv2.imread(image_path)
        # corrupt_image(image, image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids)
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


def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = COCOoneshotDataset(
        cfg,
        root_dir=cfg.dataset.val.root_dir,
        annotation_file=cfg.dataset.val.annotation_file,
        cat_id = 1,
        transform=transform,
    )
    soft_train = COCOoneshotDataset(
        cfg,
        root_dir=cfg.dataset.val.root_dir,
        annotation_file=cfg.dataset.val.annotation_file,
        cat_id = 1,
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