import os
import csv
import torch
import copy
import numpy as np
from torchsummary import summary

def freeze(model: torch.nn.Module):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def momentum_update(student_model, teacher_model, momentum=0.99):
    for (src_name, src_param), (tgt_name, tgt_param) in zip(
        student_model.named_parameters(), teacher_model.named_parameters()
    ):
        if src_param.requires_grad:
            tgt_param.data.mul_(momentum).add_(src_param.data, alpha=1 - momentum)


def decode_mask(mask):
    """
    Convert mask with shape [1, h, w] using 1, 2, 3, ... to represent different objects
    to a mask with shape [n, h, w] using a new dimension to represent the number of objects.

    Args:
        mask (torch.Tensor): Mask tensor with shape [1, h, w] using 1, 2, 3, ... to represent different objects.

    Returns:
        torch.Tensor: Mask tensor with shape [n, h, w] using a new dimension to represent the number of objects.
    """
    unique_labels = torch.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]
    n_objects = len(unique_labels)
    new_mask = torch.zeros((n_objects, *mask.shape[1:]), dtype=torch.int64)
    for i, label in enumerate(unique_labels):
        new_mask[i] = (mask == label).squeeze(0)
    return new_mask


def encode_mask(mask):
    """
    Convert mask with shape [n, h, w] using a new dimension to represent the number of objects
    to a mask with shape [1, h, w] using 1, 2, 3, ... to represent different objects.

    Args:
        mask (torch.Tensor): Mask tensor with shape [n, h, w] using a new dimension to represent the number of objects.

    Returns:
        torch.Tensor: Mask tensor with shape [1, h, w] using 1, 2, 3, ... to represent different objects.
    """
    n_objects = mask.shape[0]
    new_mask = torch.zeros((1, *mask.shape[1:]), dtype=torch.int64)
    for i in range(n_objects):
        new_mask[0][mask[i] == 1] = i + 1
    return new_mask


def copy_model(model: torch.nn.Module):
    new_model = copy.deepcopy(model)
    freeze(new_model)
    return new_model


def create_csv(filename, csv_head=["corrupt", "Mean IoU", "Mean F1", "epoch"]):
    if os.path.exists(filename):
        return 
    with open(filename, 'w') as csvfile:
        csv_write = csv.DictWriter(csvfile, fieldnames=csv_head)
        csv_write.writeheader()


def write_csv(filename, csv_dict, csv_head=["corrupt", "Mean IoU", "Mean F1", "epoch"]):
    with open(filename, 'a+') as csvfile:
        csv_write = csv.DictWriter(csvfile, fieldnames=csv_head, extrasaction='ignore')
        csv_write.writerow(csv_dict)


def check_grad(model: torch.nn.Module):
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")


def check_model(model):
    return summary(model, (3, 1024, 1024), batch_size=1, device="cuda")


def reduce_instances(bboxes, gt_masks, max_nums=50):
    bboxes_ = []
    gt_masks_ = []
    for bbox, gt_mask in zip(bboxes, gt_masks):
        idx = np.arange(bbox.shape[0])
        np.random.shuffle(idx)
        bboxes_.append(bbox[idx[:max_nums]])
        gt_masks_.append(gt_mask[idx[:max_nums]])

    bboxes = bboxes_
    gt_masks = gt_masks_
    return bboxes, gt_masks
