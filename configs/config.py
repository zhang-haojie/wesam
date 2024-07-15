from box import Box
from configs.base_config import base_config


config = {
    "gpu_ids": "4,5,6,7",
    "batch_size": 1,
    "val_batchsize": 4,
    "num_workers": 4,
    "num_iters": 30000,
    "max_nums": 40,
    "num_points": 5,
    "valid_step": 100,
    "dataset": "COCO",
    "prompt": "box",
    "out_dir": "output/debugging/mask_student_50",
    "name": "only_mask_student_50",
    "augment": True,
    "corrupt": None,
    "visual": False,
    "opt": {
        "learning_rate": 1e-4,
    },
    "model": {
        "type": "vit_b",
    },
}

cfg = Box(base_config)
cfg.merge_update(config)
