from box import Box
from configs.base_config import base_config


config = {
    "gpu_ids": "0,1,2,3",
    "batch_size": 1,
    "val_batchsize": 4,
    "num_workers": 4,
    "num_iters": 40000,
    "max_nums": 40,
    "num_points": 5,
    "eval_interval": 1,
    "dataset": "COCO",
    "prompt": "box",
    "out_dir": "output/benchmark/COCO",
    "name": "baseline",
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
