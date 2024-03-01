from box import Box
from configs.base_config import base_config


config = {
    "gpu_ids": "1",
    "batch_size": 1,
    "val_batchsize": 4,
    "num_workers": 4,
    "num_epochs": 5,
    "max_nums": 50,
    "num_points": 5,
    "resume": False,
    "dataset": "COCO",
    "visual": False,
    "load_type": "soft",
    "prompt": "box",
    "out_dir": "output/COCO/wesam/",
    "name": "base",
    "corrupt": None,
    "visual": False,
    "opt": {
        "learning_rate": 1e-4,
    }
}

cfg = Box(base_config)
cfg.merge_update(config)
