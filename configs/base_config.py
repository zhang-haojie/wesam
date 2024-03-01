base_config = {
    "eval_interval": 1,
    "ema_rate": 0.999,
    "csv_keys": ["Name", "Prompt", "Mean IoU", "Mean F1", "epoch"],
    "opt": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "corruptions": [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ],
    "model": {
        "type": "vit_b",
        "checkpoint": "./checkpoints/",
        "ckpt": "",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": True,
        },
    },
    "datasets": {
        "coco": {
            "root_dir": "./data/coco2017/val2017",            
            "annotation_file": "./data/coco2017/annotations/instances_val2017.json",
        },
        "PascalVOC": {
            "root_dir": "./data/VOC2012/",
        },
        "sa": {
            "root_dir": "./data/SA-1B",
        },
        "Polyp":{
            "root_dir": "./data/polyp/Kvasir-SEG",
            "annotation_file": "./data/polyp/Kvasir-SEG/kavsir_bboxes.json"
        },
        "ISIC": {
            "root_dir": "./data/ISIC/",
            "train_list": "./data/ISIC/ISBI2016_ISIC_Part1_Training_GroundTruth.csv",
            "test_list": "./data/ISIC/ISBI2016_ISIC_Part1_Test_GroundTruth.csv"
        },
        "ISTD": {
            "train": "./data/ISTD/train/train_A",
            "test": "./data/ISTD/test/test_A",
        },
        "MSD": {
            "train": "./data/MSD/train/image",
            "test": "./data/MSD/test/image",
        },
        "GDD": {
            "train": "./data/GDD/train/image",
            "test": "./data/GDD/test/image",
        },
        "CAMO":{
            "GT": "./data/CAMO-V.1.0-CVIU2019/GT",
            "train": "./data/CAMO-V.1.0-CVIU2019/Images/Train",
            "test": "./data/CAMO-V.1.0-CVIU2019/Images/Test",
        },
        "COD10K":{
            "GT": "./data/COD10K-v2/Test/GT_Object",
            "test": "./data/COD10K-v2/Test/Image",
        },
        "robot": {
            "OCID": "./data/OCID-dataset",
            "OSD": "./data/OSD-0.2-depth"
        },
    },
}
