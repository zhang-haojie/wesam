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
        "checkpoint": "/mnt/sdb/zhang.haojie/checkpoints/sam/",
        "ckpt": "",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": True,
        },
    },
    "datasets": {
        "train": {
            "root_dir": "/mnt/sdb/zhang.haojie/Dataset/detection/coco2017/train2017",
            "annotation_file": "/mnt/sdb/zhang.haojie/Dataset/detection/coco2017/annotations/instances_train2017.json",
        },
        "val": {
            "root_dir": "/mnt/sdb/zhang.haojie/Dataset/detection/coco2017/val2017",
            "annotation_file": "/mnt/sdb/zhang.haojie/Dataset/detection/coco2017/annotations/instances_val2017.json",
        },
        "lvis": {
            "root_dir": "/mnt/sdb/zhang.haojie/Dataset/detection/coco2017",
            "annotation_file": "/mnt/sdb/zhang.haojie/Dataset/detection/LVIS/lvis_v1_val.json"
        },
        "PascalVOC": {
            "root_dir": "/mnt/sdb/zhang.haojie/Dataset/detection/VOC2012/",
            "peopleart_dir": "/mnt/sdb/zhang.haojie/Dataset/detection/PeopleArt"
        },
        "prompt":{
            "root_dir": "/mnt/sdb/zhang.haojie/Dataset/detection/coco2017/val2017",
            "annotation_file": "/mnt/sdb/zhang.haojie/Dataset/detection/coco2017/annotations/prompts_val2017.json",
        },
        "sa": {
            "root_dir": "/mnt/sdb/zhang.haojie/Dataset/segmentation/SA-1B",
        },
        "memory": {
            "root_dir": "/mnt/sdb/zhang.haojie/Dataset/segmentation/memory",
            "train": "/mnt/sdb/zhang.haojie/Dataset/segmentation/memory/train.txt",
            "test": "/mnt/sdb/zhang.haojie/Dataset/segmentation/memory/test.txt"
        },
        "Polyp":{
            "root_dir": "/mnt/sdb/zhang.haojie/Dataset/segmentation/polyp/Kvasir-SEG",
            "annotation_file": "/mnt/sdb/zhang.haojie/Dataset/segmentation/polyp/Kvasir-SEG/kavsir_bboxes.json"
        },
        "ISIC": {
            "root_dir": "/mnt/sdb/zhang.haojie/Dataset/segmentation/ISIC/",
            "train_list": "/mnt/sdb/zhang.haojie/Dataset/segmentation/ISIC/ISBI2016_ISIC_Part1_Training_GroundTruth.csv",
            "test_list": "/mnt/sdb/zhang.haojie/Dataset/segmentation/ISIC/ISBI2016_ISIC_Part1_Test_GroundTruth.csv"
        },
        "ISTD": {
            "train": "/mnt/sdb/zhang.haojie/Dataset/segmentation/ISTD/train/train_A",
            "test": "/mnt/sdb/zhang.haojie/Dataset/segmentation/ISTD/test/test_A",
        },
        "MSD": {
            "train": "/mnt/sdb/zhang.haojie/Dataset/segmentation/MSD/train/image",
            "test": "/mnt/sdb/zhang.haojie/Dataset/segmentation/MSD/test/image",
        },
        "GDD": {
            "train": "/mnt/sdb/zhang.haojie/Dataset/segmentation/GDD/train/image",
            "test": "/mnt/sdb/zhang.haojie/Dataset/segmentation/GDD/test/image",
        },
        "CAMO":{
            "GT": "/mnt/sdb/zhang.haojie/Dataset/segmentation/CAMO-V.1.0-CVIU2019/GT",
            "train": "/mnt/sdb/zhang.haojie/Dataset/segmentation/CAMO-V.1.0-CVIU2019/Images/Train",
            "test": "/mnt/sdb/zhang.haojie/Dataset/segmentation/CAMO-V.1.0-CVIU2019/Images/Test",
        },
        "COD10K":{
            "GT": "/mnt/sdb/zhang.haojie/Dataset/segmentation/COD10K-v2/Test/GT_Object",
            "train": "/mnt/sdb/zhang.haojie/Dataset/segmentation/COD/TrainDataset/Imgs",
            "test": "/mnt/sdb/zhang.haojie/Dataset/segmentation/COD10K-v2/Test/Image",
        },
        "COD": {
            "train": "/mnt/sdb/zhang.haojie/Dataset/segmentation/COD10K-v2/TrainDataset/Imgs",
            "CAMO": "/mnt/sdb/zhang.haojie/Dataset/segmentation/COD10K-v2/TestDataset/CAMO/Imgs",
            "CHAMELEON": "/mnt/sdb/zhang.haojie/Dataset/segmentation/COD10K-v2/TestDataset/CHAMELEON/Imgs",
            "COD10k": "/mnt/sdb/zhang.haojie/Dataset/segmentation/COD10K-v2/TestDataset/COD10K/Imgs",
        },
        "indoor": {
            "OCID": "/mnt/sdb/zhang.haojie/Dataset/segmentation/OCID-dataset",
            "OSD": "/mnt/sdb/zhang.haojie/Dataset/segmentation/OSD-0.2-depth"
        },
        "Personal": {
            "PerSeg": "/mnt/sdb/zhang.haojie/Dataset/oneshot/PerSeg/data/Images"
        }
    },
}
