## Prepare


### Download Dataset

*Natural images*

- [COCO Dataset](https://cocodataset.org/)

- [PascalVOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)


*Medical images*

- [Kvasir-SEG Dataset](https://datasets.simula.no/kvasir-seg/)

- [ISIC Dataset](https://challenge.isic-archive.com/data/)


*Camouflaged Objects*

- [COD10k Dataset](https://drive.google.com/file/d/1pVq1rWXCwkMbEZpTt4-yUQ3NsnQd_DNY/view?usp=sharing) - [Camouflaged Object Detection](https://github.com/DengPingFan/SINet/)

- [CAMO](https://drive.google.com/open?id=1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6) - [Project](https://sites.google.com/view/ltnghia/research/camo)


*Robotic Images*

- [OCID](https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-clutter-indoor-dataset/): *Object Clutter Indoor Dataset*

- [OSD](https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/osd/): *Object Segmentation Database*


*Corrupted Images*

In `datasets/COCO.py`, uncomment the line that includes `corrupt_image`. Then comment line 192 of `adaptation.py` and run it.

```
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info["file_name"])
        if self.cfg.corrupt in self.cfg.corruptions:
            image_path = image_path.replace("val2017", os.path.join("corruption", self.cfg.corrupt))
        image = cv2.imread(image_path)

        # corrupt_image(image, image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```


*Glass*
- [GDD](http://gdd.dluticcd.com/) - 
[Don't Hit Me! Glass Detection in Real-world Scenes](https://github.com/Charmve/Mirror-Glass-Detection/tree/master/CVPR2020_GDNet) 
(*Glass Detection Dataset* Need Apply!)

- [GSD](https://drive.google.com/file/d/1pSEUs-8I-4YHOTJ9J0wxiEpkdbHjMQQo/view?usp=sharing) - 
[Exploiting Semantic Relations for Glass Surface Detection](https://jiaying.link/neurips2022-gsds/)
(*Glass Surface Dataset*)


*Mirror*
- [MSD](https://drive.google.com/file/d/1Znw92fO6lCKfXejjSSyMyL1qtFepgjPI/view?usp=sharing) - 
[Where is My Mirror?](https://github.com/Charmve/Mirror-Glass-Detection/tree/master/ICCV2019_MirrorNet) 
(*mirror segmentation dataset* Need Apply!)


*Shadow Detection*
- [ISTD](https://drive.google.com/file/d/1I0qw-65KBA6np8vIZzO6oeiOvcDBttAY/view) - 
[tacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal](https://github.com/DeepInsight-PCALab/ST-CGAN)
(*Image Shadow Triplets dataset* Need Apply!)


### Download Checkpoints

Click the links below to download the checkpoint for the corresponding model type.

- `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)


### Prepare

```
cd wesam/

mkdir data
mkdir checkpoints

mv DATASETS ./data

mv VIT_B ./checkpoints
```