import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from sam_lora import LoRA_Sam


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def get_checkpoint(self, model_type):
        if model_type == "vit_b":
            checkpoint = os.path.join(self.cfg.model.checkpoint, "sam_vit_b_01ec64.pth")
        elif model_type == "vit_l":
            checkpoint = os.path.join(self.cfg.model.checkpoint, "sam_vit_l_0b3195.pth")
        elif model_type == "vit_h":
            checkpoint = os.path.join(self.cfg.model.checkpoint, "sam_vit_h_4b8939.pth")
        else:
            raise ValueError("Model type error!")
        return checkpoint

    def setup(self):
        checkpoint = self.get_checkpoint(self.cfg.model.type)
        self.model = sam_model_registry[self.cfg.model.type](checkpoint=checkpoint)

        self.model.train()
        if self.cfg.model.freeze.image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

        # self.finetune()

    def finetune(self):
        LoRA_Sam(self.model, 4)
        # self.set_norm_layer()
        # self.set_evp_adaptor_layer()
        # self.set_prompt_layer()

    def set_norm_layer(self):
        for name, param in self.model.image_encoder.named_parameters():
            if "norm" in name:
                param.requires_grad = True

    def set_evp_adaptor_layer(self):
        for param in self.model.image_encoder.prompt_generator.parameters():
            param.requires_grad = True

    def set_prompt_layer(self):
        self.model.image_encoder.Prompt_Tokens.requires_grad = True

    def reset_parameters(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                if "linear_a" in name:
                    nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                if "linear_b" in name:
                    nn.init.zeros_(param)

    def forward(self, images, prompts):
        image_embeddings = self.encode(images)
        pred_masks, ious, res_masks = self.decode(prompts, image_embeddings)
        return image_embeddings, pred_masks, ious, res_masks

    def encode(self, images):
        _, _, H, W = images.shape
        self.image_shape = (H, W)
        image_embeddings = self.model.image_encoder(images)
        return image_embeddings 

    def decode(self, prompts, image_embeddings):
        pred_masks = []
        ious = []
        res_masks = []
        for prompt, embedding in zip(prompts, image_embeddings):
            if isinstance(prompt, torch.Tensor):
                prompt = prompt.to(device=embedding.device)
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=prompt,
                masks=None,
            )
            elif isinstance(prompt, tuple):
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=prompt,
                boxes=None,
                masks=None,
            )

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks = F.interpolate(
                low_res_masks,
                self.image_shape,
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)
            res_masks.append(low_res_masks)
        return pred_masks, ious, res_masks

    def get_predictor(self):
        return SamPredictor(self.model)

    def get_generator(self, output_mode):
        return SamAutomaticMaskGenerator(self.model, output_mode=output_mode)