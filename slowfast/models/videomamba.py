#!/usr/bin/env python
import json
import torch
import torch.nn as nn

import slowfast.models.videomamba_model as model
from .build import MODEL_REGISTRY

import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)


@MODEL_REGISTRY.register()
class Videomamba(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg,
        backbone = cfg.VIDEOMAMBA.BACKBONE
        # img_size=cfg.DATA.IMG_SIZE, 
        # patch_size=cfg.DATA.PATCH_SIZE,
        # depth=cfg.MODEL.DEPTH, 
        # embed_dim=cfg.MODEL.N_DIM, 
        # channels=cfg.MODEL.channels, 
        # num_classes=cfg.MODEL.num_classes,
        # drop_rate=cfg.MODEL.drop_rate,
        # drop_path_rate=cfg.MODEL.drop_path_rate,
        # norm_epsilon=cfg.MODEL.norm_epsilon, 
        # fused_add_norm=cfg.MODEL.fused_add_norm,
        # rms_norm=cfg.MODEL.rms_norm, 
        # residual_in_fp32=cfg.MODEL.residual_in_fp32,
        # bimamba=cfg.VIDEOMAMBA.bimamba,
        # # video
        # kernel_size=cfg.MODEL.kernel_size, 
        # num_frames=cfg.DATA.num_frames, 
        # fc_drop_rate=cfg.MODEL.fc_drop_rate, 
        # # checkpoint
        # use_checkpoint=cfg.MODEL.use_checkpoint,
        # checkpoint_num=cfg.MODEL.checkpoint_num,  


        # pre-trained from CLIP
        self.backbone = model.__dict__[backbone]()
            # img_size = img_size, 
            # patch_size = patch_size,
            # depth = depth, 
            # embed_dim = embed_dim, 
            # channels = channels, 
            # num_classes = num_classes,
            # drop_rate = drop_rate,
            # drop_path_rate = drop_path_rate,
            # norm_epsilon = norm_epsilon, 
            # fused_add_norm = fused_add_norm,
            # rms_norm = rms_norm, 
            # residual_in_fp32 = residual_in_fp32,
            # bimamba = bimamba,
            # # video
            # kernel_size = kernel_size, 
            # num_frames = num_frames, 
            # fc_drop_rate = fc_drop_rate, 
            # # checkpoint
            # use_checkpoint = use_checkpoint,
            # checkpoint_num = checkpoint_num,  

        if cfg.VIDEOMAMBA.PRETRAIN != '':
            # Load Kineti-700 pretrained model
            logger.info(f'load model from {cfg.VIDEOMAMBA.PRETRAIN}')
            state_dict = torch.load(cfg.VIDEOMAMBA.PRETRAIN, map_location='cpu')
            if 'head.weight' in state_dict.keys():
                if cfg.UNIFORMERV2.DELETE_SPECIAL_HEAD:
                    print("Removing head from pretrained checkpoint")
                    del state_dict['head.weight']
                    del state_dict['head.bias']
            self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = x[0]
        output = self.backbone(x)

        return output
