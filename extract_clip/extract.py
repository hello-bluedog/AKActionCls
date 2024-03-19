#!/usr/bin/env python
# coding=utf-8
import clip as clip
import torch
from collections import OrderedDict
import os



model_dir = "/mount/ccai_nas/yunzhu/Animal_Kingdom/model/"
model_name = "ViT-L-14.pt"

model, _ = clip.load(os.path.join(model_dir, model_name), device='cpu')
new_state_dict = OrderedDict()
for k, v in model.state_dict().items():
    if 'visual.' in k:
        if k[7:] not in ["proj", "ln_post.weight", "ln_post.bias"]:
            new_state_dict[k[7:]] = v
torch.save(new_state_dict, os.path.join(model_dir, 'vit_l14.pth'))
