import clip as clip
import torch
import os

model_dir = "/mount/ccai_nas/yunzhu/Animal_Kingdom/model/"
UMT_model_name = "b16_ptk710_ftk710_ftk600_f8_res224.pth"
vit_model_name = "vit_b16.pth"

vit_prefix = "transformer.resblocks"
UMT_prefix = "blocks"
vit_names = ["attn.in_proj_weight", "attn.in_proj_bias", "attn.out_proj.weight", "attn.out_proj.bias", "ln_1.weight", "ln_1.bias", "mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias", "ln_2.weight", "ln_2.bias"]
UMT_names = ["attn.qkv.weight", None, "attn.proj.weight", "attn.proj.bias", "norm1.weight", "norm1.bias", "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias", "norm2.weight", "norm1.bias"]

UMT = torch.load(os.path.join(model_dir, UMT_model_name), map_location="cpu")
vit = torch.load(os.path.join(model_dir, vit_model_name), map_location="cpu")


for i in range(12):
    for j, k in enumerate(vit_names):
        k_vit = ".".join([vit_prefix, str(i), vit_names[j]])
        if UMT_names[j] is None:
            #vit[k_vit].zero_()
            continue
        k_umt = ".".join([UMT_prefix, str(i), UMT_names[j]])
        vit[k_vit] = UMT[k_umt]
k_conv = "conv1.weight"
k_umt_conv = "patch_embed.proj.weight"
vit[k_conv] = UMT[k_umt_conv].squeeze(dim=2)
torch.save(vit, os.path.join(model_dir, 'vit_b16_k400_bias.pth'))
