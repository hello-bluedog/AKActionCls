#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from functools import partial

from losses import EQLv2

ls = EQLv2(reduction="mean")

preds = torch.randn((8, 140))
labels = torch.zeros((8, 140))
ls.forward(preds, labels)
