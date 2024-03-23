#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm


import pickle
dir = "/mount/ccai_nas/yunzhu/Animal_Kingdom/output/pretrain_l14"
model_eqlv2_b16 = './vit-b16-EQLv2-65.pkl'
model_eql_b16 = "./vit-b16-EQL-70.pkl"
model_eqlv2_l14 = "./vit-eqlv2-L14-65.pkl"


def read_label(file_path : str):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    labels = data["video_labels"]
    preds = data["video_preds"]
    return labels, preds


df = pd.read_excel("df_action.xlsx")
DF_PATH = "df_action.xlsx"

def get_map(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes.
    Returns:
        mean_ap (int): final mAP score.
    """

    df = pd.read_excel(DF_PATH)
    head_index = df.index[df["segment"] == "head"].tolist()
    middle_index = df.index[df["segment"] == "middle"].tolist()
    tail_index = df.index[df["segment"] == "tail"].tolist()
    #preds = preds[:, ~(np.all(labels == 0, axis=0))]
    #labels = labels[:, ~(np.all(labels == 0, axis=0))]
    try:
        aps = average_precision_score(labels, preds, average=None)
    except ValueError:
        print(
            "Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample."
        )
    #print("aps shape")

    mean_ap = np.sum(aps) / (aps != 0).sum()
    tail_ap = np.sum(aps[tail_index]) / (aps[tail_index] != 0).sum()
    middle_ap = np.sum(aps[middle_index]) / (aps[middle_index] != 0).sum()
    head_ap = np.sum(aps[head_index]) / (aps[head_index] != 0).sum()
    return mean_ap, head_ap, middle_ap, tail_ap

label1, pred1 = read_label(model_eql_b16)
label2, pred2 = read_label(model_eqlv2_b16)
label3, pred3 = read_label(model_eqlv2_l14)

label = label1
max_o = 0
max_h = 0
max_m = 0
max_t = 0
config1 = None
config2 = None
config3 = None
config4 = None
'''for i in tqdm(np.linspace(0,1,100)):
    for j in np.linspace(i, 1, 100):
        k = 1 - i - j
        pred = pred1 * i + pred2 * j + pred3 * k
        o, h, m, t = get_map(pred, label)
        if o > max_o:
            max_o = o
            config1 = [i, j, k]
        if h > max_h:
            max_h = h
            config2 = [i, j, k]
        if t > max_t:
            max_t = t
            config3 = [i, j, k]
        if m > max_m:
            max_m = m
            config4 = [i, j, k]

print(config1, config2, config3, config4)'''
i = 0.25
j = 0.31
k = 0.44
pred = pred1 * i + pred2 * j + pred3 * k
print(get_map(pred, label))
