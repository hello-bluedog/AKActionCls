#!/usr/bin/env python
# coding=utf-8
import pickle
file_path = './8x224x4x3.pkl'
with open(file_path, 'rb') as file:
    # 使用pickle.load()方法读取数据
    data = pickle.load(file)

labels = data["video_labels"]
preds = data["video_preds"]


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
    preds_all = preds[:, ~(np.all(labels == 0, axis=0))]
    labels_all = labels[:, ~(np.all(labels == 0, axis=0))]
    print(preds_all.shape)
    print(labels_all.shape)
    preds_head = preds[[i for i in range(preds.shape[0]) if ~np.all(preds[i][head_index] == 0)], :]
    labels_head = labels[[i for i in range(labels.shape[0]) if ~np.all(labels[i][head_index] == 0)], :]
    preds_middle = preds[[i for i in range(preds.shape[0]) if ~np.all(preds[i][middle_index] == 0)], :]
    labels_middle = labels[[i for i in range(labels.shape[0]) if ~np.all(labels[i][middle_index] == 0)], :]
    preds_tail = preds[[i for i in range(preds.shape[0]) if ~np.all(preds[i][tail_index] == 0)], :]
    labels_tail = labels[[i for i in range(labels.shape[0]) if ~np.all(labels[i][tail_index] == 0)], :]
    preds_head = preds[:, ~(np.all(labels_head == 0, axis=0))]
    labels_head = labels[:, ~(np.all(labels_head == 0, axis=0))]
    preds_middle = preds[:, ~(np.all(labels_middle == 0, axis=0))]
    labels_middle = labels[:, ~(np.all(labels_middle == 0, axis=0))]
    preds_tail = preds[:, ~(np.all(labels_tail == 0, axis=0))]
    labels_tail = labels[:, ~(np.all(labels_tail == 0, axis=0))]
    aps = [0]
    aps_head = [0]
    aps_middle = [0]
    aps_tail = [0]
    print(preds_tail)
    try:
        aps = average_precision_score(labels, preds, average=None)
        aps_head = average_precision_score(labels_head, preds_head, average=None)
        aps_middle = average_precision_score(labels_middle, preds_middle, average=None)
        aps_tail = average_precision_score(labels_tail, preds_tail, average=None)
    except ValueError:
        print(
            "Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample."
        )
    print("aps shape")
    print(aps.shape)

    mean_ap = np.mean(aps[tail_index])
    return mean_ap, np.mean(aps_head), np.mean(aps_middle), np.mean(aps_tail)

def get_map1(preds, labels):
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
    preds_all = preds[:, ~(np.all(labels == 0, axis=0))]
    labels_all = labels[:, ~(np.all(labels == 0, axis=0))]
    preds_head = preds[[i for i in range(preds.shape[0]) if ~np.all(preds[i][head_index] == 0)], :]
    labels_head = labels[[i for i in range(labels.shape[0]) if ~np.all(labels[i][head_index] == 0)], :]
    try:
        aps = average_precision_score(labels, preds, average=None)
        aps_head = average_precision_score(labels_head, preds_head, average=None)
        aps_middle = average_precision_score(labels_middle, preds_middle, average=None)
        aps_tail = average_precision_score(labels_tail, preds_tail, average=None)
    except ValueError:
        print(
            "Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample."
        )
    print("aps shape")
    print(aps.shape)

    mean_ap = np.mean(aps[tail_index])
    return mean_ap, np.mean(aps_head), np.mean(aps_middle), np.mean(aps_tail)
