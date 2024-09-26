import h5py
import json
from tqdm import tqdm
import pickle
import os 
import numpy as np
import pandas as pd 
import torch 

def compute_temporal_iou(pred, gt):
    """ deprecated due to performance concerns
    compute intersection-over-union along temporal axis
    Args:
        pred: [st (float), ed (float)]
        gt: [st (float), ed (float)]
    Returns:
        iou (float):
    Ref: https://github.com/LisaAnne/LocalizingMoments/blob/master/utils/eval.py
    """
    intersection = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])  # not the correct union though
    if union == 0:
        return 0
    else:
        return 1.0 * intersection / union


def temporal_nms(predictions, nms_thd, max_after_nms=100):
    """
    Args:
        predictions: list(sublist), each sublist is [st (float), ed(float), score (float)],
            note larger scores are better and are preserved. For metrics that are better when smaller,
            please convert to its negative, e.g., convert distance to negative distance.
        nms_thd: float in [0, 1]
        max_after_nms:
    Returns:
        predictions_after_nms: list(sublist), each sublist is [st (float), ed(float), score (float)]
    References:
        https://github.com/wzmsltw/BSN-boundary-sensitive-network/blob/7b101fc5978802aa3c95ba5779eb54151c6173c6/Post_processing.py#L42
    """
    if len(predictions) == 1:  # only has one prediction, no need for nms
        return predictions

    predictions = sorted(predictions, key=lambda x: x[2], reverse=True)  # descending order

    tstart = [e[0] for e in predictions]
    tend = [e[1] for e in predictions]
    tscore = [e[2] for e in predictions]
    rstart = []
    rend = []
    rscore = []
    while len(tstart) > 1 and len(rscore) < max_after_nms:  # max 100 after nms
        idx = 1
        while idx < len(tstart):  # compare with every prediction in the list.
            if compute_temporal_iou([tstart[0], tend[0]], [tstart[idx], tend[idx]]) > nms_thd:
                # rm highly overlapped lower score entries.
                tstart.pop(idx)
                tend.pop(idx)
                tscore.pop(idx)
                # print("--------------------------------")
                # print(compute_temporal_iou([tstart[0], tend[0]], [tstart[idx], tend[idx]]))
                # print([tstart[0], tend[0]], [tstart[idx], tend[idx]])
                # print(tstart.pop(idx), tend.pop(idx), tscore.pop(idx))
            else:
                # move to next
                idx += 1
        rstart.append(tstart.pop(0))
        rend.append(tend.pop(0))
        rscore.append(tscore.pop(0))

    if len(rscore) < max_after_nms and len(tstart) >= 1:  # add the last, possibly empty.
        rstart.append(tstart.pop(0))
        rend.append(tend.pop(0))
        rscore.append(tscore.pop(0))

    predictions_after_nms = [[st, ed, s] for s, st, ed in zip(rscore, rstart, rend)]
    return predictions_after_nms

def load_data(data_path):
        #Load MAD dataset
        with open(data_path, 'r') as f:
            datalist = json.load(f)
        for k,v in datalist.items(): v["sentence_id"] = k
        return datalist

def temporal_iou(spans1, spans2):
    """
    Args:
        spans1: (N, 2) torch.Tensor, each row defines a span [st, ed]
        spans2: (M, 2) torch.Tensor, ...

    Returns:
        iou: (N, M) torch.Tensor
        union: (N, M) torch.Tensor
    >>> test_spans1 = torch.Tensor([[0, 0.2], [0.5, 1.0]])
    >>> test_spans2 = torch.Tensor([[0, 0.3], [0., 1.0]])
    >>> temporal_iou(test_spans1, test_spans2)
    (tensor([[0.6667, 0.2000],
         [0.0000, 0.5000]]),
     tensor([[0.3000, 1.0000],
             [0.8000, 1.0000]]))
    """
    areas1 = spans1[:, 1] - spans1[:, 0]  # (N, )
    areas2 = spans2[:, 1] - spans2[:, 0]  # (M, )

    left = torch.max(spans1[:, None, 0], spans2[:, 0])  # (N, M)
    right = torch.min(spans1[:, None, 1], spans2[:, 1])  # (N, M)

    inter = (right - left).clamp(min=0)  # (N, M)
    union = areas1[:, None] + areas2 - inter  # (N, M)

    iou = inter / union
    return iou, union



def generalized_temporal_iou(spans1, spans2):
    """
    Generalized IoU from https://giou.stanford.edu/
    Also reference to DETR implementation of generalized_box_iou
    https://github.com/facebookresearch/detr/blob/master/util/box_ops.py#L40

    Args:
        spans1: (N, 2) torch.Tensor, each row defines a span in xx format [st, ed]
        spans2: (M, 2) torch.Tensor, ...

    Returns:
        giou: (N, M) torch.Tensor

    >>> test_spans1 = torch.Tensor([[0, 0.2], [0.5, 1.0]])
    >>> test_spans2 = torch.Tensor([[0, 0.3], [0., 1.0]])
    >>> generalized_temporal_iou(test_spans1, test_spans2)
    tensor([[ 0.6667,  0.2000],
        [-0.2000,  0.5000]])
    """
    spans1 = spans1.float()
    spans2 = spans2.float()
    assert (spans1[:, 1] >= spans1[:, 0]).all()
    assert (spans2[:, 1] >= spans2[:, 0]).all()
    iou, union = temporal_iou(spans1, spans2)

    left = torch.min(spans1[:, None, 0], spans2[:, 0])  # (N, M)
    right = torch.max(spans1[:, None, 1], spans2[:, 1])  # (N, M)
    enclosing_area = (right - left).clamp(min=0)  # (N, M)

    return iou - (enclosing_area - union) / enclosing_area

zero_shot = h5py.File('/mnt/storage8T/MQ/predictions/VLGNet_predictions_test.h5', 'r')
proposals= h5py.File('/mnt/storage8T/MQ/predictions/proposals_test.h5', 'r')
v_feats = h5py.File('/mnt/storage8T/MQ/Multimodal/features/CLIP_frames_features_5fps.h5','r')
data = load_data("/mnt/storage8T/MQ/Multimodal/annotations/MAD_test.json")
qids = list(zero_shot.keys())

print(":::::::Load Filter predictions::::::::::::")
folder_filter = "/mnt/storage8T/MQ/audio_txt_video/MomentRetv/baselines/guidance/original_cls_audio_txt_video_pos_neg_64/test"
pickle_filter = os.path.join(folder_filter, "pos_neg_logits_test_original_epoch_100.pickle")
with open(pickle_filter, 'rb') as file:
    filter_preds=pickle.load(file)

filter_pred_dict = dict()
for item in tqdm(filter_preds):
    sente = item['qid']
    filter_pred_dict[sente] = item

movies = {data[a]['movie']:data[a]['movie_duration'] for a in data}
list_m = list(movies.keys())

overlap = h5py.File('overlap_vlg_0shot_64vl_32stride_test.h5', 'r')
mr_res = h5py.File('./test/vlg_64_test_noNMS.h5', 'w')

for qid in tqdm(qids):
    scores_0shot = zero_shot[qid][:]
    filt_preds = filter_pred_dict[qid]
    movie = filt_preds['vid']
    filt_score =  filt_preds['score']
    indx_filter = overlap[movie][:] 
    prop = proposals[movie][:]
    if len(indx_filter) <  len(scores_0shot):
        scores_0shot = scores_0shot[0:len(prop)]
    if len(indx_filter) > len(scores_0shot):
        dif = len(indx_filter) - len(scores_0shot)
        pads = np.zeros(dif)
        scores_0shot = np.hstack([scores_0shot,pads])
    scores_0shot_nw =filt_score[indx_filter] * scores_0shot
    mr_res.create_dataset(qid, data=np.hstack((prop,scores_0shot_nw[:,None])))
