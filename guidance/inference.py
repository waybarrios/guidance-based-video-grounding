import pprint
from tqdm import tqdm, trange
import numpy as np
import os
from collections import OrderedDict, defaultdict
from utils.basic_utils import AverageMeter

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from guidance.config import TestOptions
from guidance.model import build_model
from guidance.span_utils import span_cxw_to_xx
from guidance.start_end_dataset import StartEndDataset, start_end_collate, prepare_batch_inputs,start_end_collate_test
from guidance.postprocessing_moment_detr import PostProcessorDETR
from standalone_eval.eval import eval_submission
from utils.basic_utils import save_jsonl, save_json
from utils.temporal_nms import temporal_nms
from sklearn.metrics import classification_report
import pickle
import logging
import h5py

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def post_processing_mr_nms(mr_res, nms_thd, max_before_nms, max_after_nms):
    mr_res_after_nms = []
    for e in mr_res:
        e["pred_relevant_windows"] = temporal_nms(
            e["pred_relevant_windows"][:max_before_nms],
            nms_thd=nms_thd,
            max_after_nms=max_after_nms
        )
        mr_res_after_nms.append(e)
    return mr_res_after_nms


def eval_epoch_post_processing(submission, opt, gt_data, save_submission_filename):
    # IOU_THDS = (0.5, 0.7)
    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    save_jsonl(submission, submission_path)

    if opt.eval_split_name in ["val", "test"]:  # since test_public has no GT
        metrics = eval_submission(
            submission, gt_data,
            verbose=opt.debug, match_number=False
        )
        save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
        latest_file_paths = [submission_path, save_metrics_path]
    else:
        metrics = None
        latest_file_paths = [submission_path, ]

    if opt.nms_thd != -1:
        logger.info("[MR] Performing nms with nms_thd {}".format(opt.nms_thd))
        submission_after_nms = post_processing_mr_nms(
            submission, nms_thd=opt.nms_thd,
            max_before_nms=opt.max_before_nms, max_after_nms=opt.max_after_nms
        )

        logger.info("Saving/Evaluating nms results")
        submission_nms_path = submission_path.replace(".jsonl", "_nms_thd_{}.jsonl".format(opt.nms_thd))
        save_jsonl(submission_after_nms, submission_nms_path)
        if opt.eval_split_name == "val":
            metrics_nms = eval_submission(
                submission_after_nms, gt_data,
                verbose=opt.debug, match_number=not opt.debug
            )
            save_metrics_nms_path = submission_nms_path.replace(".jsonl", "_metrics.json")
            save_json(metrics_nms, save_metrics_nms_path, save_pretty=True, sort_keys=False)
            latest_file_paths += [submission_nms_path, save_metrics_nms_path]
        else:
            metrics_nms = None
            latest_file_paths = [submission_nms_path, ]
    else:
        metrics_nms = None
    return metrics, metrics_nms, latest_file_paths


@torch.no_grad()
def compute_mr_results(model, eval_loader, opt, epoch_i=None, criterion=None, tb_writer=None):
    model.eval()
    if criterion:
        assert eval_loader.dataset.load_labels
        criterion.eval()

    loss_meters = defaultdict(AverageMeter)
    write_tb = tb_writer is not None and epoch_i is not None

    mr_res = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory,eval_window=opt.eval_window)
        if opt.eval_window:
            outputs = model(**model_inputs)
            probs = torch.sigmoid(outputs["logits_neg_pos"])  # (batch_size, #queries, #classes=2)
        else:
            #untrimmed eval (entire video)
            outputs = [model(**chunk_input) for chunk_input in model_inputs]
            probs = [torch.sigmoid(out['logits_neg_pos']) for out in outputs]
        if opt.eval_window:
            # compose predictions
            for idx, (meta,score) in enumerate(zip(query_meta, probs.cpu())):
                st = meta['window_movie'][0][0]/eval_loader.dataset.FPS
                et = meta['window_movie'][0][1]/eval_loader.dataset.FPS
                cur_query_pred = dict(
                    qid=meta["sentence_id"],
                    windows=[st,et],
                    vid=meta["movie"],
                    score = score,
                    gt=meta['ext_timestamps'],
                )
                mr_res.append(cur_query_pred)

        else:
            list_scores = list()
            for idx,scr in enumerate(probs):
                for k in range(len(scr)):
                    list_scores.append(scr[k].cpu())

            scores_total = torch.cat(list_scores,dim=0)
            windows = query_meta['chunks_movie']
            windows_flat = [item for sublist in windows for item in sublist]

            cur_query_pred = dict(
                qid=query_meta["sentence_id"],
                windows=windows_flat,
                vid=query_meta["movie"],
                score= scores_total,
                gt=query_meta['ext_timestamps'],
            )
            mr_res.append(cur_query_pred)


        if opt.debug:
            break
    sentences_ids = [item['qid'] for item in mr_res] 
    FPS = eval_loader.dataset.FPS
    overlap = eval_loader.dataset.getOverlap
    if opt.eval_window:
        preds = [item['score'] for item in mr_res]
        preds_cat = torch.cat(preds) 
        tgt = torch.ones(preds_cat.shape[0])
    
    else:
        gts_fr =[(np.array(item['gt'])*FPS).astype('int') for item in mr_res]
        preds = [item['score'] for item in mr_res]
        preds_cat= torch.cat(preds)
        #total = np.array([scores[k].shape[0] for k in range(len(mr_res))]).sum()
        tgt = [torch.tensor([ 0 if overlap(gts_fr[k],window)==0 else 1 for window in mr_res[k]['windows']]) for k in range(len(mr_res))]
        tgt = torch.cat(tgt) 

    pred_label =  (preds_cat > 0.5).float()
    acc = ( pred_label== tgt).float().sum() / tgt.shape[0]
    print(f"Accuracy: {acc}")
    print(f"Dataset: {opt.eval_path}")
    path_logits = os.path.join(opt.results_dir,"pos_neg_logits_test_clean.pickle")
    
    with open(path_logits,'wb') as handle:
            pickle.dump(mr_res, handle, protocol=pickle.HIGHEST_PROTOCOL)


    y_true = tgt.clone().numpy()
    y_pred =  pred_label.clone().numpy()
    target_names = ['class 0', 'class 1']
    import ipdb;ipdb.set_trace()
    print(classification_report(y_true, y_pred, target_names=target_names))
    return mr_res,None


def get_eval_res(model, eval_loader, opt, epoch_i, criterion, tb_writer):
    """compute and save query and video proposal embeddings"""
    eval_res, eval_loss_meters = compute_mr_results(model, eval_loader, opt, epoch_i, criterion, tb_writer)  # list(dict)
    return eval_res, eval_loss_meters


def eval_epoch(model, eval_dataset, opt, save_submission_filename, epoch_i=None, criterion=None, tb_writer=None):
    logger.info("Generate submissions")
    model.eval()
    if criterion is not None and eval_dataset.load_labels:
        criterion.eval()
    else:
        criterion = None

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate if opt.eval_window else start_end_collate_test,
        batch_size=opt.eval_bsz if opt.eval_window else 1,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory
    )

    submission, eval_loss_meters = get_eval_res(model, eval_loader, opt, epoch_i, criterion, tb_writer)
    
    return metrics, metrics_nms, eval_loss_meters, latest_file_paths


def setup_model(opt):
    """setup model/optimizer/scheduler and load checkpoints when needed"""
    logger.info("setup model/optimizer/scheduler")
    model, criterion = build_model(opt)
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        criterion.to(opt.device)

    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)

    if opt.resume is not None:
        logger.info(f"Load checkpoint from {opt.resume}")
        checkpoint = torch.load(opt.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if opt.resume_all:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            opt.start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Loaded model saved at epoch {checkpoint['epoch']} from checkpoint: {opt.resume}")
    else:
        logger.warning("If you intend to evaluate the model, please specify --resume with ckpt path")

    return model, criterion, optimizer, lr_scheduler


def start_inference():
    logger.info("Setup config, data and model...")
    opt = TestOptions().parse()
    cudnn.benchmark = True
    cudnn.deterministic = False

    assert opt.eval_path is not None
    eval_dataset = StartEndDataset(
        dset_name=opt.dset_name,
        data_path=opt.eval_path,
        v_feat_path=opt.v_feat_path,
        q_feat_path=opt.t_feat_path,
        a_feat_path=opt.a_feat_path,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        load_labels=True,  # opt.eval_split_name == "val",
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0,
        batch_per_movie=opt.batch_per_movie,
        split = opt.eval_split_name,
        eval_window=opt.eval_window
    )

    model, criterion, _, _ = setup_model(opt)
    save_submission_filename = "inference_{}_{}_{}_preds.jsonl".format(
        opt.dset_name, opt.eval_split_name, opt.eval_id)
    logger.info("Starting inference...")
    with torch.no_grad():
        metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
            eval_epoch(model, eval_dataset, opt, save_submission_filename, criterion=criterion)
    


if __name__ == '__main__':
    start_inference()

