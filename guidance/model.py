# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from guidance.span_utils import generalized_temporal_iou, span_cxw_to_xx

from guidance.matcher import build_matcher
from guidance.transformer import build_transformer
from guidance.position_encoding import build_position_encoding
from guidance.misc import accuracy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from einops import repeat

class Guidance(nn.Module):

    def __init__(self, transformer, position_embed, txt_position_embed, audio_position_embed, cls_position_embedding, txt_dim, vid_dim, audio_dim,
                 num_queries, input_dropout, aux_loss=False,
                 contrastive_align_loss=False, contrastive_hdim=64, clf_dim = 1,
                 max_v_l=75, span_loss_type="l1", use_txt_pos=False, n_input_proj=2):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        self.audio_position_embed = audio_position_embed
        self.cls_position_embed = cls_position_embedding 
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        # self.foreground_thd = foreground_thd
        # self.background_thd = background_thd
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.CLS = nn.Parameter(torch.randn(1,hidden_dim)) 
        self.pos_neg_proj = MLP(hidden_dim, hidden_dim,hidden_dim, 5)
        self.pos_neg_nn = nn.Linear(hidden_dim, 1)

        #project text
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        self.input_audio_proj = nn.Sequential(*[
            LinearLayer(audio_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_query = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_txt = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_vid = nn.Linear(hidden_dim, contrastive_hdim)

        #self.saliency_proj = nn.Linear(hidden_dim, 1)
        self.aux_loss = aux_loss

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, src_audio, src_audio_mask):
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        src_audio = self.input_audio_proj(src_audio)

        B,_,_ = src_vid.shape
        #CLS =  self.CLS
        CLS_tokens = repeat(self.CLS, 'n d -> b n d', b = B)
        src = torch.cat([CLS_tokens,src_vid, src_txt,src_audio], dim=1)  # (bsz, L_vid+L_txt, d)
        cls_mask = torch.ones([B,1]).to(src.device)
        mask = torch.cat([cls_mask,src_vid_mask, src_txt_mask,src_audio_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
        # TODO should we remove or use different positional embeddings to the src_txt?
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        #pos_cls = self.position_embed(CLS_tokens,cls_mask)
        #import ipdb;ipdb.set_trace()
        pos_cls = self.cls_position_embed(CLS_tokens)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        pos_audio = self.audio_position_embed(src_audio)
        # pos_txt = torch.zeros_like(src_txt)
        # pad zeros for txt positions
        pos = torch.cat([pos_cls,pos_vid, pos_txt,pos_audio], dim=1)
        # (#layers, bsz, #queries, d), (bsz, L_vid+L_txt, d)
        memory,cls_out = self.transformer(src, ~mask, self.query_embed.weight, pos)
        proj_cls =  self.pos_neg_proj(cls_out)
        pos_neg_logits = self.pos_neg_nn(proj_cls) 

        out = {}
        out['logits_neg_pos'] = pos_neg_logits

        return out


class SetCriterion(nn.Module):

    def __init__(self, matcher, weight_dict, eos_coef, losses, temperature, span_loss_type, max_v_l,
                 saliency_margin=1):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.temperature = temperature
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)

    def loss_pos_neg(self, outputs, targets, indices,pos_neg,log=True):
        assert 'logits_neg_pos' in outputs
        logits_pos_neg = outputs['logits_neg_pos']
        targets = targets["pos_neg_labels"]
        tgt_pos_neg = torch.cat([t['pos_neg'] for t in targets], dim=0)
        #pos_weight = torch.tensor([7/3]).cuda()
        loss_bce = F.binary_cross_entropy_with_logits(logits_pos_neg,tgt_pos_neg.unsqueeze(1),reduction='none')
        losses = {}
        losses['loss_pos_neg'] = loss_bce.mean()
        if log:
            scores = torch.sigmoid(logits_pos_neg.view(-1)).cpu()
            preds = (scores > 0.5).float().numpy()
            acc = accuracy_score(tgt_pos_neg.cpu().numpy(),preds)
            f1 = f1_score(tgt_pos_neg.cpu().numpy(),preds)
            precision = precision_score(tgt_pos_neg.cpu().numpy(),preds)
            recall = recall_score(tgt_pos_neg.cpu().numpy(),preds)
            #preds = (scores > 0.5).float()
            #acc = ( (scores > 0.5).float() == tgt_pos_neg).float().sum() / tgt_pos_neg.shape[0]
            losses['accucary'] = acc
            losses['f1_score'] = f1
            losses['precision'] = precision
            losses['recall'] = recall
        return losses
    def get_loss(self, loss, outputs, targets, indices, pos_neg, **kwargs):
        loss_map = {
            "pos_neg": self.loss_pos_neg
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, pos_neg, **kwargs)

    def forward(self, outputs, targets, pos_neg):
        indices = None
        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)
        #indices = self.matcher(outputs_without_aux, targets)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices,pos_neg))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/moment_detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding, audio_position_embedding, cls_position_embedding = build_position_encoding(args)

    model = Guidance(
        transformer,
        position_embedding,
        txt_position_embedding,
        audio_position_embedding,
        cls_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        audio_dim=args.a_feat_dim,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        aux_loss=args.aux_loss,
        contrastive_align_loss=args.contrastive_align_loss,
        contrastive_hdim=args.contrastive_hdim,
        span_loss_type=args.span_loss_type,
        use_txt_pos=args.use_txt_pos,
        n_input_proj=args.n_input_proj,
    )

    matcher = build_matcher(args)
    weight_dict = {#"loss_span": args.span_loss_coef,
                   #"loss_giou": args.giou_loss_coef,
                   #"loss_label": args.label_loss_coef,
                   "loss_pos_neg":args.pos_neg_loss_coef
                   # "loss_saliency": args.lw_saliency
                   }
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['pos_neg']
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]
    criterion = SetCriterion(
        matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, temperature=args.temperature,
        span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin
    )
    criterion.to(device)
    return model, criterion
