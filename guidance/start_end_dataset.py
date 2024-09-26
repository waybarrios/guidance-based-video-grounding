import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import random
import logging
from os.path import join, exists
from utils.basic_utils import load_jsonl, l2_normalize_np_array
from utils.tensor_utils import pad_sequences_1d
from guidance.span_utils import span_xx_to_cxw
import h5py
import json
from data.utils import movie2feats,query2feats


logger = logging.getLogger(__name__)


class StartEndDataset(Dataset):
    Q_FEAT_TYPES = ["pooler_output", "last_hidden_state"]
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }
    """

    def __init__(self, dset_name, data_path, v_feat_path, q_feat_path,a_feat_path,
                 q_feat_type="last_hidden_state",
                 max_q_l=50, max_v_l=256, data_ratio=1.0, ctx_mode="video",eval_window=True,
                 normalize_v=True, normalize_t=True, load_labels=True,test_stride = 64,batch_per_movie = 256,
                 clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0,FPS=5,split="train",neg_prob=0.7):
        self.dset_name = dset_name
        self.data_path = data_path
        self.data_ratio = data_ratio
        self.v_feat_path = v_feat_path 
        self.q_feat_path = q_feat_path
        self.a_feat_path = a_feat_path
        self.q_feat_type = q_feat_type
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        self.load_labels = load_labels
        self.clip_len = clip_len
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.span_loss_type = span_loss_type
        self.txt_drop_ratio = txt_drop_ratio
        self.FPS = FPS #FPS for feature extraction
        self.split = split # dataset split
        self.batch_per_movie = batch_per_movie
        self.eval_window = eval_window
        self.neg_prob = neg_prob
        self.test_stride = int(self.max_v_l / 2)

        # checks
        assert q_feat_type in self.Q_FEAT_TYPES

        # data
        self.data = self.load_data()
        self.keys = list(self.data.keys())

        #features
        self.movies = {self.data[a]['movie']:self.data[a]['movie_duration'] for a in self.data}
        self.v_feats = movie2feats(v_feat_path, list(self.movies.keys()))
        self.q_feats = query2feats(q_feat_path,self.keys)
        self.a_feats = movie2feats(a_feat_path, list(self.movies.keys()))

        if "val" == self.split or "test" == self.split:
            assert txt_drop_ratio == 0
            if not self.eval_window:
                self._compute_windows_per_movie(self.test_stride)
    def getOverlap(self,a, b):
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))
    def load_data(self):
        #Load MAD dataset
        with open(self.data_path, 'r') as f:
            datalist = json.load(f)
        for k,v in datalist.items(): v["sentence_id"] = k
        return datalist

    def __len__(self):
        return len(self.data)

    def get_sentence(self, idx):
        return self.data[idx]['sentence']
    
    def chunks(self,l, n,movie):
        #l = d[movie]
        n = max(1, n)
        return [l[i:i+n] for i in range(0, len(l), n)]

    def __getitem__(self, index):
        sentence_id = self.keys[index]
        meta = self.data[sentence_id]
        meta['sentence_id'] = sentence_id 
        meta['clip_duration'] = self.max_v_l / self.FPS
        model_inputs = dict()
        model_inputs["query_feat"] = self._get_query_feat_by_qid(sentence_id)  # (Dq, ) or (Lq, Dq)
        movie = meta['movie']

        if self.split == "train":
            #positive example
            if random.random() > self.neg_prob:
                positive = 1
                window_features,label,window,span_labels = self._get_feats_labels_train(meta['movie'],
                                                                                    meta['ext_timestamps'],
                                                                                    meta['clip_duration'])
                meta['relevant_windows'] = label #relative anno within window
                meta['window_movie'] = window
                if self.load_labels:
                    model_inputs['span_labels'] = span_labels
                model_inputs['video_feat'] = window_features
                model_inputs['audio_feat'] = self._get_audio_feats(window[0],movie)
                model_inputs['pos_neg_labels'] = torch.ones(1)
                meta['sample_type'] = positive
            else:
                #negative example goes here
                negative = 0
                while True:
                    start_window = random.randint(0, self.v_feats[movie].shape[0]-self.max_v_l)
                    stop_window = start_window + self.max_v_l

                    if not stop_window <= meta['movie_duration']*self.FPS:
                        stop_window = int(meta['movie_duration']*self.FPS)
                        start_window = stop_window -  self.max_v_l
                    
                    st,et = meta['ext_timestamps']
                    st = int(st * self.FPS)
                    et = int(et * self.FPS)
                    #make sure ground truth isnt within window
                    if self.getOverlap([st,et], [start_window,stop_window]) == 0: 
                        break
                if self.use_video:
                    vid_feat = torch.from_numpy(l2_normalize_np_array(self.v_feats[movie][start_window:stop_window]))
                    ctx_l = len(vid_feat)
                else:
                    ctx_l = self.max_v_l

                if self.use_tef:
                    tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
                    tef_ed = tef_st + 1.0 / ctx_l
                    tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
                    if self.use_video:
                        feats = torch.cat([vid_feat, tef], dim=1) 
                    else:
                        feats = tef.clone().detach()

                random_time = round(random.uniform(0,self.max_v_l / self.FPS), 2)
                window_label =[[random_time,random_time]] #wrong label
                meta['relevant_windows'] = window_label #relative anno within window
                meta['window_movie'] = [[start_window,stop_window]]
                audio_feat = self._get_audio_feats([start_window,stop_window],movie)
                if self.load_labels:
                    model_inputs["span_labels"] = self.get_span_labels(window_label, meta['clip_duration'])
                model_inputs["video_feat"] = feats
                model_inputs['audio_feat'] = audio_feat
                meta['sample_type'] = negative
                model_inputs['pos_neg_labels'] = torch.zeros(1)
        else:
            #test or val set 

            #windowed evaluation
            if self.eval_window:
                #meta['ext_timestamps'] = [12.23,19.2]
                window_features,label,window,span_labels = self._get_feats_labels_train(meta['movie'],
                                                                                    meta['ext_timestamps'],
                                                                                    meta['clip_duration'])
                meta['relevant_windows'] = label #relative anno within window
                meta['window_movie'] = window
                if self.load_labels:
                    model_inputs["span_labels"] = span_labels
                model_inputs["video_feat"] = window_features
                model_inputs['audio_feat'] = self._get_audio_feats(window[0],movie)

            else: 
                #untrimmed eval
                windows = self.windows[movie]
                windows_chunks = self.chunks(windows,self.batch_per_movie,movie)
                meta['chunks_movie'] = windows_chunks
                feat_batchs_movie = []
                feat_batches_audio = []
                for chunk in windows_chunks:
                    batch = []
                    batch_a = []
                    for indexes in chunk:
                        audio_feat = self._get_audio_feats(indexes,movie)
                        if self.use_video:
                            vid_feat = torch.from_numpy(l2_normalize_np_array(self.v_feats[movie][indexes[0]:indexes[1]]))
                            ctx_l = len(vid_feat)
                        else:
                             ctx_l = self.max_v_l
                        if self.use_tef:
                            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
                            tef_ed = tef_st + 1.0 / ctx_l
                            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
                            if self.use_video:
                                feats = torch.cat([vid_feat, tef], dim=1)  # (Lv, Dv+2)
                            else:
                                feats = tef.clone().detach()
                        batch_a.append(audio_feat)
                        batch.append(feats)
                    feat_batchs_movie.append(batch)
                    feat_batches_audio.append(batch_a)
                model_inputs['video_feat'] = feat_batchs_movie
                model_inputs['audio_feat'] = feat_batches_audio

        return dict(meta=meta, model_inputs=model_inputs)

    def _get_audio_feats(self,window,movie):
        st,ed = window
        audio_feat = self.a_feats[movie][st:ed]
        assert len(audio_feat) == self.max_v_l
        audio_feat = l2_normalize_np_array(audio_feat)
        return torch.from_numpy(audio_feat)

    def _get_feats_labels_train(self,movie,timestamps,duration):
        span_labels = None
        if self.use_video:
            window_features, window_label, window = self._get_window_feats_train(movie,timestamps)
            ctx_l = len(window_features)
        else:
            ctx_l = self.max_v_l

        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
            if self.use_video:
                window_features = torch.cat(
                    [window_features, tef], dim=1)  # (Lv, Dv+2)
            else:
                window_features = tef

        if self.load_labels:
            span_labels = self.get_span_labels(window_label, duration)  
     
        return window_features,window_label,window,span_labels
   
    def get_span_labels(self, windows, ctx_l):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / (ctx_l)  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        #elif self.span_loss_type == "ce":
        #    windows = torch.Tensor([
        #        [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
        #        for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows

    def _get_query_feat_by_qid(self,idx):
        
        q_feat = self.q_feats[idx]
        if self.q_feat_type == "last_hidden_state":
            q_feat = q_feat[:self.max_q_l]
        if self.normalize_t:
            q_feat = l2_normalize_np_array(q_feat)
        if self.txt_drop_ratio > 0:
            q_feat = self.random_drop_rows(q_feat)

        q_feat = l2_normalize_np_array(q_feat) #always normalize 
        return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)


    def _compute_windows_per_movie(self, test_stride):
        '''
            INPUTS:
            anno: annotation data, contains all the preprocessed information
            movie: movie id to select the correct features

            OUTPUTS:
            feat: movie features
            iou2d: target matrix 
        '''

        self.windows = {}
        for m in self.movies.keys():
            num_feats = len(self.v_feats[m])
            starts = np.arange(0, num_feats - self.max_v_l, test_stride, dtype=int)
            stops = starts + self.max_v_l
            self.windows[m] = np.stack([starts,stops]).transpose(1,0)

    def random_drop_rows(self, embeddings):
        """randomly mask num_drop rows in embeddings to be zero.
        Args:
            embeddings: np.ndarray (L, D)
        """
        num_drop_rows = round(len(embeddings) * self.txt_drop_ratio)
        if num_drop_rows > 0:
            row_indices = np.random.choice(
                len(embeddings), size=num_drop_rows, replace=False)
            embeddings[row_indices] = 0
        return embeddings

    def _get_window_feats_train(self, movie,timestamps):

        _feat = self.v_feats[movie]

        # Moment start and end 
        start_ground = int(timestamps[0] * self.FPS) 
        end_ground = int(timestamps[1] * self.FPS)
        
        #moment length
        clip_len = end_ground - start_ground

        #if moment len < window len

        if clip_len < self.max_v_l:
            delta = self.max_v_l - clip_len
            offset = random.randint(0,delta)
            _window_start = max(start_ground - offset,0)
            _window_end =  end_ground  + (delta - offset)
            if _window_start == 0:
                _window_end = _window_start + self.max_v_l
            #if window is outside movie 
            if _window_end > len(_feat):
                _window_end = len(_feat)
                _window_start = _window_end - self.max_v_l
            #verifying window len    
            new_start_ground = start_ground - _window_start
            new_end_ground = end_ground - _window_start
            #verifying groundtruth len
            assert clip_len == new_end_ground - new_start_ground
        else:
                center = (start_ground + end_ground) /2
                offset = int(round(center / 2))
                _window_start = max(start_ground - offset, 0)
                _window_end   = _window_start + self.max_v_l
                new_start_ground,new_end_ground = 0 , self.max_v_l 

        assert _window_end - _window_start == self.max_v_l
        
        #relative annotation based on sliding window
        new_start_ground = new_start_ground / self.FPS
        new_end_ground = new_end_ground / self.FPS

        #moment within window
        relevant_windows = [[new_start_ground,new_end_ground]] #it's hack to keep same format

        #selecting features based on sliding window
        _feat = _feat[_window_start:_window_end] #(Lv, D)
        v_feat = l2_normalize_np_array(_feat)    #(Lv, D)

        return torch.from_numpy(v_feat),relevant_windows,[[_window_start,_window_end]]

def start_end_collate(batch):
    batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?

    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=e["model_inputs"]["span_labels"]) for e in batch]
            continue
        if k == "pos_neg_labels":
            batched_data[k] = [dict(pos_neg=e["model_inputs"]["pos_neg_labels"]) for e in batch]
            continue
        batched_data[k] = pad_sequences_1d(
            [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
    return batch_meta, batched_data

def start_end_collate_test(batch):
    batch_meta = batch[0]['meta'] #eval batch is only 1 per sentence. 
    model_inputs_keys = batch[0]["model_inputs"].keys()
    chunk_data = dict()
    for idx,vid_chunk in enumerate(batch[0]['model_inputs']['video_feat']):
        batched_data = dict()
        num_clips =  len(vid_chunk)
        batched_data['query_feat'] = pad_sequences_1d(
                [batch[0]['model_inputs']['query_feat'] for _ in range(num_clips)], dtype=torch.float32, fixed_length=None)
        batched_data['video_feat'] = pad_sequences_1d(vid_chunk, dtype=torch.float32, fixed_length=None)
        audio_chunk = batch[0]['model_inputs']['audio_feat'][idx]
        batched_data['audio_feat'] = pad_sequences_1d(audio_chunk, dtype=torch.float32, fixed_length=None)
        chunk_data[f"{idx}"] = batched_data 

    return batch_meta, chunk_data

def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False,eval_window=True):
    targets = {}
    if eval_window:
        model_inputs = dict(
            src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
            src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
            src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
            src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
            src_audio=batched_model_inputs["audio_feat"][0].to(device, non_blocking=non_blocking),
            src_audio_mask=batched_model_inputs["audio_feat"][1].to(device, non_blocking=non_blocking),
        )
        if "span_labels" in batched_model_inputs:
            targets["span_labels"] = [
                dict(spans=e["spans"].to(device, non_blocking=non_blocking))
                for e in batched_model_inputs["span_labels"]
            ]
    
        if "pos_neg_labels" in batched_model_inputs:
             targets["pos_neg_labels"] = [
                dict(pos_neg=e["pos_neg"].to(device, non_blocking=non_blocking))
                for e in batched_model_inputs["pos_neg_labels"]
            ]
        #if "saliency_pos_labels" in batched_model_inputs:
         #   for name in ["saliency_pos_labels", "saliency_neg_labels"]:
         #       targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)

    else:
        model_inputs = []
        for k in batched_model_inputs:
            batch_inputs = dict(
                src_txt=batched_model_inputs[k]["query_feat"][0].to(device, non_blocking=non_blocking),
                src_txt_mask=batched_model_inputs[k]["query_feat"][1].to(device, non_blocking=non_blocking),
                src_vid=batched_model_inputs[k]["video_feat"][0].to(device, non_blocking=non_blocking),
                src_vid_mask=batched_model_inputs[k]["video_feat"][1].to(device, non_blocking=non_blocking),
                src_audio=batched_model_inputs[k]["audio_feat"][0].to(device, non_blocking=non_blocking),
                src_audio_mask=batched_model_inputs[k]["audio_feat"][1].to(device, non_blocking=non_blocking),
            )
            model_inputs.append(batch_inputs)


    targets = None if len(targets) == 0 else targets

    return model_inputs, targets
