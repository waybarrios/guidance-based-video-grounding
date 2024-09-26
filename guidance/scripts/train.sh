dset_name=MAD
ctx_mode=video_tef
results_root=query_dependent_visual_txt_mad
exp_id=exp

######## data paths
train_path="/jumbo/jinlab/datasets/MAD/annotations/MAD_train_clean.json"
eval_path="/jumbo/jinlab/datasets/MAD/annotations/MAD_val_clean.json"
eval_split_name=val

######## setup video+text features
feat_root=features

# video features
v_feat_path="/jumbo/jinlab/datasets/MAD/features/CLIP_frames_features_5fps.h5"
v_feat_dim=512


# text features
t_feat_path="/jumbo/jinlab/datasets/MAD/features/CLIP_language_tokens_features.h5"
t_feat_dim=512

#audio features
a_feat_path="/jumbo/jinlab/datasets/MAD/features/MAD_audio_openl3_feats.h5"
a_feat_dim=512


#### training
bsz=512
neg_prob=0.5
batch_per_movie=256
enc_layers=6
lr=1e-4
wd=1e-4
hidden_dim=256
max_v_l=64
PYTHONPATH=$PYTHONPATH:. python guidance/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_path ${v_feat_path} \
--v_feat_dim ${v_feat_dim} \
--t_feat_path ${t_feat_path} \
--t_feat_dim ${t_feat_dim} \
--a_feat_path ${a_feat_path} \
--a_feat_dim ${a_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--neg_prob ${neg_prob} \
--batch_per_movie ${batch_per_movie} \
--enc_layers ${enc_layers}   \
--lr ${lr} \
--wd ${wd} \
--hidden_dim ${hidden_dim} \
--max_v_l ${max_v_l} \
--eval_window  \
${@:1}
