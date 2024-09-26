ckpt_path=$1
eval_split_name=$2
eval_path="/home/alberto/workspace/datasets/pmad/data_mad/annotations/MAD_val.json"

# video features
v_feat_path="/home/alberto/workspace/datasets/pmad/data_mad/features/CLIP_frames_features_5fps.h5"
v_feat_dim=512


# text features
t_feat_path="/home/alberto/workspace/datasets/pmad/data_mad/features/CLIP_language_tokens_features.h5"
t_feat_dim=512

#audio features
a_feat_path="/mnt/storage8T/MQ/Multimodal/features/MAD_audio_openl3_feats.h5"
a_feat_dim=512

#batch - clips per forward pass
batch_per_movie=1024

PYTHONPATH=$PYTHONPATH:. python moment_detr/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
--v_feat_path ${v_feat_path} \
--v_feat_dim ${v_feat_dim} \
--t_feat_path ${t_feat_path} \
--t_feat_dim ${t_feat_dim} \
--a_feat_path ${a_feat_path} \
--a_feat_dim ${a_feat_dim} \
--batch_per_movie ${batch_per_movie} \
#--eval_window  \
${@:3}
