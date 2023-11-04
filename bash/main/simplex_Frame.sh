#!/bin/bash 
# 单边测试
lr="$1"
l2="$2"
logdir="./log_pos/"
gnn="$3"
n_negs="$4"
bsz="$5"
gpus="$6"
dataset="$7"
t_1="$8"
t_2="${9}"
sampling_method="${10}"
loss_fn="CosineContrastiveLoss"
t_patience="50"
cd ..
cd ..
name1="ICDE_rebuttal_${dataset}_SimpleX_lr_${lr}_l2_${l2}_batch_${bsz}_neg_${n_negs}_${loss_fn}_margin_${t_1}_neg_weight_${t_2}_${sampling_method}"
echo $name1
CUDA_VISIBLE_DEVICES=$gpus python main.py --name $name1 --dataset $dataset --gnn $gnn --dim 64 --lr $lr --l2 $l2\
                                        --batch_size $bsz --gpu_id 0 --logdir $logdir --n_negs  $n_negs  \
                                        --generate_mode cosine --u_norm --i_norm \
                                        --sampling_method $sampling_method --loss_fn $loss_fn \
                                        --temperature $t_1  --temperature_2 $t_2 --t_patience $t_patience\
                                        > ./outputs/${name1}.log