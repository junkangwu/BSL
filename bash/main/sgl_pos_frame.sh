#!/bin/bash 
# 单边测试
cd ..
cd ..
dataset="$5"
lr="$1"
l2="$2"
# logdir="./log_DRO/GD/"
logdir="./log_pos/"
gnn="sgl_frame"
n_negs="1"
bsz="$3" # 2048

gpus="$4"
sampling_method="neg"
# loss_fn="PairwiseLogisticDROLoss"
t_patience="10"
# SGL params
context_hops="$6" # hop
pos_mode="$7" # aug mode
w1="$8" # drop_ratio
w2="$9" # ssl_weight
temperature="${10}"
name1="ICDE_rebuttal_${dataset}_${gnn}_${bsz}_lr_${lr}_l2_${l2}_HOP_${context_hops}_T_${temperature}_AUG_${pos_mode}_drop_${w1}_ssl_weight_${w2}"
# name1="yelp2018_YY_loss_Easy_else_t1_0.11_MF_output_512_800_lr_1e-4_l2_1e-3_DROP_13_05_2022_01:27:07"
echo $name1
CUDA_VISIBLE_DEVICES=$gpus python main.py --name $name1 --dataset $dataset --gnn $gnn --dim 64 --lr $lr \
                        --batch_size $bsz --gpu_id 0 --logdir $logdir \
                        --n_negs  $n_negs --l2 $l2 \
                        --sampling_method $sampling_method \
                        --t_patience $t_patience \
                        --context_hops $context_hops --temperature $temperature --pos_mode $pos_mode --w1 $w1 --w2 $w2 \
                        > ./outputs/${name1}.log