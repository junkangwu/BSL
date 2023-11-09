#!/bin/bash 
# 单边测试
cd ..
cd ..
dataset="$5"
lr="$1"
l2="$2"
# logdir="./log_DRO/GD/"
logdir="./log_pos/"
gnn="simsgl_frame"
n_negs="1"
bsz="$3" # 2048

gpus="$4"
sampling_method="neg"
# loss_fn="PairwiseLogisticDROLoss"
t_patience="50"
# SGL params
context_hops="$6" # hop
w1="$7" # cl_rate
w2="$8" # eps
dim="$9"
name1="ICDE_rebuttal_${dataset}_${gnn}_${bsz}_lr_${lr}_l2_${l2}_HOP_${context_hops}_cl_rate_${w1}_eps_${w2}_DIM_${dim}"
# name1="yelp2018_YY_loss_Easy_else_t1_0.11_MF_output_512_800_lr_1e-4_l2_1e-3_DROP_13_05_2022_01:27:07"
echo $name1
CUDA_VISIBLE_DEVICES=$gpus  python main.py --name $name1 --dataset $dataset --gnn $gnn --lr $lr \
                        --batch_size $bsz --gpu_id 0 --logdir $logdir \
                        --n_negs  $n_negs --l2 $l2 \
                        --sampling_method $sampling_method --dim $dim \
                        --t_patience $t_patience \
                        --context_hops $context_hops --w1 $w1 --w2 $w2 \
                        > ./outputs/${name1}.log 