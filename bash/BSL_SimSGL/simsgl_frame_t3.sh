#!/bin/bash 
# 单边测试
cd ..
cd ..
dataset="$5"
lr="$1"
l2="$2"
# logdir="./log_DRO/GD/"
logdir="./log_pos/"
gnn="simsgl_frame_bsl"
bsz="$3" # 2048

gpus="$4"
# loss_fn="PairwiseLogisticDROLoss"
t_patience="50"
# SGL params
context_hops="$6" # hop
pos_mode="$7" # aug mode
w1="$8" # cl_rate
w2="$9" # eps
temperature="${10}"
temperature_2="${11}"
temperature_3="${12}"
n_negs="${13}"
sampling_method="${14}"
generate_mode="${15}"

for temperature_3 in 0.80 0.85 0.90 0.95 1.00 1.05 1.10
do 
    name1="${dataset}_${gnn}_Rebuttal_temp_${bsz}_lr_${lr}_l2_${l2}_HOP_${context_hops}_T_${temperature}_t_${temperature_2}_${temperature_3}_cl_rate_${w1}_eps_${w2}_${sampling_method}_${n_negs}_${generate_mode}"
    echo $name1
    CUDA_VISIBLE_DEVICES=$gpus python main.py --name $name1 --dataset $dataset --gnn $gnn --dim 64 --lr $lr \
                            --batch_size $bsz --gpu_id 0 --logdir $logdir \
                            --n_negs  $n_negs --l2 $l2 \
                            --sampling_method $sampling_method --generate_mode $generate_mode \
                            --t_patience $t_patience --temperature_2 $temperature_2 --temperature_3 $temperature_3\
                            --context_hops $context_hops --temperature $temperature --pos_mode $pos_mode --w1 $w1 --w2 $w2 \
                            > ./outputs/${name1}.log
done

