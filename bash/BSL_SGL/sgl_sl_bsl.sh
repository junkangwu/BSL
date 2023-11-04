#!/bin/bash 
# 单边测试
cd ..
cd ..
dataset="$5"
lr="$1"
l2="$2"
# logdir="./log_DRO/GD/"
logdir="./log_pos/"
gnn="sgl_frame_bsl"
n_negs="${13}"
bsz="$3" # 2048

gpus="$4"
sampling_method="${14}"
# loss_fn="PairwiseLogisticDROLoss"
t_patience="50"
# SGL params
context_hops="$6" # hop
pos_mode="$7" # aug mode
w1="$8" # drop_ratio
w2="$9" # ssl_weight
temperature="${10}"
temperature_2="${11}"
temperature_3="${12}"
generate_mode="${15}"
num_epoch="${16}"
add_speed="${17}"
for ((i=1; i<=$num_epoch; i++))
do
    name1="ICDE_check_${dataset}_${gnn}_${bsz}_lr_${lr}_l2_${l2}_HOP_${context_hops}_T_${temperature}_t_${temperature_2}_${temperature_3}_AUG_ED_${pos_mode}_drop_${w1}_ssl_weight_${w2}_${sampling_method}_${n_negs}_${generate_mode}"
    echo $name1
    CUDA_VISIBLE_DEVICES=$gpus python main.py --name $name1 --dataset $dataset --gnn $gnn --dim 64 --lr $lr \
                            --batch_size $bsz --gpu_id 0 --logdir $logdir \
                            --n_negs  $n_negs --l2 $l2 \
                            --sampling_method $sampling_method --generate_mode $generate_mode \
                            --t_patience $t_patience --temperature_2 $temperature_2 --temperature_3 $temperature_3\
                            --context_hops $context_hops --temperature $temperature --pos_mode $pos_mode --w1 $w1 --w2 $w2 \
                            > ./outputs/${name1}.log
    temperature_3=`echo $temperature_3  $add_speed | awk '{ print $1+$2}'`
done