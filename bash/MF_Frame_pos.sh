#!/bin/bash 
# 单边测试
dataset="$9"
lr="$1"
l2="$2"
# logdir="./log_DRO/GD/"
logdir="./log_pos/"
gnn="mf_frame"

n_negs="$3"
bsz="$4"
# n_negs_input="$6"
t_1="$5"
t_2="$6"
t_3="1.0"
# pos_num="$4"
gpus="$7"
# bsz="$4"
pos_mode="$8"
trans_mode="else"
drop_bool="${10}"
sampling_method="uniform"

cd ..

if (( $(echo "$t_2 == 1.0" | bc -l) )); then
  loss_name="SL"
else
  loss_name="BSL"
fi


if [[ $drop_bool = "drop" ]]
then
        echo "start to drop embedding"
        name1="ICDE_rebuttal_${dataset}_${loss_name}_${gnn}_t1_${t_1}_t2_${t_2}_t3_${t_3}_${bsz}_${n_negs}_lr_${lr}_l2_${l2}_DROP_MODE_${pos_mode}"
        echo $name1
        CUDA_VISIBLE_DEVICES=$gpus python main.py --name $name1 --dataset $dataset --gnn $gnn --dim 64 --lr $lr \
                                --batch_size $bsz --gpu_id 0 --logdir $logdir \
                                --n_negs  $n_negs --l2 $l2 --mess_dropout True --mess_dropout_rate 0.1  \
                                --sampling_method $sampling_method \
                                --generate_mode cosine --u_norm --i_norm \
                                --temperature $t_1  --temperature_2 $t_2 --temperature_3 $t_3\
                                --pos_mode $pos_mode\
                                > ./outputs/${name1}.log

else
        echo "NOT equal"
        echo "Do not drop embedding"
        name1="ICDE_rebuttal_${dataset}_${loss_name}_${gnn}_t1_${t_1}_t2_${t_2}_t3_${t_3}_${bsz}_${n_negs}_lr_${lr}_l2_${l2}_NODROP_MODE_${pos_mode}"
        echo $name1
        CUDA_VISIBLE_DEVICES=$gpus python main.py --name $name1 --dataset $dataset --gnn $gnn --dim 64 --lr $lr \
                                --batch_size $bsz --gpu_id 0 --logdir $logdir \
                                --n_negs  $n_negs --l2 $l2\
                                --sampling_method $sampling_method \
                                --generate_mode cosine --u_norm --i_norm \
                                --temperature $t_1  --temperature_2 $t_2 --temperature_3 $t_3 \
                                --pos_mode $pos_mode\
                                > ./outputs/${name1}.log
fi


