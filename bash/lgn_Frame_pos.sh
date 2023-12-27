#!/bin/bash 
# 单边测试
dataset="${10}"
lr="$1"
l2="$2"
# logdir="./log_DRO/GD/"
logdir="./log_pos/"
gnn="lgn_frame"

n_negs="$4"
bsz="$3"
# n_negs_input="$6"
t_1="$5"
t_2="$6"
t_3="${t_1}"
# pos_num="$4"
gpus="$7"
# bsz="$4"
pos_mode="$8"
drop_bool="${9}"
sampling_method="${14}"
context_hops="${11}"
t_patience="${12}"
generate_mode="${13}"
cd ..

if (( $(echo "$t_2 == 1.0" | bc -l) )); then
  loss_name="SL"
else
  loss_name="BSL"
fi

if [[ $drop_bool = "drop" ]]
then
        echo "start to drop embedding"
        echo $generate_mode
        name1="ICDE_rebuttal_${dataset}_${loss_name}_${gnn}_t1_${t_1}_t2_${t_2}_t3_${t_3}_HOPS_${context_hops}_${bsz}_${n_negs}_lr_${lr}_l2_${l2}_DROP_MODE_${pos_mode}_p_${t_patience}_G_${generate_mode}"
        echo $name1
        CUDA_VISIBLE_DEVICES=$gpus python main.py --name $name1 --dataset $dataset --gnn $gnn --dim 64 --lr $lr \
                                --batch_size $bsz --gpu_id 0 --logdir $logdir \
                                --n_negs  $n_negs --l2 $l2 --mess_dropout True --mess_dropout_rate 0.1  \
                                --sampling_method $sampling_method \
                                --generate_mode $generate_mode --u_norm --i_norm \
                                --temperature $t_1  --temperature_2 $t_2 --temperature_3 $t_3\
                                --pos_mode $pos_mode --context_hops $context_hops --t_patience $t_patience\
                                > ./outputs/${name1}.log
       
else
        echo "NOT equal"
        echo "Do not drop embedding"
        name1="ICDE_rebuttal_${dataset}_${loss_name}_${gnn}_t1_${t_1}_t2_${t_2}_t3_${t_3}_HOPS_${context_hops}_${bsz}_${n_negs}_lr_${lr}_l2_${l2}_NODROP_MODE_${pos_mode}_P_${t_patience}_${sampling_method}_G_${generate_mode}"
        echo $name1
        CUDA_VISIBLE_DEVICES=$gpus python main.py --name $name1 --dataset $dataset --gnn $gnn --dim 64 --lr $lr \
                                --batch_size $bsz --gpu_id 0 --logdir $logdir \
                                --n_negs  $n_negs --l2 $l2\
                                --sampling_method $sampling_method\
                                --generate_mode $generate_mode --u_norm --i_norm \
                                --temperature $t_1  --temperature_2 $t_2 --temperature_3 $t_3 \
                                --pos_mode $pos_mode --context_hops $context_hops --t_patience $t_patience\
                                > ./outputs/${name1}.log
fi


