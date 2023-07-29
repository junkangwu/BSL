#!/bin/bash 
# 单边测试
dataset="${11}"
lr="$1"
l2="$2"
# logdir="./log_DRO/GD/"
logdir="./log_pos/"
gnn="lgn_frame"

n_negs="$3"
bsz="$4"
# n_negs_input="$6"
t_1="$5"
t_2="$6"
t_3="${t_1}"
# pos_num="$4"
gpus="$7"
# bsz="$4"
pos_mode="$8"
trans_mode="else"
drop_bool="${9}"
sampling_method="${15}"
# loss_fn="PairwiseLogisticDROLoss"
loss_re="False"
loss_fn="${10}"
context_hops="${12}"
t_patience="${13}"
generate_mode="${14}"
cd ..

if [[ $drop_bool = "drop" ]]
then
        echo "start to drop embedding"
        echo "loss_re is ${loss_re}"
        echo $generate_mode
        name1="${dataset}_${loss_fn}_${gnn}_t1_${t_1}_t2_${t_2}_t3_${t_3}_HOPS_${context_hops}_${bsz}_${n_negs}_lr_${lr}_l2_${l2}_DROP_${loss_re}_MODE_${pos_mode}_p_${t_patience}_${sampling_method}_G_${generate_mode}"
        echo $name1
        CUDA_VISIBLE_DEVICES=$gpus python main.py --name $name1 --dataset $dataset --gnn $gnn --dim 64 --lr $lr \
                                --batch_size $bsz --gpu_id 0 --logdir $logdir \
                                --n_negs  $n_negs --l2 $l2 --mess_dropout True --mess_dropout_rate 0.1  \
                                --n_negs  $n_negs --l2 $l2\
                                --sampling_method $sampling_method --loss_fn $loss_fn\
                                --generate_mode $generate_mode --u_norm --i_norm \
                                --temperature $t_1  --temperature_2 $t_2 --temperature_3 $t_3 \
                                --trans_mode $trans_mode --pos_mode $pos_mode --context_hops $context_hops --t_patience $t_patience\
                                > ./outputs/${name1}.log
else
        echo "NOT equal"
        echo "Do not drop embedding"
        echo "loss_re is ${loss_re}"
        name1="${dataset}_${loss_fn}_${gnn}_t1_${t_1}_t2_${t_2}_t3_${t_3}_HOPS_${context_hops}_${bsz}_${n_negs}_lr_${lr}_l2_${l2}_NODROP_${loss_re}_MODE_${pos_mode}_P_${t_patience}_${sampling_method}_G_${generate_mode}"
        # name1="yelp2018_YY_loss_Easy_else_t1_0.11_MF_output_512_800_lr_1e-4_l2_1e-3_DROP_13_05_2022_01:27:07"
        echo $name1
        CUDA_VISIBLE_DEVICES=$gpus python main.py --name $name1 --dataset $dataset --gnn $gnn --dim 64 --lr $lr \
                                --batch_size $bsz --gpu_id 0 --logdir $logdir \
                                --n_negs  $n_negs --l2 $l2\
                                --sampling_method $sampling_method --loss_fn $loss_fn\
                                --generate_mode $generate_mode --u_norm --i_norm \
                                --temperature $t_1  --temperature_2 $t_2 --temperature_3 $t_3 \
                                --trans_mode $trans_mode --pos_mode $pos_mode --context_hops $context_hops --t_patience $t_patience\
                                > ./outputs/${name1}.log
fi


