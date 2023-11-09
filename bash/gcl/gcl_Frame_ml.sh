#!/bin/bash 
# 单边测试
dataset="${1}"
lr="$2"
l2="$3"
logdir="./log_pos/"
gnn="lightgcl_frame_bsl"

n_negs="800"
bsz="$4"
t_1="$5"
gpus="$6"
trans_mode="else"
drop_bool_rate="${7}"
sampling_method="uniform_gpu"
loss_re="False"
context_hops="${8}"
t_patience="${9}"
w1="${10}"
t_2="${11}"
t_3="${12}"
generate_mode="${13}"
cd ..
cd ..
echo "start to drop embedding"
name1="ICDE_rebuttal_${dataset}_${gnn}_t1_${t_1}_T_${t_2}_${t_3}_lambda_${w1}_HOPS_${context_hops}_${bsz}_${n_negs}_lr_${lr}_l2_${l2}_DROP_${drop_bool_rate}_p_${t_patience}_${sampling_method}_${generate_mode}"
echo $name1
CUDA_VISIBLE_DEVICES=$gpus python main.py --name $name1 --dataset $dataset --gnn $gnn --dim 64 --lr $lr \
                        --batch_size $bsz --gpu_id 0 --logdir $logdir \
                        --n_negs  $n_negs --l2 $l2 --mess_dropout_rate $drop_bool_rate  \
                        --sampling_method $sampling_method \
                        --temperature $t_1  --w1 $w1 --temperature_2 $t_2 --temperature_3 $t_3 --generate_mode $generate_mode\
                        --trans_mode $trans_mode --context_hops $context_hops --t_patience $t_patience \
                        > ./outputs/${name1}.log

