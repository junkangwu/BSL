#!/bin/bash 
# 单边测试
dataset="${11}"
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
cd ..
if [[ $gnn = "mf_simplex" ]]
then
        gnn_name="MF"
elif [[ $gnn = "lgn_frame" ]]
then
        gnn_name="LGN_frame"
elif [[ $gnn = "mf_output" ]]
then
        gnn_name="MF_output"
elif [[ $gnn = "mf_pos2" ]]
then
        gnn_name="mf_posDRO"
elif [[ $gnn = "mf_frame" ]]
then
        gnn_name="mf_frame"
else
        echo "NO GNN"
        exit 1
fi

if [[ $loss_fn = "PairwiseLogisticDROLoss" ]]
then 
        loss_name="DRO1"
elif [[ $loss_fn = "PairwiseLogisticDRO2Loss" ]]
then
        loss_name="DRO2"
elif [[ $loss_fn = "PairwiseLogisticDRO3Loss" ]]
then
        loss_name="DRO3"
elif [[ $loss_fn = "PairwiseLogisticEasyLoss" ]]
then
        loss_name="Easy"
elif [[ $loss_fn = "RinceLoss" ]]
then
        loss_name="RINCE"
elif [[ $loss_fn = "DCL_Loss" ]]
then
        loss_name="DCL"
elif [[ $loss_fn = "InforNCEPosDROLoss" ]]
then
        loss_name="POSDRO"
elif [[ $loss_fn = "Pos_DROLoss" ]]
then
        loss_name="DRO_POS"
else
        echo "NO loss"
        exit 1
fi

if [[ $drop_bool = "drop" ]]
then
        if [[ $loss_re = "False" ]]
        then
                echo "start to drop embedding"
                echo "loss_re is ${loss_re}"
                echo $generate_mode
                name1="ICDE_rebuttal_${dataset}_${loss_name}_${gnn_name}_t1_${t_1}_t2_${t_2}_t3_${t_3}_HOPS_${context_hops}_${bsz}_${n_negs}_lr_${lr}_l2_${l2}_DROP_${loss_re}_MODE_${pos_mode}_p_${t_patience}_G_${generate_mode}"
                # name1="yelp2018_YY_loss_Easy_else_t1_0.11_MF_output_512_800_lr_1e-4_l2_1e-3_DROP_13_05_2022_01:27:07"
                echo $name1
                CUDA_VISIBLE_DEVICES=$gpus python main.py --name $name1 --dataset $dataset --gnn $gnn --dim 64 --lr $lr \
                                        --batch_size $bsz --gpu_id 0 --logdir $logdir \
                                        --n_negs  $n_negs --l2 $l2 --mess_dropout True --mess_dropout_rate 0.1  \
                                        --sampling_method $sampling_method --loss_fn $loss_fn\
                                        --generate_mode $generate_mode --u_norm --i_norm \
                                        --temperature $t_1  --temperature_2 $t_2 --temperature_3 $t_3\
                                        --trans_mode $trans_mode --pos_mode $pos_mode --context_hops $context_hops --t_patience $t_patience\
                                        > ./outputs/${name1}.log
        else
                echo "start to drop embedding"
                echo "loss_re is ${loss_re}"
                name1="ICDE_rebuttal_${dataset}_${loss_name}_${gnn_name}_t1_${t_1}_t2_${t_2}_t3_${t_3}_HOPS_${context_hops}_${bsz}_${n_negs}_lr_${lr}_l2_${l2}_DROP_${loss_re}_MODE_${pos_mode}"
                # name1="yelp2018_YY_loss_Easy_else_t1_0.11_MF_output_512_800_lr_1e-4_l2_1e-3_DROP_13_05_2022_01:27:07"
                echo $name1
                CUDA_VISIBLE_DEVICES=$gpus python main.py --name $name1 --dataset $dataset --gnn $gnn --dim 64 --lr $lr \
                                        --batch_size $bsz --gpu_id 0 --logdir $logdir \
                                        --n_negs  $n_negs --l2 $l2 --mess_dropout True --mess_dropout_rate 0.1  \
                                        --sampling_method $sampling_method --loss_fn $loss_fn\
                                        --generate_mode cosine --u_norm --i_norm \
                                        --temperature $t_1  --temperature_2 $t_2 --temperature_3 $t_3 \
                                        --trans_mode $trans_mode --loss_re --pos_mode $pos_mode --context_hops $context_hops\
                                        > ./outputs/${name1}.log 
        fi
        # done
else
        echo "NOT equal"
        echo "Do not drop embedding"
        echo "loss_re is ${loss_re}"
        for t_2 in $t_2 1.00
        do 
        name1="ICDE_rebuttal_${dataset}_${loss_name}_${gnn_name}_t1_${t_1}_t2_${t_2}_t3_${t_3}_HOPS_${context_hops}_${bsz}_${n_negs}_lr_${lr}_l2_${l2}_NODROP_${loss_re}_MODE_${pos_mode}_P_${t_patience}_${sampling_method}_G_${generate_mode}"
        echo $name1
        CUDA_VISIBLE_DEVICES=$gpus python main.py --name $name1 --dataset $dataset --gnn $gnn --dim 64 --lr $lr \
                                --batch_size $bsz --gpu_id 0 --logdir $logdir \
                                --n_negs  $n_negs --l2 $l2\
                                --sampling_method $sampling_method --loss_fn $loss_fn\
                                --generate_mode $generate_mode --u_norm --i_norm \
                                --temperature $t_1  --temperature_2 $t_2 --temperature_3 $t_3 \
                                --trans_mode $trans_mode --pos_mode $pos_mode --context_hops $context_hops --t_patience $t_patience\
                                > ./outputs/${name1}.log
        done
fi

