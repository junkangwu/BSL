<h2 align="center">
BSL: Understanding and Improving Softmax Loss
for Recommendation
</h2>
<p align='center'>
<img src='https://github.com/junkangwu/ADNCE/blob/master/adnce.jpg?raw=true' width='500'/>
</p>
<div align="center">

[![](https://img.shields.io/badge/paper-pink?style=plastic&logo=GitBook)](https://arxiv.org/pdf/2312.12882.pdf)
[![](https://img.shields.io/badge/-github-green?style=plastic&logo=github)](https://github.com/junkangwu/BSL) 

</div>

This is the PyTorch implementation for our ICDE 2024 paper. 
> Junkang Wu, Jiawei Chen, Jiancan Wu, Wentao Shi, Jizhi Zhang, Xiang Wang 2024. BSL: Understanding and Improving Softmax Loss for Recommendation. [arxiv link](https://arxiv.org/pdf/2312.12882.pdf)

## Prerequisites
- Python 3.9
- PyTorch 1.11.0

## Training & Evaluation

Commands for reproducing the reported results:

### MF
Usage (example):
```bash
cd bash
bash lgn_Frame.sh $lr $l2 $n_negs $bsz $t1 $t2 $GPU_ID $loss_mode $DATASET_NAME $drop $loss_fn
```

```$GPU_ID``` refers to the ID of the launched GPU, while ```$lr, $l2, and $n_negs``` represent the learning rate, decay, and number of negative sampling, respectively. ```$DATASET_NAME``` is the name of the dataset (e.g. yelp2018). Finally, ```$loss_fn``` denotes the loss function used.

#### Yelp2018
```
# SL
bash MF_Frame_pos.sh 1e-4 1e-3 800 1024 0.11 1.00 0 reweight yelp2018 drop Pos_DROLoss
# BSL
bash MF_Frame_pos.sh 1e-4 1e-3 800 1024 0.11 1.10 0 multi yelp2018 drop Pos_DROLoss
```
#### Amazon
```
# SL
bash MF_Frame_pos.sh 5e-4 1e-3 1024 1024 0.14 1.00 1 reweight amazon drop Pos_DROLoss
# BSL
bash MF_Frame_pos.sh 5e-4 1e-3 1024 1024 0.14 1.32 1 reweight amazon drop Pos_DROLoss
```
#### Gowalla
```
# SL
bash MF_Frame_pos.sh 1e-4 1e-9 800 1024 0.08 1.00 0 reweight gowalla drop Pos_DROLoss
# BSL
bash MF_Frame_pos.sh 1e-4 1e-9 800 1024 0.08 1.22 0 multi gowalla drop Pos_DROLoss
```
#### Movielens-1M
```
# SL
bash MF_Frame_pos.sh 1e-4 1e-3 800 2048 0.17 1.00 0 reweight ml nodrop Pos_DROLoss
# BSL
bash MF_Frame_pos.sh 1e-4 1e-3 800 2048 0.17 1.06 0 multi ml nodrop Pos_DROLoss
```

### LightGCN
Usage (example):
```bash
cd bash
bash lgn_Frame_pos.sh $lr $l2 $n_negs $bsz $t1 $t2 $GPU_ID $loss_mode $drop $loss_fn $DATASET_NAME $context_hops $patience $generate_method $sampling_method
```

```$GPU_ID``` refers to the ID of the launched GPU, while ```$lr, $l2, and $n_negs``` represent the learning rate, decay, and number of negative sampling, respectively. ```$DATASET_NAME``` is the name of the dataset (e.g. yelp2018), and ```$context_hops``` indicates the number of layers (1, 2, or 3). ```$patience```, ```generate_method``` and ```$sampling_method``` refer to the patience for early stopping (50 as default ), prediction score (cosine similarity or inner product) and the sampling method (uniformly sampling and in batch sampling), respectively. Finally, ```$loss_fn``` denotes the loss function used and ```$loss_fn``` denotes the loss function used.

#### yelp2018
```
# SL
bash lgn_Frame_pos.sh 1e-3 1e-5 1024 1024 0.15 1.00 0 reweight nodrop Pos_DROLoss yelp2018 3 50 no_cosine no_sample
# BSL
bash lgn_Frame_pos.sh 1e-3 1e-5 1024 1024 0.15 1.12 0 reweight nodrop Pos_DROLoss yelp2018 3 50 no_cosine no_sample
```
#### Amazon
```
# SL
bash lgn_Frame_pos.sh 1e-3 1e-1 4096 800 0.30  1.00 0 reweight nodrop Pos_DROLoss amazon 3 50 no_cosine no_sample
# BSL
bash lgn_Frame_pos.sh 1e-3 1e-1 4096 800 0.30  0.80 0 reweight nodrop Pos_DROLoss amazon 3 50 no_cosine no_sample
```

#### Gowalla
```
# SL
bash lgn_Frame_pos.sh 1e-3 1e-5 1024 800 0.15  1.00 0 reweight nodrop Pos_DROLoss ml 1 50 cosine uniform
# BSL
bash lgn_Frame_pos.sh 1e-3 1e-5 1024 800 0.15  1.10 3 reweight nodrop Pos_DROLoss ml 1 50 cosine uniform
```
#### Movielens-1M
```
# SL
bash lgn_Frame_pos.sh 1e-3 1e-5 1024 800 0.15  1.00 0 reweight nodrop Pos_DROLoss ml 1 50 cosine uniform
# BSL
bash lgn_Frame_pos.sh 1e-3 1e-5 1024 800 0.15  1.10 0 reweight nodrop Pos_DROLoss ml 1 50 cosine uniform
```


## Documentation
Thanks to their simple forms, these losses are implemented in just a few lines of code in [`utils/losses.py.py`](utils/losses.py#L429-L432):
```py
# bsz : batch size (number of positive pairs)
# y_pred[:, 0]:  prediction score of postive samples, shape=[bsz]
# y_pred[:, 1:]: prediction score of negative samples, shape=[bsz, bsz-1]
# temperature: t1
# temperature_2: t2
pos_logits = torch.exp(y_pred[:, 0] / self.temperature)
neg_logits = torch.exp(y_pred[:, 1:] / self.temperature)
neg_logits = neg_logits.sum(dim=-1)
neg_logits = torch.pow(neg_logits, self.temperature_2)
loss = - torch.log(pos_logits / neg_logits).mean()
```

The [training log](./logs) is also provided. The results fluctuate slightly under different running environment.

For any clarification, comments, or suggestions please create an issue or contact me (jkwu0909@mail.ustc.edu.cn).