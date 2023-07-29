## Prerequisites
- Python 3.9
- PyTorch 1.11.0

## Training & Evaluation
Usage (example):
```bash
cd bash
bash lgn_Frame.sh $lr $l2 $n_negs $bsz $t1 $t2 $GPU_ID $loss_mode $drop $loss_fn $DATASET_NAME $context_hops $patience $sampling_method
```

```$GPU_ID``` refers to the ID of the launched GPU, while ```$lr, $l2, and $n_negs``` represent the learning rate, decay, and number of negative sampling, respectively. ```$DATASET_NAME``` is the name of the dataset (e.g. yelp2018), and ```$context_hops``` indicates the number of layers (1, 2, or 3). ```$patience``` and ```$sampling_method``` refer to the patience for early stopping and the sampling method, respectively. Finally, ```$loss_fn``` denotes the loss function used.

Commands for reproducing the reported results:

```bash
### Yelp2018
bash lgn_Frame.sh 1e-3 1e-5 1024 4096 0.15 1.12 0 reweight nodrop Pos_DROLoss yelp2018 3 50 no_cosine no_sample
```

## Documentation
Thanks to their simple forms, these losses are implemented in just a few lines of code in [`utils/losses.py.py`](utils/losses.py.py#L429-L432):
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
