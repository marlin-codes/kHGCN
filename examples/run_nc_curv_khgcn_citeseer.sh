#!/bin/bash
# citeseer: 73.80
python train.py \
--task nc \
--dataset citeseer  \
--model kHGCN \
--manifold PoincareBall \
--lr 0.01 \
--dim 16 \
--num-layers 2 \
--act relu \
--bias 1 \
--dropout 0.6 \
--weight-decay 5e-4 \
--cuda 0 \
--agg_type curv \
--patience 200 \
--log-freq 100 \
--sample-mode II \
--edge_drop_ratio 0.01 \
--beta 1.0