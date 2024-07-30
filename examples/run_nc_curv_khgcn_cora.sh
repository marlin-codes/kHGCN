#!/bin/bash
# Acc Cora: 82.50
python train.py \
--task nc \
--dataset cora  \
--model kHGCN \
--manifold PoincareBall \
--lr 0.01 \
--dim 16 \
--num-layers 2 \
--act relu \
--bias 1 \
--dropout 0.6 \
--weight-decay 5e-4 \
--log-freq 50 \
--cuda 0 \
--c 1 \
--patience 200 \
--agg_type curv \
--repeat 1 \
--sample-mode II \
--edge_drop_ratio 0.05 \
--beta 3.0