#!/bin/bash
# airport: 96.29
python train_lp.py \
--task lp \
--dataset airport \
--model kHGCN \
--lr 0.01 \
--dim 16 \
--num-layers 1 \
--act relu \
--bias 1 \
--dropout 0 \
--weight-decay 0 \
--c 1.0 \
--manifold PoincareBall \
--log-freq 1 \
--patience 500 \
--cuda 0 \
--agg_type curv \
--use-feats 1

