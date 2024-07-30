#!/bin/bash
python train.py \
--task nc \
--dataset airport  \
--model kHGCN \
--manifold PoincareBall \
--lr 0.01 \
--dim 16 \
--num-layers 4 \
--act relu \
--bias 1 \
--dropout 0 \
--weight-decay 0 \
--log-freq 100 \
--patience 500 \
--cuda 0 \
--agg_type curv \
--beta 0.5
