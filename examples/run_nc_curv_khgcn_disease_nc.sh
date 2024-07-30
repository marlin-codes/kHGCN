#!/bin/bash
python train.py \
--task nc \
--dataset disease_nc  \
--model kHGCN \
--dim 16 \
--lr 0.01 \
--num-layers 4 \
--act relu \
--bias 1 \
--dropout 0 \
--weight-decay 0 \
--manifold PoincareBall \
--log-freq 100 \
--c 1.0 \
--cuda 0 \
--patience 500 \
--agg_type curv