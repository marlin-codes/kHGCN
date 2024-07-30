#!/bin/bash
# citeseer: 96.63
python train_lp.py \
--task lp \
--dataset citeseer \
--model kHGCN \
--lr 0.01 \
--dim 16 \
--num-layers 1 \
--act relu \
--bias 1 \
--c None \
--dropout 0.6 \
--weight-decay 5e-4 \
--manifold PoincareBall \
--log-freq 100 \
--patience 200 \
--agg_type curv \
--cuda 0
