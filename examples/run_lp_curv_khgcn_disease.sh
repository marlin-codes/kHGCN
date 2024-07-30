#!/bin/bash
# disease: 95.25
python train_lp.py \
--task lp \
--dataset disease_lp \
--model kHGCN \
--lr 0.01 \
--dim 16 \
--num-layers 1 \
--act relu \
--bias 1 \
--dropout 0 \
--normalize-feats 0 \
--use-feats 1 \
--weight-decay 0 \
--manifold PoincareBall \
--agg_type curv \
--patience 500 \
--c None \
--cuda 0
