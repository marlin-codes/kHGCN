#!/bin/bash
# Acc Pubmed: 82.0
python train.py \
--task nc \
--dataset pubmed  \
--model kHGCN \
--manifold PoincareBall \
--lr 0.005 \
--dim 16 \
--num-layers 1 \
--act relu \
--bias 1 \
--dropout 0.6 \
--weight-decay 1e-4 \
--log-freq 50 \
--cuda 0 \
--c 1.0 \
--patience 200 \
--agg_type curv \
--repeat 1 \
--sample-mode II \
--edge_drop_ratio 0.01 \
--beta 1.0 \
--pretrained_embeddings ./embeddings/pubmed/HGCurvCN/lp/PoincareBall/embeddings.npy