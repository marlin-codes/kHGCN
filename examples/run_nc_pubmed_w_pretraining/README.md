We follow the similar processing for pubmed in node classification task, that is
1. pretrain using structural information
2. save the above pretraining embeddings
3. using the pretraining embeddings as features
4. train a kHGCN model

```bash
python run_lp_curv_khgcn_pubmed_for_pretraining.sh # to get the pretraining embeddings
python run_nc_curv_khgcn_pubmed.sh # to get the trained model
```