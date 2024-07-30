## 1. Overview

PyTorch Implementation for ["kHGCN: Tree-likeness Modeling via Continuous and Discrete Curvature Learning (KDD2023)"](https://arxiv.org/abs/2212.01793)

Authors: Menglin Yang (CUHK), Min Zhou (Huawei), Lujia Pan (Huawei), Irwin King (CUHK) 

Paper: [https://arxiv.org/abs/2212.01793](https://arxiv.org/abs/2212.01793)

Code: [https://github.com/marlin-codes/khgcn](https://github.com/marlin-codes/khgcn)

### Abstract

The prevalence of tree-like structures, encompassing hierarchical structures and power law distributions, exists extensively in real-world applications, including recommendation systems, ecosystems, financial networks, social networks, etc. Recently, the exploitation of hyperbolic space for tree-likeness modeling has garnered considerable attention owing to its exponential growth volume. Compared to the flat Euclidean space, the curved hyperbolic space provides a more amenable and embeddable room, especially for datasets exhibiting implicit tree-like architectures. However, the intricate nature of real-world tree-like data presents a considerable challenge, as it frequently displays a heterogeneous composition of tree-like, flat, and circular regions. The direct embedding of such heterogeneous structures into a homogeneous embedding space (i.e., hyperbolic space) inevitably leads to heavy distortions. To mitigate the aforementioned shortage, this study endeavors to explore the curvature between discrete structure and continuous learning space, aiming at encoding the message conveyed by the network topology in the learning process, thereby improving tree-likeness modeling. To the end, a curvature-aware hyperbolic graph convolutional neural network, \{kappa}HGCN, is proposed, which utilizes the curvature to guide message passing and improve long-range propagation. Extensive experiments on node classification and link prediction tasks verify the superiority of the proposal as it consistently outperforms various competitive models by a large margin.


## 2. Experiments

### 2.1 Environment
`pip install -r requirements.txt`

### 2.2 Dataset
The dataset is in `data/` folder.

### 2.3 Examples
```bash
bash run_lp_curv_khgcn_cora.sh
bash run_lp_curv_khgcn_disease.sh
bash run_lp_curv_khgcn_airport.sh
bash run_nc_curv_khgcn_cora.sh
bash run_nc_curv_khgcn_disease_nc.sh
bash run_nc_curv_khgcn_airport.sh
```
Note: in node classification task, accurary is for Cora, Citeseer, and PubMed, F1-score is for Airport and Disease

## 3.References
The code is heavily based on the following projects
- [https://github.com/HazyResearch/hgcn/tree/master](https://github.com/HazyResearch/hgcn/tree/master)
- [https://github.com/BUPT-GAMMA/lgcn_torch](https://github.com/BUPT-GAMMA/lgcn_torch)
- [https://github.com/CheriseZhu/GIL](https://github.com/CheriseZhu/GIL)

Thanks for the above awesome projects and authors!

## 4.Citations

```bibtex
@inproceedings{yang2023kappahgcn,
  title={$\kappa$hgcn: Tree-likeness modeling via continuous and discrete curvature learning},
  author={Yang, Menglin and Zhou, Min and Pan, Lujia and King, Irwin},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={2965--2977},
  year={2023}
}
```

