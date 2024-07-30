"""Base model class."""
import os.path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx

from layers.layers import FermiDiracDecoder
import layers.hyp_layers as hyp_layers
import manifolds as manifolds
import models.encoders as encoders
from models.decoders import model2decoder
from utils.eval_utils import acc_f1, curv_acc
from utils.negative_sampling import negative_sampling


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:
            self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.edge_false = None
        self.filtered_edges = None
        self.args = args

    def lp_decode(self, h, edge_index):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        edge_i = edge_index[0]
        edge_j = edge_index[1]
        x_i = torch.nn.functional.embedding(edge_i, h)
        x_j = torch.nn.functional.embedding(edge_j, h)
        sqdist = self.manifold.sqdist(x_i, x_j, self.c)
        probs = self.dc.forward(sqdist)
        return probs

    def filter_leaf(self, edges_true, orc, threshold, dataset):
        if self.filtered_edges is not None:
            return self.filtered_edges
        edge_i = list(edges_true[0].cpu().numpy())
        edge_j = list(edges_true[1].cpu().numpy())

        DG = networkx.Graph()
        for edge in edges_true.transpose(1, 0).cpu().numpy():
            if edge[0] != edge[1]:
                DG.add_edge(edge[0], edge[1])

        filtered = []
        for ix in range(edges_true.size(1)):
            node_i = edge_i[ix]
            node_j = edge_j[ix]

            if DG.has_node(node_i) and DG.has_node(node_j):  # Non-isolated nodes
                if DG.degree(node_i) == 1 or DG.degree(node_j) == 1:  # If it is a leaf node, it will not be considered.
                    filtered.append(False)
                else:
                    if orc[ix] > threshold:
                        filtered.append(True)  # If not a leaf node and the value of orc satisfies the condition
                    else:  # Filter out nodes with smaller orc values
                        filtered.append(False)
            else:  # Isolated nodes
                filtered.append(False)
        self.filtered_edges = filtered_edges = edges_true[:, filtered]
        return filtered_edges

    def compute_metrics_orc(self, embeddings, data, split, **kargs):
        assert self.args.agg_type in ['curv', 'attcurv']
        orc = data['adj_train_norm'][1].squeeze()
        threshold = kargs['threshold']
        dataset = kargs['dataset']
        edges_true = data['edges_true']
        filtered_edges = self.filter_leaf(edges_true, orc, threshold, dataset)

        if filtered_edges.size(1) == 0:
            return {'loss': 0}

        if not self.edge_false:
            self.edges_false = negative_sampling(edges_true, num_nodes=embeddings.shape[0],
                                                 num_neg_samples=filtered_edges.size(1))

        pos_scores = self.lp_decode(embeddings, filtered_edges)
        neg_scores = self.lp_decode(embeddings, self.edges_false)

        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))

        metrics = {'loss': loss}
        return metrics

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output[idx], dim=1)

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)

        # curv = curv_acc(output, data['labels'][idx], data['node_curvature'])

        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges

    def decode(self, h, idx):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        try:
            assert torch.max(sqdist) >= 0
        except:
            print(self.c)
            print(sqdist)
            print('==========')
            print(emb_in)
            print('=========')
            print(emb_out)
        probs = self.dc.forward(sqdist)
        return probs

    def tri_dist(self, h, idx):
        # emb_i = h[idx[:, 0], :]
        # emb_j = h[idx[:, 1], :]
        # emb_k = h[idx[:, 2], :]
        # ij_sqdist = self.manifold.sqdist(emb_i, emb_j, self.c)**0.5
        # ik_sqdist = self.manifold.sqdist(emb_i, emb_k, self.c)**0.5
        # kj_sqdist = self.manifold.sqdist(emb_j, emb_k, self.c)**0.5
        #
        # total_dist = 0
        # for m in range(h.shape[0]):
        #     for n in range(m+1, h.shape[0]):
        #         total_dist += self.manifold.sqdist(h[m], h[n], self.c) ** 0.5

        h = self.manifold.logmap0(h, self.c)

        # Step 1: Remove triangles with edges in multiple triangles
        edge_to_triangle = {}
        triangle_indices = []
        for t_idx, (i, j, k) in enumerate(idx):
            for e in [(i, j), (j, k), (k, i)]:
                if e in edge_to_triangle:
                    # this edge already appears in another triangle, remove this triangle
                    break
                else:
                    edge_to_triangle[e] = t_idx
            else:
                # this triangle doesn't share edges with any other triangle, keep it
                triangle_indices.append((i, j, k))
        triangle_indices = torch.tensor(triangle_indices)

        # Step 2: Compute distances for each triangle
        dists = torch.cdist(h, h)
        triangle_dists = []
        for i, j, k in triangle_indices:
            ij_dist = dists[i, j]
            jk_dist = dists[j, k]
            ki_dist = dists[k, i]
            triangle_dists.append((ij_dist, jk_dist, ki_dist))

        # Step 3: Compute total pairwise distance
        total_dist = dists.sum()

        # Step 4: Compute ratio of triangle distance to total distance
        triangle_sum = sum(sum(t_dists) for t_dists in triangle_dists)
        ratio = triangle_sum / total_dist
        return ratio * 100

    def compute_metrics(self, embeddings, data, split):

        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']

        edges_true = data[f'{split}_edges']
        edges_true = edges_true[edges_true[:, 0] != edges_true[:, 1]]
        edges_false = edges_false[edges_false[:, 0] != edges_false[:, 1]]

        pos_scores = self.decode(embeddings, edges_true)
        neg_scores = self.decode(embeddings, edges_false)
        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)

        for c in self.encoder.curvatures:
            c.data.clamp_(0.5, 2)

        # print(self.encoder.curvatures)
        tridist = self.tri_dist(embeddings, data['triangle']).cpu().item()
        metrics = {'loss': loss, 'roc': roc, 'ap': ap, 'tdist': tridist}
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])
