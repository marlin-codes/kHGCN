"""Euclidean layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax
from torch_scatter import scatter, scatter_add
from torch_geometric.nn import SAGEConv, SGConv
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from config import parser
import networkx as nx

args = parser.parse_args()


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def get_dim_act(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
    return dims, acts


class GraphConvolution(Module):
    """
    Simple GCN layer.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input):
        x, adj = input
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        if adj.is_sparse:
            support = torch.spmm(adj, hidden)
        else:
            support = torch.mm(adj, hidden)
        output = self.act(support), adj
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
            self.in_features, self.out_features
        )


class GraphSAGE(Module):
    """
    Simple GCN layer.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(GraphSAGE, self).__init__()
        self.act = act
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.layer = SGConv(self.in_features, self.out_features, K=2, bias=use_bias)

    def forward(self, input):
        x, adj = input
        support = self.layer(x, adj._indices())
        support = F.dropout(support, p=self.dropout, training=self.training)
        output = self.act(support), adj
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
            self.in_features, self.out_features
        )


class SGCConvolution(Module):
    """
    Simple GCN layer.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(SGCConvolution, self).__init__()
        self.act = act
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.layer = SAGEConv(self.in_features, self.out_features, normalize=False, bias=use_bias)

    def forward(self, input):
        x, adj = input
        support = self.layer(x, adj._indices())
        support = F.dropout(support, p=self.dropout, training=self.training)
        output = self.act(support), adj
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
            self.in_features, self.out_features
        )


class GraphCurvConvolution(Module):
    """
    Simple GCN layer.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(GraphCurvConvolution, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act
        self.in_features = in_features
        self.out_features = out_features

        self.curvature = None
        self.edge_index = None
        self.mlp = nn.Sequential(*[
            nn.Linear(1, out_features, bias=1.0),
            nn.LeakyReLU(0.2, True),
            nn.Linear(out_features, out_features, bias=1.0),
        ])

    def forward(self, input):
        x, adj = input
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)

        if self.curvature is None:
            self.edge_index, curvature = compute_curvature(adj._indices())
            self.curvature = curvature.detach()
            self.edge_index = self.edge_index.to(x.device)
            self.curvature = self.curvature.to(x.device)

        edge_i = self.edge_index[0]
        edge_j = self.edge_index[1]
        norm_curv = self.mlp(self.curvature)
        norm_curv = F.dropout(norm_curv, 0.2)
        # 注意下面的式子，邻居节点为index j
        norm_curv = softmax(norm_curv, edge_i, len(torch.unique(self.edge_index)))  # 按照i聚合，计算邻居j的分布
        x_j = torch.nn.functional.embedding(edge_j, hidden) * norm_curv  # 根据j的分布和j的表示计算节点的表示
        support_t_curv = scatter(x_j, edge_i, dim=0, reduce="sum")  # 按照i聚合，计算邻居的和

        output = self.act(support_t_curv), adj
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
            self.in_features, self.out_features
        )


class Linear(Module):
    """
    Simple Linear layer with dropout.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act

    def forward(self, x):
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        out = self.act(hidden)
        return out


class FermiDiracDecoder(Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs


def compute_curvature(edge_index, edge_weight=None, dtype=None):
    mkdirs('./curvature/')
    if args.task == 'nc':
        curvature_cached_file = './curvature/{}_curv.pt'.format(args.dataset)
    else:
        curvature_cached_file = './curvature/{}_curv_lp.pt'.format(args.dataset)
    if os.path.isfile(curvature_cached_file):
        print('>> loading curvature file ...')
        return torch.load(curvature_cached_file)
    print('>> computing curvature ...')
    edge_index_no_loop, _ = remove_self_loops(edge_index)
    edge_index = edge_index_no_loop
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
    weight = edge_weight.cpu().unsqueeze(1).numpy()
    links = edge_index.cpu().numpy().T
    G = nx.Graph()
    weighted_links = np.concatenate([links, weight], axis=1)
    G.add_weighted_edges_from(weighted_links)
    orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
    orc.compute_ricci_curvature()
    curvature_list = []
    for (i, j) in links:
        curvature_list.append(orc.G[i][j]["ricciCurvature"])
    curvature = torch.from_numpy(np.array(curvature_list)).to(edge_index.device)
    edge_index, edge_weight = add_remaining_self_loops(edge_index_no_loop)
    self_loop_curvature = torch.zeros(edge_index.shape[1] - curvature.shape[0]).to(edge_index.device)

    curvature = torch.cat([curvature.float(), self_loop_curvature.float()]).unsqueeze(1)
    print('saved curvature to file ...')
    torch.save([edge_index, curvature], curvature_cached_file)
    return edge_index, curvature
