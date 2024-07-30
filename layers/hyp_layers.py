"""Hyperbolic layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import numpy as np
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax
from torch_geometric.nn.inits import glorot
# from layers.att_layers import SpGraphAttentionLayer
from torch_scatter import scatter, scatter_add
from utils.data_utils import load_data
from layers.att_layers import DenseAtt, SpGraphAttentionLayer, HSpAttLayer
# from layers.agg_curv_layers import HypAggCurvAtt
from layers.CurvConv import HypAggCurvAtt
from layers.att_pyg_layers import PYGAtt
from torch_geometric.nn.conv import MessagePassing
# from layers.hypatt_layers import HypAggAttdense
from geoopt import PoincareBall
import os


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


# data = load_data(args, os.path.join('./data/', args.dataset))


def get_dim_act_curv(args):
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
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias,
                 heads, concat=False, agg_type=''):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output


class CurvHGCN(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias,
                 heads=1, concat=False, agg_type='attcurv', position='origin'):
        super(CurvHGCN, self).__init__()
        out_features = heads * out_features
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        if agg_type == 'attcurv' or agg_type == 'curv':
            self.agg = HypAggCurvAtt(manifold, out_features, dropout, c_in, agg_type, heads=heads, concat=concat)
        if agg_type == 'att':
            self.agg = PYGAtt(manifold, out_features, dropout, c_in, original='att', heads=heads, concat=concat)
            # self.agg = HypAggAtt(manifold, c_in, out_features, dropout)
            # self.agg = PyGAtt(out_features, dropout, manifold, c_in, heads=1)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

        # self.ball = PoincareBall(c=c_in)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)

        h = self.hyp_act.forward(h)
        output = h, adj
        return output


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)

        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)

        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, dropout):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.dropout = dropout

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)

        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAggAtt(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, out_features, dropout, concat=None, heads=None):
        super(HypAggAtt, self).__init__()
        self.manifold = manifold
        self.c = c
        self.att = DenseAtt(out_features, dropout)
        print('using dense attention')
        # self.att = HSpAttLayer(out_features, dropout=dropout)
        # print('using spares attention')
        self.dropout = dropout

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        support_t = self.att(x_tangent, adj)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAggAttsparse(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, concat, heads):
        super(HypAggAttsparse, self).__init__()
        self.manifold = manifold
        self.c = c
        self.alpha = 0.2
        self.nheads = heads
        self.dropout = dropout
        self.act = torch.nn.LeakyReLU()
        self.concat = concat
        self.in_features = in_features
        self.attentions = [SpGraphAttentionLayer(in_features,
                                                 in_features,
                                                 dropout=dropout,
                                                 alpha=self.alpha,
                                                 activation=self.act) for _ in range(self.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        # =====================================================================
        if self.concat:
            # h = torch.cat([att(x_tangent, adj) for att in self.attentions], dim=1)
            h = torch.cat([att(x_tangent, adj) for att in self.attentions], dim=1)
        else:
            h_cat = torch.cat([att(x_tangent, adj).view((-1, self.in_features, 1)) for att in self.attentions], dim=2)
            h = torch.mean(h_cat, dim=2)
        # =====================================================================
        output = self.manifold.proj(self.manifold.expmap0(h, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

        # xt = self.act(self.manifold1.logmap0(x))
        # xt = self.manifold.proj_tan0(xt, c=self.c_out)
        # return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )


class HypAggPyG(MessagePassing):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, dim, dropout):
        super(HypAggPyG, self).__init__()
        self.manifold = manifold
        self.c = c
        self.node_dim = 0
        self.dropout = dropout
        self.cached = True
        self.cached_result = None
        self.cached_num_edges = None
        self.normalize = True

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, 1, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, adj=None, edge_weight=None):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        # ================================================
        edge_index = adj._indices()
        edge_index, norm = self.norm(edge_index, x_tangent.size(0))
        edge_i = edge_index[0]
        edge_j = edge_index[1]
        x_j = torch.nn.functional.embedding(edge_j, x_tangent)
        support = norm.view(-1, 1) * x_j
        result = scatter(support, edge_i, dim=0, dim_size=x.size(0), reduce='sum')
        # ===============================================

        output = self.manifold.proj(self.manifold.expmap0(result, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)
