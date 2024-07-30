"""Curvature based aggregation layers."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import ones, zeros, glorot
from torch_scatter import scatter
from layers.att_pyg_layers import PYGAtt
from geoopt import PoincareBall, Lorentz
import math


class HypAggCurvAtt(Module):
    def __init__(self, manifold, in_features, dropout, c, agg_type='curv', heads=1, concat=False):
        super(HypAggCurvAtt, self).__init__()
        self.manifold = manifold
        self.in_dim = in_features
        self.c = c
        self.poincare = PoincareBall(learnable=False)
        self.dropout = dropout
        self.heads = heads
        self.concat = concat
        self.act = torch.nn.LeakyReLU(0.2)
        self.edge_index = None
        self.sqdist = self.manifold.sqdist
        self.initAgg(agg_type)
        self.add_level_info = True

    def initAgg(self, agg_type):
        self.mlp = nn.Sequential(*[nn.Linear(1, 64, bias=1.0), nn.LeakyReLU(0.2, True), nn.Linear(64, 1, bias=1.0), ])
        # nn.init.xavier_normal_(self.mlp[0].weight, gain=1.414)
        # nn.init.xavier_normal_(self.mlp[-1].weight, gain=1.414)
        # nn.init.ones_(self.mlp[0].bias)
        # nn.init.ones_(self.mlp[-1].bias)
        nn.init.kaiming_uniform_(self.mlp[0].weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp[-1].weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.mlp[0].weight)
        fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.mlp[-1].weight)
        bound = 1 / math.sqrt(fan_in)
        bound2 = 1 / math.sqrt(fan_in2)
        nn.init.uniform_(self.mlp[0].bias, -bound, bound)
        nn.init.uniform_(self.mlp[-1].bias, -bound2, bound2)
        self.use_attcurv = True if agg_type == 'attcurv' else False
        if self.use_attcurv:
            self.W_si = nn.Parameter(torch.zeros(size=(1, 1)), requires_grad=True)
            self.W_ei = nn.Parameter(torch.zeros(size=(1, 1)), requires_grad=True)
            zeros(self.W_si)
            zeros(self.W_ei)
            print('>> using {:.2f} curvature + {:.2f} attention aggregation'.format(self.W_si.item(), self.W_ei.item()))
            self.aggatt = PYGAtt(self.manifold, self.in_dim, self.dropout, c=self.c, heads=self.heads,
                                 concat=self.concat)
        else:
            print('>> Using curvature only')

    def forward(self, x, adj):
        assert isinstance(adj, list)
        assert len(adj) == 2
        edge_index, curvature = adj
        edge_index = edge_index.to(x.device)
        curvature = curvature.to(x.device)
        h = self.aggcurv(x, edge_index, curvature)
        if self.use_attcurv:
            h_att = self.aggatt(x, edge_index)
            h = (self.W_si.sigmoid() * h + self.W_ei.sigmoid() * h_att) / (self.W_si.sigmoid() + self.W_si.sigmoid())
            # h = self.W_si* h_curv + self.W_ei * h_att  # other method is applicable
        output = self.manifold.proj(self.manifold.expmap0(h, c=self.c), c=self.c)

        return output

    def aggcurv(self, h, edge_index, curvature):
        node_level = self.poincare.dist0(h.detach()).detach()
        # node_level = (self.manifold.logmap0(h, c=self.c)**2).sum(dim=1).sqrt().detach()
        num_nodes = len(torch.unique(edge_index))
        h_tan = self.manifold.logmap0(h, c=self.c)
        edge_i = edge_index[0]
        edge_j = edge_index[1]

        # node_level_xi = torch.nn.functional.embedding(edge_i, node_level).unsqueeze(1)
        # node_level_xj = torch.nn.functional.embedding(edge_j, node_level).unsqueeze(1)

        node_level_xi = node_level[edge_i].unsqueeze(1)
        node_level_xj = node_level[edge_j].unsqueeze(1)
        if self.add_level_info:
            curv = self.mlp((curvature - abs(node_level_xi - node_level_xj))).to(edge_index.device)
        else:
            curv = self.mlp(curvature).to(edge_index.device)
        curv = F.dropout(curv, 0.2, training=self.training)
        norm_curv = softmax(src=curv, index=edge_i, num_nodes=num_nodes)

        x_j = h_tan[edge_j] * norm_curv
        # x_j = torch.nn.functional.embedding(edge_j, h_tan) * (norm_curv)
        x_curv = scatter(x_j, edge_i, dim=0, reduce="sum")
        return x_curv

    def extra_repr(self):
        return 'c={}'.format(self.c)
