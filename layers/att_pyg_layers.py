import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, ones, zeros
from torch_geometric.nn import MessagePassing, GATConv
from torch_scatter import scatter


class PYGAtt(MessagePassing):
    def __init__(self, manifold, out_features, dropout, c=None, heads=1, original='curv', concat=False):
        super(PYGAtt, self).__init__()
        self.manifold = manifold
        self.dropout = dropout
        self.out_channels = out_features // heads
        self.negative_slope = 0.2
        self.heads = heads
        self.original = original
        self.c = c
        self.concat = concat
        self.original = original
        self.att_i = Parameter(torch.Tensor(1, heads, self.out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, self.out_channels))
        glorot(self.att_i)
        glorot(self.att_j)
        if self.original == 'att':
            print('>> att agg from only att')

    '''
    def forward(self, x, edge_index):
        if edge_index.layout is torch.sparse_coo:
            edge_index = edge_index._indices()

        x_tangent0 = self.manifold.logmap0(x, c=self.c)  # project to origin
        out = self.propagate(edge_index, x=x_tangent0, num_nodes=x.size(0))
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.original == 'att':
            out = self.manifold.proj(self.manifold.expmap0(out, c=self.c), c=self.c)
        return out

    def message(self, edge_index_i, x_i, x_j, num_nodes):
        # Compute attention coefficients.
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        alpha = (x_i * self.att_i).sum(-1) + (x_j * self.att_j).sum(-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)
    '''

    def forward(self, x, edge_index):
        if edge_index.layout is torch.sparse_coo:
            edge_index = edge_index._indices()
        edge_index_i = edge_index[0]
        edge_index_j = edge_index[1]

        x_tangent0 = self.manifold.logmap0(x, c=self.c)  # project to origin
        x_i = torch.nn.functional.embedding(edge_index_i, x_tangent0)
        x_j = torch.nn.functional.embedding(edge_index_j, x_tangent0)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        alpha = (x_i * self.att_i).sum(-1) + (x_j * self.att_j).sum(-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(src=alpha, index=edge_index_i, num_nodes=x_i.size(0))
        # print(alpha[edge_index_i==10])
        # print(edge_index_j[edge_index_i==10])
        alpha = F.dropout(alpha, self.dropout, training=self.training)

        support_t = scatter(x_j * alpha.view(-1, self.heads, 1), edge_index_i, dim=0, reduce='sum')

        if self.concat:
            support_t = support_t.view(-1, self.heads * self.out_channels)
        else:
            support_t = support_t.mean(dim=1)

        if self.original == 'att':
            support_t = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), self.c)
        return support_t
