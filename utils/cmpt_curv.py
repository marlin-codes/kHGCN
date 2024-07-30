import torch
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
import numpy as np
import os
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops
from config import parser

args = parser.parse_args()


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


'''
Computation process:
(1) remove self-loop edges
(2) compute curvature and assign value to self-loop
(3) cat the above two parts curvatures
'''


def compute_curvature(edge_index, edge_weight=None, dtype=None, curvature_type='ricci', alpha=0.5):
    root = mkdirs('./cached/curvatures')
    if args.task == 'nc':
        curvature_cached_file = root + '/{}_{}_{}_curv_nc.pt'.format(args.dataset, curvature_type, alpha)

    if args.task == 'lp':
        curvature_cached_file = root + '/{}_{}_{}_curv_lp.pt'.format(args.dataset, curvature_type, alpha)
    if os.path.isfile(curvature_cached_file):
        print('INFO:loading {} {} {} curvature file from {}'.format(args.dataset, curvature_type, alpha,
                                                                    curvature_cached_file))
        return torch.load(curvature_cached_file)
    else:
        print('>> computing curvature ...')

    edge_index_no_loop, _ = remove_self_loops(edge_index)  # remove self-loop
    edge_index = edge_index_no_loop

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

    weight = edge_weight.cpu().unsqueeze(1).numpy()
    links = edge_index.cpu().numpy().T
    G = nx.Graph()

    weighted_links = np.concatenate([links, weight], axis=1)  # concatenate links and weight
    G.add_weighted_edges_from(weighted_links)
    if curvature_type == 'ricci':
        print('>> computing {} {} ollivier Ricci curvature for {}...'.format(args.dataset, alpha, args.task))
        orc = OllivierRicci(G, alpha=alpha, verbose="INFO")  # compute curvature
    OG = orc.compute_ricci_curvature()
    curvature_list = []
    positive_curvature_list = []
    negative_curvature_list = []
    zero_curvature_List = []

    for (i, j) in links:
        curvature = orc.G[i][j]["{}Curvature".format(curvature_type)]
        curvature_list.append(curvature)
        if curvature > 0:
            positive_curvature_list.append([i, j, curvature])
        if curvature < 0:
            negative_curvature_list.append([i, j, curvature])
        if curvature == 0:
            zero_curvature_List.append([i, j, curvature])

    curvature = torch.from_numpy(np.array(curvature_list)).to(edge_index.device)
    edge_index, edge_weight = add_remaining_self_loops(edge_index_no_loop)  # add self-loop
    self_loop_curvature = torch.zeros(edge_index.shape[1] - curvature.shape[0]).to(
        edge_index.device)  # curvature for self-loop
    curvature = torch.cat([curvature.float(), self_loop_curvature.float()]).unsqueeze(1)  # cat curvature with self-loop

    positive_curvature = torch.from_numpy(np.array(positive_curvature_list)).transpose(1, 0)
    # negative_curvature = np.array(negative_curvature_list)
    # zero_curvature = np.array(zero_curvature_List)
    node_curv_list = []
    negative_node = 0
    positive_node = 0
    zero_node = 0
    total_num_nodes = len(OG.nodes)
    for n in range(len(OG.nodes())):
        try:
            ncurv = OG.nodes[n]['ricciCurvature']
        except:
            print('cannot find the ricci cuvature of node {}'.format(n))
            ncurv == 0.0
        node_curv_list.append(ncurv)
        if ncurv > 0.1:
            positive_node += 1
        if -0.01 < ncurv <= 0.1:
            zero_node += 1
        if ncurv <= -0.01:
            negative_node += 1
    # print(negative_node/total_num_nodes, zero_node/total_num_nodes, positive_node/total_num_nodes)
    print(
        '\tCurvature statistics of {}  \n\t\t{}/{} positive curvature nodes \n\t\t{}/{} zero curvature nodes \n\t\t{}/{} negative curvature nodes'.format(
            args.dataset, positive_node, total_num_nodes, zero_node, total_num_nodes, negative_node, total_num_nodes))

    node_curvature = torch.tensor(data=node_curv_list)
    print('Max curvature: {:.2f}'.format(node_curvature.max()))
    print('Min curvature: {:.2f}'.format(node_curvature.min()))
    print('Mean curvature: {:.2f} ± {:.2f}'.format(node_curvature.mean(), node_curvature.std()))
    torch.save([edge_index, curvature, positive_curvature, node_curvature], curvature_cached_file)
    print('>> {} {} curvature saved to {}'.format(args.dataset, alpha, curvature_cached_file))
    return edge_index, curvature, positive_curvature, node_curvature


if __name__ == '__main__':
    from utils.data_utils import load_data
    import os
    import sys

    root = os.path.dirname(os.path.dirname(os.getcwd()))
    args.dataset = 'cora'
    args.task = 'lp'

    if args.task == 'nc':
        edge_name = 'adj_train'  # all edges

    if args.task == 'lp':
        edge_name = 'train_edges'  # only tain edges

    data = load_data(args, os.path.join(root, 'data', args.dataset))
    edge_index = data[edge_name]._indices()
    compute_curvature(edge_index)
