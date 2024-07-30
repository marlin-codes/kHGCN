"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils.negative_sampling import structured_negative_sampling
from torch_geometric.utils import softmax, k_hop_subgraph
from utils.cmpt_curv import compute_curvature
from utils.cmpt_feature import getfeature
from utils.add_edges import add_edges_to_tree
from utils.find_triangle import find_triangles


def load_data(args, datapath):
    processed_pubmed = args.data_root + '/pubmed/cached_{}.pt'.format(args.task)
    if args.dataset == 'pubmed' and os.path.isfile(processed_pubmed):
        print('>> loading cached pubmed {} data meta file from {}'.format(args.task, processed_pubmed))
        data = torch.load(processed_pubmed)
        I = data['adj_train_norm'][0]
        V = data['adj_train_norm'][1]
        size = data['adj_train_norm'][2]
        data['adj_train_norm'] = torch.sparse_coo_tensor(I, V, size)
        if args.pretrained_embeddings is not None:
            data['features'] = torch.from_numpy(np.load(args.pretrained_embeddings))
            print('>> loading pretraining feataures from {}'.format(args.pretrained_embeddings))
    else:
        if args.task == 'nc':
            data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed, args)
        else:
            data = load_data_lp(args.dataset, args.use_feats, datapath, args)
            adj = data['adj_train']
            if args.task == 'lp':
                adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                    adj, args.val_prop, args.test_prop, args.split_seed
                )
                data['adj_train'] = adj_train
                data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
                data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
                data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
        data['adj_train_norm'], data['features'] = process(data['adj_train'], data['features'], args.normalize_adj,
                                                           args.normalize_feats)
        train_edge_index = data['adj_train_norm']._indices()
        # data['features'] = getfeature(train_edge_index, args)
        data['triangle'] = find_triangles(data['adj_train_norm'].shape[0], train_edge_index)

        if args.dataset == 'pubmed' and (not os.path.isfile(processed_pubmed)):
            I = data['adj_train_norm']._indices()
            V = data['adj_train_norm']._values()
            size = data['adj_train_norm'].size()
            data['adj_train_norm'] = (I, V, size)
            print('save pubmed {} data meta to {}'.format(args.task, processed_pubmed))
            torch.save(data, processed_pubmed)
            data['adj_train_norm'] = torch.sparse_coo_tensor(I, V, size)

    edge_index, edge_curvature, positive_edge, node_curvature = compute_curvature(data['adj_train_norm']._indices())
    data['node_curvature'] = node_curvature

    if args.agg_type == 'curv' or args.agg_type == 'attcurv':
        data['edges_true'] = edge_index
        ijk = structured_negative_sampling(edge_index)
        data['edges_false'] = torch.cat([ijk[0].unsqueeze(0), ijk[2].unsqueeze(0)], dim=0)
        data['adj_train_norm'] = [edge_index, edge_curvature]
        if args.sample_mode is None:
            pass
        elif args.sample_mode == 'II':
            data['edges_true'] = sampling_edges(edge_index, edge_curvature, mode=args.sample_mode,
                                                drop_ratio=args.edge_drop_ratio)
        elif args.sample_mode == 'I':
            data['edges_true'], neg_edges_index = sampling_edges(edge_index, edge_curvature, mode=args.sample_mode)
            data['edges_false'] = torch.cat([data['edges_false'], neg_edges_index], dim=1)
        else:
            raise Exception('wrong sample modes')
        print('>> sampling mode is {}'.format(args.sample_mode))

    return data


def sampling_edges(edge_index, curvature, mode='I', drop_ratio=0.05):
    edge_curv = torch.cat([curvature, edge_index.transpose(1, 0).float()], dim=1)
    # mask 5% edges and treat it as negative samples
    if mode == 'I':
        print('>> Sampling strategies: masking and drop top {}% positive edges and as negative edges')

        new_edge_index = [item.unsqueeze(0) for item in sorted(edge_curv, key=lambda x: x[0])]
        new_edge_index = torch.cat(new_edge_index, dim=0)
        new_edge_index = new_edge_index[:, 1:].long()

        pos_edge_index = new_edge_index[int(new_edge_index.size(0) * drop_ratio):, :]
        neg_edges_index = new_edge_index[:int(new_edge_index.size(0) * drop_ratio), :]

        pos_edge_index = pos_edge_index.transpose(1, 0)
        neg_edges_index = neg_edges_index.transpose(1, 0)

        return pos_edge_index, neg_edges_index

    # mask 10% edges
    if mode == 'II':
        print('>> Sampling strategies: masking top {}% edges'.format(drop_ratio * 100))
        new_edge_index = [item.unsqueeze(0) for item in sorted(edge_curv, key=lambda x: x[0])]
        new_edge_index = torch.cat(new_edge_index, dim=0)
        new_edge_index = new_edge_index[:, 1:].long()
        pos_edge_index = new_edge_index[int(new_edge_index.size(0) * drop_ratio):, :]
        pos_edge_index = pos_edge_index.transpose(1, 0)
        return pos_edge_index


# ############### FEATURES PROCESSING ####################################
def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATA SPLITS #####################################################


def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
        torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
        test_edges_false)


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


# ############### LINK PREDICTION DATA LOADERS ####################################


def load_data_lp(dataset, use_feats, data_path, args):
    if dataset in ['cora', 'pubmed', 'citeseer']:
        adj, features = load_citation_data(dataset, use_feats, data_path, args=args)[:2]
    elif dataset == 'disease_lp':
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'airport':
        adj, features = load_data_airport(dataset, data_path, return_label=False)
    elif dataset in ['tree_0.0', 'tree_0.1', 'tree_0.2', 'tree_0.3', 'tree_0.4', 'tree_0.5']:
        adj, features, labels = load_data_tree(dataset, use_feats, data_path)
        val_prop, test_prop = 0.15, 0.15
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    data = {'adj_train': adj, 'features': features}
    return data


# ############### NODE CLASSIFICATION DATA LOADERS ####################################


def load_data_nc(dataset, use_feats, data_path, split_seed, args):
    if dataset in ['cora', 'pubmed', 'citeseer']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(dataset, use_feats, data_path,
                                                                                 split_seed, args)
    else:
        if dataset == 'disease_nc' or dataset == 'disease_mc':
            adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.60
        elif dataset == 'airport':
            adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        else:
            raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val,
            'idx_test': idx_test}
    return data


# ############### DATASETS ####################################


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None, args=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + 500)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])
    if args.pretrained_embeddings is not None:
        features = np.load(args.pretrained_embeddings)
        print('loading pretraining feataures from {}'.format(args.pretrained_embeddings))

    return adj, features, labels, idx_train, idx_val, idx_test


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    # print(len(edges))
    # edges += [(8, 9), (7, 428), (427, 428)]
    # edges += [(8, 9), (7, 6), (7, 428), (10, 1), (20, 500)]

    # edges += [(8, 9)]
    # edges += [(10, 11), (14, 15), (1, 2), (3, 4)]
    # edges += [(10, 11), (14, 15), (1, 2), (3, 4), (62, 63),(50,51), (8,9)]
    # print(len(edges))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels


def load_data_tree(data_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path.rstrip(data_str), "trees/{}.txt".format(data_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split('\t')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    features = sp.eye(adj.shape[0])
    labels = np.ones(adj.shape[0])
    return sp.csr_matrix(adj), features, labels


def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph.nodes[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0 / 7, 8.0 / 7, 9.0 / 7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features


if __name__ == '__main__':
    import time
    import numpy as np
    import networkx as nx
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='nc')
    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--use_feats', type=int, default=1)
    parser.add_argument('--split_seed', type=int, default=1234)
    parser.add_argument('--agg_type', type=str, default='curv')
    parser.add_argument('--normalize_adj', type=int, default=1)
    parser.add_argument('--normalize_feats', type=int, default=1)
    parser.add_argument('--val-prop', type=float, default=0.05)
    parser.add_argument('--test-prop', type=float, default=0.1)

    time_costs = []
    args = parser.parse_args()
    data = load_data(args, '{}'.format(args.dataset))
    edge_index = data['adj_train_norm']._indices()
    links = edge_index.cpu().numpy().T
    G = nx.Graph()
    G.add_edges_from(links)
    orc = OllivierRicci(G, alpha=0.5, verbose="INFO")  # compute curvature

    for _ in range(10):
        start_time = time.time()
        orc.compute_ricci_curvature()
        time_interval = time.time() - start_time
        print('cost:{:.4f}'.format(time_interval))
        time_costs.append(time_interval)
    print('>>{:.4f}\t{:.4f}'.format(np.mean(time_costs), np.std(time_costs)))
