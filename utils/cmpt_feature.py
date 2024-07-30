import networkx as nx
import numpy as np
import torch
from node2vec import Node2Vec
import os


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def getfeature(edge_index, args):
    # Create a graph from the edge_index tensor
    feat_cached_dir = './cached/structure_feature/'
    feat_cached_file = feat_cached_dir + f'{args.dataset}_{args.task}.pt'
    mkdirs(feat_cached_dir)

    if os.path.isfile(feat_cached_file):
        print('>> loading cached feat file from directly...')
        return torch.load(feat_cached_file)

    G = nx.Graph()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        G.add_edge(src.item(), dst.item())

    # Calculate degree centrality for each node
    degree_centrality = nx.degree_centrality(G)

    # Calculate Katz centrality for each node
    katz_centrality = nx.katz_centrality_numpy(G)

    # Calculate clustering coefficient for each node
    clustering_coefficient = nx.clustering(G)

    # Set the dimensions of the node2vec embeddings
    dimensions = 64

    # Set the walk length and number of walks for node2vec
    walk_length = 20
    num_walks = 10

    # Create a node2vec object with the appropriate parameters
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks)

    # Fit the node2vec model to the graph
    model = node2vec.fit(window=10, min_count=1)

    # Calculate the node2vec embeddings for each node
    node2vec_features = {}
    for node in G.nodes():
        node2vec_features[node] = model.wv[str(node)]

    # Combine all features into a single dictionary for each node
    node_features = {}
    for node in G.nodes():
        node_features[node] = {
            "degree_centrality": degree_centrality[node],
            "katz_centrality": katz_centrality[node],
            "clustering_coefficient": clustering_coefficient[node],
            "node2vec_feature": node2vec_features[node]
        }

    # Convert the dictionary of dictionaries into a feature matrix
    n = max(G.nodes()) + 1
    d = dimensions + 3
    feature_matrix = torch.zeros((n, d))  # initialize feature matrix with zeros
    for i, node in enumerate(range(n)):
        if node in G.nodes():
            feature_matrix[i, 0] = degree_centrality[node]
            feature_matrix[i, 1] = katz_centrality[node]
            feature_matrix[i, 2] = clustering_coefficient[node]
            feature_matrix[i, 3:] = torch.tensor(node2vec_features[node])
        else:
            print(f'node {i} is randomly sampled')
            feature_matrix[i] = torch.rand(d)

    # Print the feature matrix
    print("Feature matrix:")
    print(feature_matrix.shape)
    torch.save(feature_matrix, feat_cached_file)
    print(f'{args.dataset} feat matrix has been saved!')
    return feature_matrix


if __name__ == '__main__':
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 7], [1, 0, 2, 1, 4, 8]])  # replace with your edge_index tensor
    getfeature(edge_index)
