import torch


def find_triangles(n, edge_index):
    # Convert edge_index to a set of edges for faster lookups
    edges = {(edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.shape[1])}

    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            # Check if i and j are connected by an edge
            if (i, j) in edges or (j, i) in edges:
                for k in range(j + 1, n):
                    # Check if i and k are connected by an edge
                    if (i, k) in edges or (k, i) in edges:
                        # Check if j and k are connected by an edge
                        if (j, k) in edges or (k, j) in edges:
                            triangles.append([i, j, k])
    # Convert list of triangles to tensor
    triangles = torch.tensor(triangles)
    return triangles
