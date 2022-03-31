import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

worker_args = {}
neigh_cache = {}


def init_radius_neighbor_worker(kwargs):
    worker_args['distance_metric'] = kwargs.get('distance_metric', None)
    worker_args['margin'] = kwargs.get('margin', None)
    worker_args['reduce_redundancy'] = kwargs.get('reduce_redundancy', None)
    worker_args['sample_points_dic'] = kwargs.get('sample_points_dic', None)
    worker_args['sample_points_shape_dic'] = kwargs.get('sample_points_shape_dic', None)
    worker_args['sample_points_dtype_dic'] = kwargs.get('sample_points_dtype_dic', None)


def radius_neighbor_worker(idx_pair):
    distance_metric = worker_args['distance_metric']
    margin = worker_args['margin']
    reduce_redundancy = worker_args['reduce_redundancy']
    sample_points_dic = worker_args['sample_points_dic']
    sample_points_shape_dic = worker_args['sample_points_shape_dic']
    sample_points_dtype_dic = worker_args['sample_points_dtype_dic']

    # Unpack data.
    idx_1, idx_2 = idx_pair
    object_1_points_shape = sample_points_shape_dic[idx_1]
    object_2_points_shape = sample_points_shape_dic[idx_2]
    object_1_points = np.frombuffer(sample_points_dic[idx_1],
                                    dtype=sample_points_dtype_dic[idx_1]).reshape(object_1_points_shape)
    object_2_points = np.frombuffer(sample_points_dic[idx_2],
                                    dtype=sample_points_dtype_dic[idx_2]).reshape(object_2_points_shape)

    # Check cache for neigh.
    neigh = neigh_cache.get(idx_1, None)

    if neigh is None:
        # Create neighbors graph.
        # Get connectivity for all within margin.
        # Create nearest neighbors tree.
        neigh = NearestNeighbors(radius=margin, metric=distance_metric)
        neigh.fit(object_1_points.reshape(-1, object_1_points.shape[-1]))

        # Put neigh in cache.
        neigh_cache[idx_1] = neigh

    # Create neighbors graph.
    # Get connectivity for all within margin.
    radius_neighbors_graph = neigh.radius_neighbors_graph(object_2_points.reshape(-1, object_2_points.shape[-1]))

    # Get indices for all combined graph connections.
    indices_2, indices_1, _ = sparse.find(radius_neighbors_graph)

    if indices_1.size == 0:
        # If there are no neighbors, return.
        return indices_1, indices_2

    # Remove redundany neighbors.
    if reduce_redundancy:

        # Find sizes of the columns.
        column_size_1 = np.product(object_1_points_shape[1:-1])
        column_size_2 = np.product(object_2_points_shape[1:-1])

        # Get the column indices of the node indices.
        column_indices_2 = indices_2 % column_size_2
        # Get first unique combination of comlumns.
        _, unique_column_indices = np.unique([indices_1, column_indices_2], return_index=True, axis=1)

        # Filter indices to have only one edge between each column.
        indices_1 = indices_1[unique_column_indices]
        indices_2 = indices_2[unique_column_indices]

        # Get the column indices of the node indices.
        column_indices_1 = indices_1 % column_size_1
        # Get first unique combination of comlumns.
        _, unique_column_indices = np.unique([column_indices_1, indices_2], return_index=True, axis=1)

        # Filter indices to have only one edge between each column.
        indices_1 = indices_1[unique_column_indices]
        indices_2 = indices_2[unique_column_indices]

    return indices_1, indices_2