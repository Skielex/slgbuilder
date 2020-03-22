import gc
from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


class SLGBuilder(ABC):

    INF_CAP_INT16 = 1 << 14
    INF_CAP_INT32 = 1 << 24
    INF_CAP_INT64 = 1 << 32
    INF_CAP_FLOAT32 = 1 << 16
    INF_CAP_FLOAT64 = 1 << 32

    def __init__(self, estimated_nodes=0, estimated_edges=0, flow_type=np.int32, jit_build=True):
        """Creates a ```SLGBuilder``` object used by subclasses to create graphs."""
        self.estimated_nodes = estimated_nodes
        self.estimated_edges = estimated_edges
        self.jit_build = jit_build

        self.graph = None

        self._set_flow_type_and_inf_cap(flow_type)

        if not jit_build:
            self.create_graph_object()

        self.objects = []
        self.nodes = []

        # Create lists for terms.
        self.unary_nodes = []
        self.unary_e0 = []
        self.unary_e1 = []
        self.pairwise_from = []
        self.pairwise_to = []
        self.pairwise_e00 = []
        self.pairwise_e01 = []
        self.pairwise_e10 = []
        self.pairwise_e11 = []

    @abstractmethod
    def _add_nodes(self, object):
        pass

    @abstractmethod
    def _set_flow_type_and_inf_cap(self, flow_type):
        pass

    @abstractmethod
    def add_object(self, graph_object, pack_nodes=False):
        pass

    def add_objects(self, graph_objects, pack_nodes=False):
        return [self.add_object(o, pack_nodes=pack_nodes) for o in graph_objects]

    def get_nodeids(self, graph_object):
        nodeids = self.nodes[self.objects.index(graph_object)]
        if np.isscalar(nodeids):
            return np.arange(nodeids, nodeids + graph_object.data.size).reshape(graph_object.data.shape)
        else:
            # It is an array.
            return nodeids

    @abstractmethod
    def add_unary_terms(self, i, e0, e1):
        pass

    @abstractmethod
    def add_pairwise_terms(self, i, j, e00, e01, e10, e11):
        pass

    @abstractmethod
    def get_labels(self, i):
        pass

    @abstractmethod
    def solve(self):
        pass

    def build_graph(self, pack_nodes=False):

        if self.graph is not None:
            return

        if self.estimated_nodes == 0:
            node_count = sum([obj.data.size for obj in self.objects])
            self.estimated_nodes = node_count

        if self.estimated_edges == 0:
            # This may be an overestimate.
            edge_count = sum([f.size for f in self.pairwise_from])
            self.estimated_edges = edge_count

        self.create_graph_object()

        for obj in self.objects:
            # Add nodes to graph.
            self._add_nodes(obj)

        # Terminal edges.
        if self.unary_nodes:

            # Concatenate lists.
            u_i = np.concatenate(self.unary_nodes)

            e0 = np.concatenate(self.unary_e0)
            e1 = np.concatenate(self.unary_e1)

            # No longer needed.
            self.unary_nodes.clear()
            self.unary_e0.clear()
            self.unary_e1.clear()

            # Add edges to graph.
            step = 100000
            for r in range(0, u_i.size, step):
                self.add_unary_terms(u_i[r:r + step], e0[r:r + step], e1[r:r + step])

            # No longer needed.
            del u_i
            del e0
            del e1

        # Node edges.
        if self.pairwise_from:

            # Concatenate lists.
            i = np.concatenate(self.pairwise_from)
            j = np.concatenate(self.pairwise_to)

            e00 = np.concatenate(self.pairwise_e00)
            e01 = np.concatenate(self.pairwise_e01)
            e10 = np.concatenate(self.pairwise_e10)
            e11 = np.concatenate(self.pairwise_e11)

            # No longer needed.
            self.pairwise_from.clear()
            self.pairwise_to.clear()
            self.pairwise_e00.clear()
            self.pairwise_e01.clear()
            self.pairwise_e10.clear()
            self.pairwise_e11.clear()

            # Sort.
            sort_indices = np.lexsort((-i, -j))

            # Sorted edges.
            i = i[sort_indices]
            j = j[sort_indices]
            e00 = e00[sort_indices]
            e01 = e01[sort_indices]
            e10 = e10[sort_indices]
            e11 = e11[sort_indices]

            # GC before adding edges.
            gc.collect()

            # Add edges to graph.
            step = 100000
            for r in range(0, i.size, step):
                self.add_pairwise_terms(i[r:r + step], j[r:r + step], e00[r:r + step],
                                        e01[r:r + step], e10[r:r + step], e11[r:r + step])

    def add_boundary_cost(self, objects=None, sigma=None, symmetric=True):
        """Add boundary cost by adding edges between non-terminal nodes based on object data."""

        if objects is None:
            objects = self.objects

        # Calculates boundary penalties using.
        def get_boundary_cost(diff):
            if issubclass(self.flow_type, np.integer):
                diff = diff.astype(np.float)
            else:
                diff = diff.copy()

            min_value = np.min(diff)
            diff -= min_value
            max_value = np.max(diff)
            diff /= max_value

            if symmetric:
                if sigma is None:
                    diff = 1 - np.abs(diff)
                else:
                    diff = np.exp(-(diff**2) / (2 * sigma**2))
            else:
                diff_mask = diff > 0
                if sigma is None:
                    diff[diff_mask] = 1 - np.abs(diff[diff_mask])
                else:
                    diff[diff_mask] = np.exp(-(diff[diff_mask]**2) / (2 * sigma**2))
                diff[~diff_mask] = 1

            diff *= max_value
            diff += min_value

            if issubclass(self.flow_type, np.integer):
                diff = np.round(diff).astype(self.flow_type)

            return diff

        for obj in objects:
            # Assume grid for now.
            # Get nodes for object in this graph.
            nodeids = self.get_nodeids(obj)

            for dim in range(nodeids.ndim):

                # Move current dim ot front.
                data = np.moveaxis(obj.data, dim, 0)
                data = data.reshape(data.shape[0], -1)

                # Find difference.
                diff = np.diff(data, axis=0)

                # Calculate cost.
                cost_f = get_boundary_cost(-diff)
                cost_b = get_boundary_cost(diff)

                ids = np.moveaxis(nodeids, dim, 0)
                ids = ids.reshape(ids.shape[0], -1)

                # Add boundary terms (edges).
                self.add_pairwise_terms(ids[:-1], ids[1:], 0, cost_f, cost_b, 0)

    def add_region_cost(self, objects=None, alpha=0.1):
        """Add region cost by adding terminal edges with capacity based on the object data."""

        if objects is None:
            objects = self.objects

        def add_region_edges(g, b, nodeids):

            b_mask = b > 0
            not_b_mask = ~b_mask
            b = np.abs(b)

            # Add unary terms (terminal edges).
            self.add_unary_terms(nodeids[b_mask], 0, b[b_mask])
            self.add_unary_terms(nodeids[not_b_mask], b[not_b_mask], 0)

        for obj in self.objects:
            # Get nodes for object in this graph.
            nodeids = self.get_nodeids(obj)

            # Get region cost.
            b = obj.data
            b -= np.min(b)
            b /= np.max(b)
            b = 2 * b - 1

            # Add region edges.
            add_region_edges(self.graph, b, nodeids)

    def add_containment(self, outer_object, inner_object, margin=1, distance_metric='l1'):
        """Add containment constraint edges forcing inner_object to be within outer_object."""

        if outer_object == inner_object:
            raise ValueError('outer_object and inner_object is the same object.')

        if margin is None:
            return

        # Get nodeids.
        outer_nodeids = self.get_nodeids(outer_object)
        inner_nodeids = self.get_nodeids(inner_object)

        # Get points.
        outer_points = outer_object.sample_points
        inner_points = inner_object.sample_points

        # TODO Avoid searching for neigbours if possible.
        if margin == 0 and outer_points.shape == inner_points.shape and np.all(outer_points == inner_points):
            # Add containment edges.
            self.add_pairwise_terms(outer_nodeids, inner_nodeids, 0, self.inf_cap, 0, 0)
        else:
            # Create nearest neighbors tree.
            neigh = NearestNeighbors(radius=margin, metric=distance_metric)
            neigh.fit(outer_points.reshape(-1, outer_points.shape[-1]))

            # Create neighbors graph.
            # Get connectivity for all within margin.
            radius_neighbors_graph = neigh.radius_neighbors_graph(inner_points.reshape(-1, inner_points.shape[-1]))

            # Get indices for all combined graph connections.
            indices_2, indices_1, _ = sparse.find(radius_neighbors_graph)

            outer_ids = np.take(outer_nodeids, indices_1)
            inner_ids = np.take(inner_nodeids, indices_2)

            # Add containment edges.
            self.add_pairwise_terms(outer_ids, inner_ids, 0, self.inf_cap, 0, 0)

    def add_exclusion(self, object_1, object_2, margin=1, distance_metric='l1'):
        """Add exclsion constraint edges forcing object_1 and object_2 not to overlap."""

        if object_1 == object_2:
            raise ValueError('object_1 and object_2 is the same object.')

        # Get nodeids.
        object_1_nodeids = self.get_nodeids(object_1)
        object_2_nodeids = self.get_nodeids(object_2)

        # Get points.
        object_1_points = object_1.sample_points
        object_2_points = object_2.sample_points

        # TODO Avoid searching for neigbours if possible.
        if margin == 0 and object_1_points.shape == object_2_points.shape and np.all(object_1_points == object_2_points):
            # Add containment edges.
            self.add_pairwise_terms(object_1_nodeids, object_2_nodeids, 0, 0, 0, self.inf_cap)
        else:
            # Create nearest neighbors tree.
            neigh = NearestNeighbors(radius=margin, metric=distance_metric)
            neigh.fit(object_1_points.reshape(-1, object_1_points.shape[-1]))

            # Create neighbors graph.
            # Get connectivity for all within margin.
            radius_neighbors_graph = neigh.radius_neighbors_graph(
                object_2_points.reshape(-1, object_2_points.shape[-1]))

            # Get indices for all combined graph connections.
            indices_2, indices_1, _ = sparse.find(radius_neighbors_graph)

            if indices_1.size == 0:
                # If there are no neighbors, return.
                return

            # Take the ids to add pairwise terms to.
            object_1_ids = np.take(object_1_nodeids, indices_1)
            object_2_ids = np.take(object_2_nodeids, indices_2)

            # Add exclusion edges.
            self.add_pairwise_terms(object_1_ids, object_2_ids, 0, 0, 0, self.inf_cap)

    def add_labels(self, graph_object, source_capacities, sink_capacities):
        """Add terminal edges."""
        nodeids = self.get_nodeids(graph_object)

        # Filter to avoid calling function with irrelevant arguments (zeros).
        mask = (sink_capacities != 0) | (source_capacities != 0)

        if np.any(mask):
            self.add_unary_terms(nodeids[mask], sink_capacities[mask], source_capacities[mask])

    def add_smoothness(self, objects=None, beta=0.01):
        """Add soft smoothness constraint."""

        if objects is None:
            objects = self.objects

        for obj in objects:

            # Get node ids.
            nodeids = self.get_nodeids(obj)

            # For each dimension we want to join neighbors.
            for dim in range(0, nodeids.ndim):

                # Move axis to front.
                ids = np.moveaxis(nodeids, dim, 0)

                # Add pairwise terms (edges) between neighbors.
                self.add_pairwise_terms(ids[:-1], ids[1:], 0, beta, beta, 0)

    def add_layered_boundary_cost(self, objects=None):
        """Add layered boundary cost. This function assumes an N-D regular grid."""

        if objects is None:
            objects = self.objects

        for obj in objects:
            # Get nodes for object in this graph.
            nodeids = self.get_nodeids(obj)

            # Calculate weights (Eq1).
            # Prevent empty solution (sec 4.1).
            w = np.zeros(obj.data.shape, dtype=self.flow_type)
            w[0] = -self.inf_cap
            w[1:] = np.diff(obj.data, axis=0)

            # Add intracolumn edges (Eq2).
            self.add_pairwise_terms(nodeids[:-1], nodeids[1:], 0, self.inf_cap, 0, 0)

            # Add terminal edges (sec 4.2).
            positive_mask = w >= 0
            # Connect positive weights to source and negative weights to sink.
            e1 = w
            e0 = -w.copy()
            e1[~positive_mask] = 0
            e0[positive_mask] = 0

            # Add unary terms (terminal edges).
            self.add_unary_terms(nodeids, e0, e1)

    def add_layered_region_cost(self, graph_object, outer_region_cost, inner_region_cost):
        """Add layered region cost. This function assumes an N-D regular grid."""

        # Get nodes for object in this graph.
        nodeids = self.get_nodeids(graph_object)

        # Calculate weights.
        w = inner_region_cost - outer_region_cost
        # Prevent empty solution.
        w[0] = -self.inf_cap

        # Add terminal edges.
        positive_mask = w >= 0
        # Connect positive weights to source and negative weights to sink.
        e1 = w
        e0 = -w
        e1[~positive_mask] = 0
        e0[positive_mask] = 0

        # Add terminal edges.
        self.add_unary_terms(nodeids, e0, e1)

    def add_layered_smoothness(self, objects=None, delta=1, wrap=True):
        """Add hard smoothness constraint to layered object. This function assumes an N-D regular grid."""

        if objects is None:
            objects = self.objects

        # Create delta per object if not supplied.
        if np.isscalar(delta):
            delta = (delta * np.ones(len(objects))).astype(np.int)

        # Create wrap per object if not supplied.
        if np.isscalar(wrap):
            wrap = (wrap * np.ones(len(objects))).astype(np.bool)

        for i, obj in enumerate(objects):

            # Get nodes for object in this graph.
            nodeids = self.get_nodeids(obj)

            if nodeids.shape[0] <= 1:
                # If the first axis is 0 or 1, skip object.
                continue

            object_delta = delta[i]
            # Expand delta to object data dimensions.
            if np.isscalar(object_delta):
                object_delta = np.repeat(object_delta[np.newaxis], nodeids.ndim - 1, axis=-1)

            object_wrap = wrap[i]
            # Expand wrap to object data dimensions.
            if np.isscalar(object_wrap):
                object_wrap = np.repeat(object_wrap[np.newaxis], nodeids.ndim - 1, axis=-1)

            # For each dimentions we will work on the two first axes.
            for dim in range(1, nodeids.ndim):
                # Move axis to second dim.
                ids = np.moveaxis(nodeids, dim, 1)

                # Delta for this dimension.
                # The is the distance between nodes on the axes of interest.
                dx = object_delta[dim - 1]

                if ids.shape[0] <= dx:
                    # If the first axis is less or equal dx, skip dimension.
                    continue

                if ids.shape[1] <= 1:
                    # If the second axis is 0 or 1, skip dimension.
                    continue

                if dx == 0:
                    # If dx is 0.
                    # Add intercolumn edges (Eq3).
                    # Add pairwise terms.
                    self.add_pairwise_terms(ids[:, :-1], ids[:, 1:], 0, self.inf_cap, 0, 0)
                    self.add_pairwise_terms(ids[:, 1:], ids[:, :-1], 0, self.inf_cap, 0, 0)

                    if object_wrap[dim - 1]:
                        # Add pairwise wrapping terms (connecting first and last).
                        self.add_pairwise_terms(ids[:, -1], ids[:, 0], 0, self.inf_cap, 0, 0)
                        self.add_pairwise_terms(ids[:, 0], ids[:, -1], 0, self.inf_cap, 0, 0)

                else:
                    # Add intercolumn edges (Eq3).
                    # Add pairwise terms.
                    self.add_pairwise_terms(ids[:-dx, :-1], ids[dx:, 1:], 0, self.inf_cap, 0, 0)
                    self.add_pairwise_terms(ids[:-dx, 1:], ids[dx:, :-1], 0, self.inf_cap, 0, 0)

                    if object_wrap[dim - 1]:
                        # Add pairwise wrapping terms (connecting first and last).
                        self.add_pairwise_terms(ids[:-dx, -1], ids[dx:, 0], 0, self.inf_cap, 0, 0)
                        self.add_pairwise_terms(ids[:-dx, 0], ids[dx:, -1], 0, self.inf_cap, 0, 0)

    def add_layered_containment(self, outer_object, inner_object, min_margin=0, max_margin=None, distance_metric='l2', reduce_redundancy=True):
        """Add layered containment."""

        if outer_object == inner_object:
            raise ValueError('outer_object and inner_object is the same object.')

        # Get nodeids.
        outer_nodeids = self.get_nodeids(outer_object)
        inner_nodeids = self.get_nodeids(inner_object)

        # Get points.
        outer_points = outer_object.sample_points
        inner_points = inner_object.sample_points

        if outer_points.ndim != inner_points.ndim or outer_points.shape[-1] != inner_points.shape[-1]:
            raise ValueError(
                'outer_object points and inner_object points must have the same number of dimensions and the same size last dimension.')

        # Check if the points are identical.
        if outer_points.shape == inner_points.shape and np.all(outer_points == inner_points):
            # If shapes and points match, this is the fast way.

            if max_margin is not None and outer_nodeids.shape[0] > max_margin:
                # Add max margin edges.
                if max_margin == 0:
                    self.add_pairwise_terms(inner_nodeids, outer_nodeids, 0, self.inf_cap, 0, 0)
                else:
                    self.add_pairwise_terms(inner_nodeids[:-max_margin],
                                            outer_nodeids[max_margin:], 0, self.inf_cap, 0, 0)
                    self.add_pairwise_terms(inner_nodeids[-max_margin:], outer_nodeids[-1], 0, self.inf_cap, 0, 0)

            if min_margin is not None and outer_nodeids.shape[0] > min_margin:
                # Add min margin edges.
                if min_margin == 0:
                    self.add_pairwise_terms(outer_nodeids, inner_nodeids, 0, self.inf_cap, 0, 0)
                else:
                    self.add_pairwise_terms(outer_nodeids[min_margin:],
                                            inner_nodeids[:-min_margin], 0, self.inf_cap, 0, 0)
                    self.add_pairwise_terms(outer_nodeids[:min_margin], inner_nodeids[0], 0, self.inf_cap, 0, 0)

        # Else we need to find nodes to connect.
        else:
            # Create flattened arrays of points.
            outer_points_flat = outer_points.reshape(-1, outer_points.shape[-1])
            inner_points_flat = inner_points.reshape(-1, inner_points.shape[-1])

            # Find sizes of the columns.
            outer_columns_size = np.product(outer_nodeids.shape[1:])
            inner_column_size = np.product(inner_nodeids.shape[1:])

            # Create nearest neighbors tree.
            neigh = NearestNeighbors(metric=distance_metric)
            neigh.fit(outer_points_flat)

            if max_margin is not None:

                # Find direction of points.
                outer_point_gradients = np.gradient(outer_points, axis=0)
                inner_point_gradients = np.gradient(inner_points, axis=0)

                # Move inner points in the direction of the gradient.
                # The distance moved is the max margin.
                inner_points_moved = inner_points + \
                    (max_margin * inner_point_gradients /
                     np.sqrt(np.sum(inner_point_gradients**2, axis=-1)[..., np.newaxis]))
                inner_points_moved_flat = inner_points_moved.reshape(-1, outer_points.shape[-1])

                # Find the 4 nearest neighbours for moved points. This should be enough.
                radius_neighbors_graph = neigh.kneighbors_graph(
                    inner_points_moved_flat, n_neighbors=4, mode='connectivity')

                # Get indices for all combined graph connections.
                inner_indices, outer_indices, _ = sparse.find(radius_neighbors_graph)
                if outer_indices.size > 0:
                    # The following code filters out redundant terms before adding terms and tries to ensure a meaningful max margin.

                    # Find distances between neighbours.
                    # Create mask for neighbours futher than max margin away.
                    distance_mask = np.sum(
                        (outer_points_flat[outer_indices] - inner_points_flat[inner_indices])**2, axis=-1) > max_margin**2

                    # Only keep edges longer than max margin.
                    outer_indices = outer_indices[distance_mask]
                    inner_indices = inner_indices[distance_mask]

                    # Find angles between gradients.
                    angles = np.einsum('ij,ij->i',
                                       outer_point_gradients.reshape(-1, outer_points.shape[-1])[outer_indices],
                                       inner_point_gradients.reshape(-1, outer_points.shape[-1])[inner_indices])
                    angle_mask = angles > 0

                    # Only keep edges where the gradients are points in the same direction.
                    outer_indices = outer_indices[angle_mask]
                    inner_indices = inner_indices[angle_mask]
                    angles = angles[angle_mask]

                    # Reverse indices to get bigger indices first.
                    outer_indices = np.flip(outer_indices)
                    inner_indices = np.flip(inner_indices)
                    angles = np.flip(angles)

                    # Get the column indices of the node indices.
                    inner_column_indices = inner_indices % inner_column_size
                    # Get first unique combination of comlumns.
                    _, unique_column_indices = np.unique(
                        [outer_indices, inner_column_indices], return_index=True, axis=1)

                    # Filter indices to have only one from an outer node to each inner column.
                    outer_indices = outer_indices[unique_column_indices]
                    inner_indices = inner_indices[unique_column_indices]
                    angles = angles[unique_column_indices]

                    # Get sort indices, large angles (dot product) first.
                    angle_sort = np.argsort(-angles)

                    # Sort indices.
                    outer_indices = outer_indices[angle_sort]
                    inner_indices = inner_indices[angle_sort]

                    # Get the outer connection with biggest dot product.
                    _, unique_column_indices = np.unique(outer_indices, return_index=True)

                    # Only keep the connections to the outer node with the largest angle.
                    outer_indices = outer_indices[unique_column_indices]
                    inner_indices = inner_indices[unique_column_indices]

                    outer_ids = np.take(outer_nodeids, outer_indices)
                    inner_ids = np.take(inner_nodeids, inner_indices)

                    # Add containment edges.
                    self.add_pairwise_terms(inner_ids, outer_ids, 0, self.inf_cap, 0, 0)

            if min_margin is not None:

                radius_neighbors_graph = neigh.radius_neighbors_graph(inner_points_flat, radius=min_margin)
                # Removed K-nieghbors search for now.
                # Adding them may improve stability when resolution is low.
                # kneighbors_graph = neigh.kneighbors_graph(inner_points_flat, n_neighbors=2, mode='connectivity')
                # radius_neighbors_graph += kneighbors_graph

                # Get indices for all combined graph connections.
                inner_indices, outer_indices, _ = sparse.find(radius_neighbors_graph)
                if outer_indices.size > 0:

                    # Remove redundany neighbors.
                    if reduce_redundancy:

                        # The following code filters out redundant terms before adding terms.

                        # Get the column indices of the node indices.
                        inner_column_indices = inner_indices % inner_column_size
                        # Get first unique combination of comlumns.
                        _, unique_column_indices = np.unique(
                            [outer_indices, inner_column_indices], return_index=True, axis=1)

                        # Filter indices to have only one edge between each column.
                        outer_indices = outer_indices[unique_column_indices]
                        inner_indices = inner_indices[unique_column_indices]

                        # Reverse indices.
                        outer_indices = np.flip(outer_indices)
                        inner_indices = np.flip(inner_indices)

                        # Get the column indices of the node indices.
                        outer_column_indices = outer_indices % outer_columns_size
                        # Get first unique combination of comlumns.
                        _, unique_column_indices = np.unique(
                            [outer_column_indices, inner_indices], return_index=True, axis=1)

                        # Filter indices to have only one edge between each column.
                        outer_indices = outer_indices[unique_column_indices]
                        inner_indices = inner_indices[unique_column_indices]

                    outer_ids = np.take(outer_nodeids, outer_indices)
                    inner_ids = np.take(inner_nodeids, inner_indices)

                    # Add containment edges.
                    self.add_pairwise_terms(outer_ids, inner_ids, 0, self.inf_cap, 0, 0)

    def add_layered_exclusion(self, object_1, object_2, margin=1, distance_metric='l1', reduce_redundancy=True):
        """Add exclsion constraint edges forcing object_1 and object_2 not to overlap.
        This function assumes a layered boundary cost has been applied to the objects.
        """

        if object_1 == object_2:
            raise ValueError('object_1 and object_2 is the same object.')

        # Get nodeids.
        object_1_nodeids = self.get_nodeids(object_1)
        object_2_nodeids = self.get_nodeids(object_2)

        # Get points.
        object_1_points = object_1.sample_points
        object_2_points = object_2.sample_points

        # TODO Avoid searching for neigbours if possible.

        # Create nearest neighbors tree.
        neigh = NearestNeighbors(radius=margin, metric=distance_metric)
        neigh.fit(object_1_points.reshape(-1, object_1_points.shape[-1]))

        # Create neighbors graph.
        # Get connectivity for all within margin.
        radius_neighbors_graph = neigh.radius_neighbors_graph(object_2_points.reshape(-1, object_2_points.shape[-1]))

        # Get indices for all combined graph connections.
        indices_2, indices_1, _ = sparse.find(radius_neighbors_graph)

        if indices_1.size == 0:
            # If there are no neighbors, return.
            return

        # Remove redundany neighbors.
        if reduce_redundancy:

            # Find sizes of the columns.
            column_size_1 = np.product(object_1_nodeids.shape[1:])
            column_size_2 = np.product(object_2_nodeids.shape[1:])

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

        # Add exclusion terms.
        self.add_pairwise_terms(object_1_nodeids.flat[indices_1],
                                object_2_nodeids.flat[indices_2], 0, 0, 0, self.inf_cap)
