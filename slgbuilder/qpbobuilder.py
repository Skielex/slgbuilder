import numpy as np
import thinqpbo

from .graphobject import GraphObject
from .slgbuilder import SLGBuilder


class QPBOBuilder(SLGBuilder):
    def __init__(
        self,
        estimated_nodes=0,
        estimated_edges=0,
        capacity_type=np.int32,
        jit_build=True,
    ):
        """Creates a helper for creating and solves a maxflow graph using the QPBO implementation Vladimir Kolmogorov. 
        int (int32), float (float32) or double (float64) edge capacities/energies are supported.

        This class requires the ```thinqpbo``` Python package available on PyPi or https://github.com/Skielex/thinqpbo.
        It uses a modified version of QPBO algorithm by Vladimir Kolmogorov availble at https://github.com/Skielex/QPBO.
        The QPBO algorithm uses the BK maxflow implementation, which is a Augmenting-Path type algorithm.
        """
        super().__init__(
            estimated_nodes=estimated_nodes,
            estimated_edges=estimated_edges,
            flow_type=capacity_type,
            capacity_type=capacity_type,
            arc_index_type=np.uint32,
            node_index_type=np.uint32,
            jit_build=jit_build,
        )

    def _add_nodes(self, graph_object):
        return self.graph.add_node(graph_object.data.size)

    def _test_types_and_set_inf_cap(self):
        if self.capacity_type == np.int32:
            self.inf_cap = self.INF_CAP_INT32
        elif self.capacity_type == np.float32:
            self.inf_cap = self.INF_CAP_FLOAT32
        elif self.capacity_type == np.float64:
            self.inf_cap = self.INF_CAP_FLOAT64
        else:
            raise ValueError(
                f"Invalid capacity type '{self.capacity_type}'. Only 'int32', 'float32' and 'float64' allowed.")

    def create_graph_object(self):
        if self.capacity_type == np.int32:
            self.graph = thinqpbo.QPBOInt(self.estimated_nodes, self.estimated_edges)
        elif self.capacity_type == np.float32:
            self.graph = thinqpbo.QPBOFloat(self.estimated_nodes, self.estimated_edges)
        elif self.capacity_type == np.float64:
            self.graph = thinqpbo.QPBODouble(self.estimated_nodes, self.estimated_edges)
        else:
            raise ValueError(
                f"Invalid capacity type '{self.capacity_type}'. Only 'int32', 'float32' and 'float64' allowed.")

    def add_object(self, graph_object):
        if graph_object in self.object_ids:
            # If object is already added, return its id.
            return self.object_ids[graph_object]

        # Add object to graph.
        object_id = len(self.objects)
        self.object_ids[graph_object] = object_id

        if self.graph is None:
            first_id = (np.min(self.nodes[self.objects[-1]]) + self.objects[-1].data.size) if self.objects else 0
        else:
            first_id = self._add_nodes(graph_object)

        self.objects.append(graph_object)
        self.nodes[graph_object] = first_id

        return object_id

    def add_unary_terms(self, i, e0, e1):
        if self.graph is None:
            i, e0, e1 = np.broadcast_arrays(i, e0, e1)

            self.unary_nodes.append(i.ravel().astype(self.node_index_type))
            self.unary_e0.append(e0.ravel().astype(self.capacity_type))
            self.unary_e1.append(e1.ravel().astype(self.capacity_type))
        else:
            np.vectorize(self.graph.add_unary_term, otypes=[bool])(i, e0, e1)

    def add_pairwise_terms(self, i, j, e00, e01, e10, e11):
        if self.graph is None:
            i, j, e00, e01, e10, e11 = np.broadcast_arrays(i, j, e00, e01, e10, e11)

            self.pairwise_from.append(i.ravel().astype(self.node_index_type))
            self.pairwise_to.append(j.ravel().astype(self.node_index_type))
            self.pairwise_e00.append(e00.ravel().astype(self.capacity_type))
            self.pairwise_e01.append(e01.ravel().astype(self.capacity_type))
            self.pairwise_e10.append(e10.ravel().astype(self.capacity_type))
            self.pairwise_e11.append(e11.ravel().astype(self.capacity_type))
        else:
            return np.vectorize(self.graph.add_pairwise_term, otypes=[int])(i, j, e00, e01, e10, e11)

    def get_labels(self, i):
        if isinstance(i, GraphObject):
            return self.get_labels(self.get_nodeids(i))
        return np.vectorize(self.graph.get_label, otypes=[np.int8])(i)

    def solve(self, compute_weak_persistencies=True):
        self.build_graph()
        self.graph.solve()
        if compute_weak_persistencies:
            self.graph.compute_weak_persistencies()
        return self.graph.compute_twice_energy()
