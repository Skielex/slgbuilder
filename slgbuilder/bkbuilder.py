import numpy as np
import thinmaxflow

from .graphobject import GraphObject
from .slgbuilder import SLGBuilder


class BKBuilder(SLGBuilder):
    def __init__(self, estimated_nodes=0, estimated_edges=0, capacity_type=np.int32, jit_build=True):
        """Creates a helper for creating and solves a maxflow graph using the Boykov-Kolmogorov (BK) maxflow implementation. 
        int (int32), short (int16), float (float32) or double (float64) edge capacities/energies are supported.

        This class requires the ```thinmaxflow``` Python package available on PyPi or https://github.com/Skielex/thinmaxflow.
        It uses a modified version of Maxflow algorithm by Yuri Boykov and Vladimir Kolmogorov availble at https://github.com/Skielex/maxflow.
        The BK maxflow implementation is a Augmenting-Path type algorithm.
        """
        flow_type = capacity_type if np.issubdtype(capacity_type, np.floating) else np.int32
        super().__init__(
            estimated_nodes=estimated_nodes,
            estimated_edges=estimated_edges,
            flow_type=flow_type,
            capacity_type=capacity_type,
            arc_index_type=np.int64,
            node_index_type=np.int32,
            jit_build=jit_build,
        )

    def _test_types_and_set_inf_cap(self):
        if self.capacity_type == np.int16:
            self.inf_cap = self.INF_CAP_INT16
        elif self.capacity_type == np.int32:
            self.inf_cap = self.INF_CAP_INT32
        elif self.capacity_type == np.float32:
            self.inf_cap = self.INF_CAP_FLOAT32
        elif self.capacity_type == np.float64:
            self.inf_cap = self.INF_CAP_FLOAT64
        else:
            raise ValueError(
                f"Invalid capacity type '{self.capacity_type}'. Only 'int16', 'int32', 'float32' and 'float64' allowed."
            )

    def _add_nodes(self, graph_object):
        return self.graph.add_node(graph_object.data.size)

    def create_graph_object(self):
        if self.capacity_type == np.int16:
            self.graph = thinmaxflow.GraphShort(self.estimated_nodes, self.estimated_edges)
        elif self.capacity_type == np.int32:
            self.graph = thinmaxflow.GraphInt(self.estimated_nodes, self.estimated_edges)
        elif self.capacity_type == np.float32:
            self.graph = thinmaxflow.GraphFloat(self.estimated_nodes, self.estimated_edges)
        elif self.capacity_type == np.float64:
            self.graph = thinmaxflow.GraphDouble(self.estimated_nodes, self.estimated_edges)
        else:
            raise ValueError(
                f"Invalid capacity type '{self.capacity_type}'. Only 'int16', 'int32', 'float32' and 'float64' allowed."
            )

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
            first_id = self.graph.add_node(graph_object.data.size)

        self.objects.append(graph_object)
        self.nodes[graph_object] = first_id

        return object_id

    def add_pairwise_terms(self, i, j, e00, e01, e10, e11):
        # TODO: Warn that e00 and e11 are ignores.
        self.add_edges(i, j, e01, e10)

    def add_edges(self, i, j, cap, rcap):
        if self.graph is None:
            i, j, e00, e01, e10, e11 = np.broadcast_arrays(i, j, 0, cap, rcap, 0)

            self.pairwise_from.append(i.ravel().astype(self.node_index_type))
            self.pairwise_to.append(j.ravel().astype(self.node_index_type))
            self.pairwise_e00.append(e00.ravel().astype(self.capacity_type))
            self.pairwise_e01.append(e01.ravel().astype(self.capacity_type))
            self.pairwise_e10.append(e10.ravel().astype(self.capacity_type))
            self.pairwise_e11.append(e11.ravel().astype(self.capacity_type))
        else:
            np.vectorize(self.graph.add_edge, otypes=[bool])(i, j, cap, rcap)

    def add_unary_terms(self, i, e0, e1):
        self.add_terminal_edges(i, e1, e0)

    def add_terminal_edges(self, i, source_cap, sink_cap):
        if self.graph is None:
            i, e0, e1 = np.broadcast_arrays(i, sink_cap, source_cap)

            self.unary_nodes.append(i.ravel().astype(self.node_index_type))
            self.unary_e0.append(e0.ravel().astype(self.capacity_type))
            self.unary_e1.append(e1.ravel().astype(self.capacity_type))
        else:
            np.vectorize(self.graph.add_tweights, otypes=[bool])(i, source_cap, sink_cap)

    def get_labels(self, i):
        return self.what_segments(i)

    def what_segments(self, i):
        if isinstance(i, GraphObject):
            return self.what_segments(self.get_nodeids(i))
        return np.vectorize(self.graph.what_segment, otypes=[bool])(i)

    def mark_nodes(self, i):
        np.vectorize(self.graph.mark_node, otypes=[bool])(i)

    def solve(self):
        self.build_graph()
        return self.graph.maxflow()
