import numpy as np
import shrdr

from .graphobject import GraphObject
from .slgbuilder import SLGBuilder


class MBKBuilder(SLGBuilder):
    def __init__(
        self,
        estimated_nodes=0,
        estimated_edges=0,
        capacity_type=np.int32,
        arc_index_type=np.uint32,
        node_index_type=np.uint32,
        jit_build=True,
    ):
        """TODO"""
        flow_type = np.int64 if np.issubdtype(capacity_type, np.integer) else np.float64
        self.merge_edges = False
        super().__init__(
            estimated_nodes=estimated_nodes,
            estimated_edges=estimated_edges,
            flow_type=flow_type,
            capacity_type=capacity_type,
            arc_index_type=arc_index_type,
            node_index_type=node_index_type,
            jit_build=jit_build,
        )

    def _add_nodes(self, graph_object):
        return self.graph.add_node(graph_object.data.size)

    def _test_types_and_set_inf_cap(self):

        # Test if flow type is valid.
        shrdr.bk(
            capacity_type=self.capacity_type,
            arc_index_type=self.arc_index_type,
            node_index_type=self.node_index_type,
        )

        # Set infinite capacity value.
        self.inf_cap = self.INF_CAP_MAP.get(self.capacity_type.name, None)

        # Check if a value was found.
        if self.inf_cap is None:
            raise ValueError(
                f"Invalid capacity type '{self.capacity_type}'. Supported types are: {', '.join(self.INF_CAP_MAP)}")

    def create_graph_object(self):
        self.graph = shrdr.bk(
            self.estimated_nodes,
            self.estimated_edges,
            capacity_type=self.capacity_type,
            arc_index_type=self.arc_index_type,
            node_index_type=self.node_index_type,
        )

    def add_object(self, graph_object):
        if graph_object in self.objects:
            # If object is already added, return its id.
            return self.objects.index(graph_object)

        # Add object to graph.
        object_id = len(self.objects)
        self.object_ids[graph_object] = object_id

        if self.graph is None:
            first_id = (np.min(self.nodes[-1]) + self.objects[-1].data.size) if self.objects else 0
        else:
            first_id = self.graph.add_node(graph_object.data.size)

        self.objects.append(graph_object)
        self.nodes[graph_object] = first_id

        return object_id

    def add_terminal_edges(self, i, source_cap, sink_cap):
        self.add_unary_terms(i, sink_cap, source_cap)

    def add_unary_terms(self, i, e0, e1):
        i, e0, e1 = self.broadcast_terms([i], [e0, e1])

        if self.graph is None:
            self.unary_nodes.append(i)
            self.unary_e0.append(e0)
            self.unary_e1.append(e1)
        else:
            i = np.ascontiguousarray(i)
            e0 = np.ascontiguousarray(e0)
            e1 = np.ascontiguousarray(e1)
            if self.solve_count > 0:
                self.graph.mark_nodes(i)
            self.graph.add_tweights(i, e1, e0)

    def add_edges(self, i, j, cap, rcap):
        self.add_pairwise_terms(i, j, 0, cap, rcap, 0)

    def add_pairwise_terms(self, i, j, e00, e01, e10, e11):
        # TODO: Warn that e00 and e11 are ignores.
        i, j, e01, e10 = self.broadcast_terms([i, j], [e01, e10])

        if self.graph is None:
            self.pairwise_from.append(i)
            self.pairwise_to.append(j)
            self.pairwise_e01.append(e01)
            self.pairwise_e10.append(e10)
        else:
            i = np.ascontiguousarray(i)
            j = np.ascontiguousarray(j)
            e01 = np.ascontiguousarray(e01)
            e10 = np.ascontiguousarray(e10)
            if self.solve_count > 0:
                self.graph.mark_nodes(i)
                self.graph.mark_nodes(j)
            self.graph.add_edges(i, j, e01, e10, self.merge_edges)

    def get_labels(self, i):
        return self.what_segments(i)

    def what_segments(self, i):
        if isinstance(i, GraphObject):
            return self.what_segments(self.get_nodeids(i))
        return np.vectorize(self.graph.what_segment, otypes=[np.int8])(i)

    def mark_nodes(self, i):
        np.vectorize(self.graph.mark_node, otypes=[bool])(i)

    def solve(self, reuse_trees=True):
        self.build_graph()
        flow = self.graph.maxflow(reuse_trees and self.solve_count > 0)
        self.solve_count += 1
        return flow
