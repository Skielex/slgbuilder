import numpy as np
import thinhpf

from .graphobject import GraphObject
from .slgbuilder import SLGBuilder


class HPFBuilder(SLGBuilder):

    SOURCE_NODE_ID = 0
    SINK_NODE_ID = 1

    def __init__(
        self,
        estimated_nodes=0,
        estimated_edges=0,
        capacity_type=np.int32,
        label_order='hf',
        root_order='lifo',
        jit_build=True,
    ):
        """TODO"""
        self.label_order = label_order
        self.root_order = root_order
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

        # Test if flow type is valid.
        thinhpf.hpf(
            capacity_type=self.capacity_type,
            label_order=self.label_order,
            root_order=self.root_order,
        )

        # Set infinite capacity value.
        self.inf_cap = self.INF_CAP_MAP.get(self.capacity_type.name, None)

        # Check if a value was found.
        if self.inf_cap is None:
            raise ValueError(
                f"Invalid capacity type '{self.capacity_type}'. Supported types are: {', '.join(self.INF_CAP_MAP)}")

    def create_graph_object(self):
        self.graph = thinhpf.hpf(
            self.estimated_nodes,
            self.estimated_edges,
            capacity_type=self.capacity_type,
        )
        self.graph.set_source(self.SOURCE_NODE_ID)
        self.graph.set_sink(self.SINK_NODE_ID)
        # Add source and sink nodes.
        self.graph.add_node(2)

    def add_object(self, graph_object):
        if graph_object in self.objects:
            # If object is already added, return its id.
            return self.objects.index(graph_object)

        # Add object to graph.
        object_id = len(self.objects)
        self.object_ids[graph_object] = object_id

        if self.graph is None:
            first_id = (np.min(self.nodes[-1]) + self.objects[-1].data.size) if self.objects else 2  # Start at two.
        else:
            first_id = self.graph.add_node(graph_object.data.size) - graph_object.data.size

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
            if np.any(e1):
                e1 = np.ascontiguousarray(e1)
                self.graph.add_edges(np.full_like(i, self.SOURCE_NODE_ID), i, e1)
            if np.any(e0):
                e0 = np.ascontiguousarray(e0)
                self.graph.add_edges(i, np.full_like(i, self.SINK_NODE_ID), e0)

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
            if np.any(e01):
                e01 = np.ascontiguousarray(e01)
                self.graph.add_edges(i, j, e01)
            if np.any(e10):
                e10 = np.ascontiguousarray(e10)
                self.graph.add_edges(j, i, e10)

    def get_labels(self, i):
        if isinstance(i, GraphObject):
            return self.get_labels(self.get_nodeids(i))
        return np.vectorize(self.graph.what_label, otypes=[np.uint32])(i)

    def mark_nodes(self, i):
        np.vectorize(self.graph.mark_node, otypes=[bool])(i)

    def solve(self):
        if self.solve_count > 0:
            raise Exception("solve may only be called once.")
        self.build_graph()
        self.graph.mincut()
        flow = self.graph.compute_maxflow()
        self.solve_count += 1
        return flow
