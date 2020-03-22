import numpy as np
from ortools.graph import pywrapgraph

from .graphobject import GraphObject
from .slgbuilder import SLGBuilder


class ORBuilder(SLGBuilder):

    def __init__(self, estimated_nodes=0, estimated_edges=0, flow_type=np.int64, jit_build=True):
        """Creates a helper for creating and solves a maxflow graph using the Google OR-Tools maxflow implementation. 
        Only integer edge capacities/energies are supported.

        This class requires the ```ortools``` Python package by Google available on PyPi or https://github.com/google/or-tools.
        The Google OR-Tools maxflow implementation is a Push-Relabel type algorithm.
        """
        self.source_nodeids = []
        self.sink_nodeids = []
        super().__init__(estimated_nodes=estimated_nodes, estimated_edges=estimated_edges, flow_type=flow_type, jit_build=jit_build)

    def _set_flow_type_and_inf_cap(self, flow_type):
        if np.issubdtype(flow_type, np.integer):
            self.flow_type = np.int64
            self.inf_cap = self.INF_CAP_INT64
        else:
            raise ValueError("Invalid flow_type '%s'. Only 'integer' allowed." % str(flow_type))

    def _add_nodes(self, graph_object):
        pass

    def create_graph_object(self):
        if np.issubdtype(self.flow_type, np.integer):
            self.graph = pywrapgraph.SimpleMaxFlow()
        else:
            raise ValueError("Invalid flow_type '%s'. Only 'integer' allowed." % str(flow_type))

    def add_object(self, graph_object, pack_nodes=False):
        if graph_object in self.objects:
            # If object is already added, return its id.
            return self.objects.index(graph_object)

        # Add object to graph.
        object_id = len(self.objects)

        first_id = (np.min(self.nodes[-1]) + self.objects[-1].data.size) if self.objects else 2

        self.objects.append(graph_object)
        self.nodes.append(first_id)

        if pack_nodes:
            self.nodes[-1] = self.pack_object_nodes(graph_object)

        return object_id

    def add_pairwise_terms(self, i, j, e00, e01, e10, e11):
        # TODO: Warn that e00 and e11 are ignores.
        self.add_edges(i, j, e01, e10)

    def add_edges(self, i, j, cap, rcap):
        i, j, e00, e01, e10, e11 = np.broadcast_arrays(i, j, 0, cap, rcap, 0)

        if self.graph is None:
            self.pairwise_from.append(i.flatten().astype(np.int32))
            self.pairwise_to.append(j.flatten().astype(np.int32))
            self.pairwise_e00.append(e00.flatten().astype(self.flow_type))
            self.pairwise_e01.append(e01.flatten().astype(self.flow_type))
            self.pairwise_e10.append(e10.flatten().astype(self.flow_type))
            self.pairwise_e11.append(e11.flatten().astype(self.flow_type))
        else:
            e01_mask = e01 > 0
            e10_mask = e10 > 0

            if np.any(e01_mask):
                [self.graph.AddArcWithCapacity(i, j, c) for i, j, c in zip(
                    i[e01_mask].tolist(), j[e01_mask].tolist(), e01[e01_mask].tolist())]

            if np.any(e10_mask):
                [self.graph.AddArcWithCapacity(j, i, c) for i, j, c in zip(
                    i[e10_mask].tolist(), j[e10_mask].tolist(), e10[e10_mask].tolist())]

    def add_unary_terms(self, i, e0, e1):
        self.add_terminal_edges(i, e1, e0)

    def add_terminal_edges(self, i, source_cap, sink_cap):
        i, e0, e1 = np.broadcast_arrays(i, sink_cap, source_cap)

        if self.graph is None:
            self.unary_nodes.append(i.flatten().astype(np.int32))
            self.unary_e0.append(e0.flatten())
            self.unary_e1.append(e1.flatten())
        else:
            e0_mask = e0 > 0
            e1_mask = e1 > 0

            if np.any(e1_mask):
                [self.graph.AddArcWithCapacity(0, i, c) for i, c in zip(i[e1_mask].tolist(), e1[e1_mask].tolist())]
            if np.any(e0_mask):
                [self.graph.AddArcWithCapacity(i, 1, c) for i, c in zip(i[e0_mask].tolist(), e0[e0_mask].tolist())]

    def get_labels(self, i):
        if isinstance(i, GraphObject):
            return self.get_labels(self.get_nodeids(i))
        elif np.isscalar(i):
            return not np.any(i == self.source_nodeids)
        else:
            return ~np.in1d(i, self.source_nodeids).reshape(i.shape)

    def solve(self):
        self.build_graph()
        if self.graph.Solve(0, 1) == self.graph.OPTIMAL:
            self.source_nodeids = np.asarray(self.graph.GetSourceSideMinCut())
            self.sink_nodeids = np.asarray(self.graph.GetSinkSideMinCut())
            return self.graph.OptimalFlow()

        return None
