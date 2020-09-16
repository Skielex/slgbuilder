import numpy as np
import shrdr

from .graphobject import GraphObject
from .slgbuilder import SLGBuilder


class PQPBOBuilder(SLGBuilder):

    def __init__(self, estimated_nodes=0, estimated_edges=0, flow_type=np.int32, jit_build=True, num_threads=-1):
        """TODO
        """
        self.num_threads = num_threads
        super().__init__(estimated_nodes=estimated_nodes, estimated_edges=estimated_edges, flow_type=flow_type, jit_build=jit_build)

    def _add_nodes(self, graph_object):
        return self.graph.add_node(graph_object.data.size, self.objects.index(graph_object))

    def _set_flow_type_and_inf_cap(self, flow_type):
        if flow_type == np.int32:
            self.flow_type = np.int32
            self.inf_cap = self.INF_CAP_INT32
        elif flow_type == np.float32:
            self.flow_type = np.float32
            self.inf_cap = self.INF_CAP_FLOAT32
        else:
            raise ValueError("Invalid flow_type '%s'. Only 'int32' and 'float32' allowed.")

    def create_graph_object(self):
        if self.flow_type == np.int32:
            self.graph = shrdr.ParallelQpboInt(self.estimated_nodes, self.estimated_edges, expect_nonsubmodular=True, expected_blocks=len(self.objects))
        elif self.flow_type == np.float32:
            self.graph = shrdr.ParallelQpboFloat(self.estimated_nodes, self.estimated_edges, expect_nonsubmodular=True)
        else:
            raise ValueError("Invalid flow_type '%s'. Only 'int32' and 'float32' allowed.")

        if self.num_threads > 0:
            self.graph.set_num_threads(self.num_threads)

    def add_object(self, graph_object):
        if graph_object in self.objects:
            # If object is already added, return its id.
            return self.objects.index(graph_object)

        # Add object to graph.
        object_id = len(self.objects)

        if self.jit_build:
            first_id = (np.min(self.nodes[-1]) + self.objects[-1].data.size) if self.objects else 0
        else:
            first_id = self._add_nodes(graph_object)

        self.objects.append(graph_object)
        self.nodes.append(first_id)

        return object_id

    def add_unary_terms(self, i, e0, e1):
        i, e0, e1 = self.broadcast_unary_terms(i, e0, e1)

        if self.graph is None:
            self.unary_nodes.append(i)
            self.unary_e0.append(e0)
            self.unary_e1.append(e1)
        else:
            i = np.ascontiguousarray(i)
            e0 = np.ascontiguousarray(e0)
            e1 = np.ascontiguousarray(e1)
            self.graph.add_unary_terms(i, e0, e1)

    def add_pairwise_terms(self, i, j, e00, e01, e10, e11):
        i, j, e00, e01, e10, e11 = self.broadcast_pairwise_terms(i, j, e00, e01, e10, e11)

        if self.graph is None:
            self.pairwise_from.append(i)
            self.pairwise_to.append(j)
            self.pairwise_e00.append(e00)
            self.pairwise_e01.append(e01)
            self.pairwise_e10.append(e10)
            self.pairwise_e11.append(e11)
        else:
            i = np.ascontiguousarray(i)
            j = np.ascontiguousarray(j)
            e00 = np.ascontiguousarray(e00)
            e01 = np.ascontiguousarray(e01)
            e10 = np.ascontiguousarray(e10)
            e11 = np.ascontiguousarray(e11)
            self.graph.add_pairwise_terms(i, j, e00, e01, e10, e11)

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

