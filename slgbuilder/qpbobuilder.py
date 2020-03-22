import numpy as np
import thinqpbo

from .graphobject import GraphObject
from .slgbuilder import SLGBuilder


class QPBOBuilder(SLGBuilder):

    def __init__(self, estimated_nodes=0, estimated_edges=0, flow_type=np.int32, jit_build=True):
        """Creates a helper for creating and solves a maxflow graph using the QPBO implementation Vladimir Kolmogorov. 
        int (int32), float (float32) or double (float64) edge capacities/energies are supported.

        This class requires the ```thinqpbo``` Python package available on PyPi or https://github.com/Skielex/thinqpbo.
        It uses a modified version of QPBO algorithm by Vladimir Kolmogorov availble at https://github.com/Skielex/QPBO.
        The QPBO algorithm uses the BK maxflow implementation, which is a Augmenting-Path type algorithm.
        """
        super().__init__(estimated_nodes=estimated_nodes, estimated_edges=estimated_edges, flow_type=flow_type, jit_build=jit_build)

    def _add_nodes(self, graph_object):
        return self.graph.add_node(graph_object.data.size)

    def _set_flow_type_and_inf_cap(self, flow_type):
        if flow_type == np.int32:
            self.flow_type = np.int32
            self.inf_cap = self.INF_CAP_INT32
        elif flow_type == np.float32:
            self.flow_type = np.float32
            self.inf_cap = self.INF_CAP_FLOAT32
        elif flow_type == np.float64:
            self.flow_type = np.float64
            self.inf_cap = self.INF_CAP_FLOAT64
        else:
            raise ValueError("Invalid flow_type '%s'. Only 'int32', 'float32' and 'float64' allowed.")

    def create_graph_object(self):
        if self.flow_type == np.int32:
            self.graph = thinqpbo.QPBOInt(self.estimated_nodes, self.estimated_edges)
        elif self.flow_type == np.float32:
            self.graph = thinqpbo.QPBOFloat(self.estimated_nodes, self.estimated_edges)
        elif self.flow_type == np.float64:
            self.graph = thinqpbo.QPBODouble(self.estimated_nodes, self.estimated_edges)
        else:
            raise ValueError("Invalid flow_type '%s'. Only 'int32', 'float32' and 'float64' allowed.")

    def add_object(self, graph_object, pack_nodes=False):
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

        if pack_nodes:
            self.nodes[-1] = self.pack_object_nodes(graph_object)

        return object_id

    def add_unary_terms(self, i, e0, e1):
        if self.graph is None:
            i, e0, e1 = np.broadcast_arrays(i, e0, e1)

            self.unary_nodes.append(i.flatten().astype(np.int32))
            self.unary_e0.append(e0.flatten().astype(self.flow_type))
            self.unary_e1.append(e1.flatten().astype(self.flow_type))
        else:
            np.vectorize(self.graph.add_unary_term, otypes=[np.bool])(i, e0, e1)

    def add_pairwise_terms(self, i, j, e00, e01, e10, e11):
        if self.graph is None:
            i, j, e00, e01, e10, e11 = np.broadcast_arrays(i, j, e00, e01, e10, e11)

            self.pairwise_from.append(i.flatten().astype(np.int32))
            self.pairwise_to.append(j.flatten().astype(np.int32))
            self.pairwise_e00.append(e00.flatten().astype(self.flow_type))
            self.pairwise_e01.append(e01.flatten().astype(self.flow_type))
            self.pairwise_e10.append(e10.flatten().astype(self.flow_type))
            self.pairwise_e11.append(e11.flatten().astype(self.flow_type))
        else:
            return np.vectorize(self.graph.add_pairwise_term, otypes=[np.int])(i, j, e00, e01, e10, e11)

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
