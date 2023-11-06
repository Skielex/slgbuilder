import numpy as np
import shrdr

from .graphobject import GraphObject
from .slgbuilder import SLGBuilder


class PQPBOBuilder(SLGBuilder):
    def __init__(
        self,
        estimated_nodes=0,
        estimated_edges=0,
        expected_blocks=0,
        expect_nonsubmodular=True,
        capacity_type=np.int32,
        arc_index_type=np.uint32,
        node_index_type=np.uint32,
        jit_build=True,
        num_threads=-1,
    ):
        """TODO"""
        flow_type = np.int64 if np.issubdtype(capacity_type, np.integer) else np.float64
        self.num_threads = num_threads
        self.expected_blocks = expected_blocks
        self.expect_nonsubmodular = expect_nonsubmodular,
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
        if graph_object.block_ids is not None:
            if graph_object.data.shape != graph_object.block_ids.shape:
                raise ValueError('Shapes of object data %s and block_ids %s do not match.' %
                                 (graph_object.data.shape, graph_object.block_ids.shape))

            block_ids_flat = graph_object.block_ids.ravel()
            change_indices = np.nonzero(np.diff(block_ids_flat))[0]
            change_indices += 1
            change_indices = np.concatenate([[0], change_indices])
            change_counts = np.diff(change_indices)
            change_counts = np.concatenate([change_counts, [block_ids_flat.size - change_indices[-1]]])
            change_block_ids = block_ids_flat[change_indices]

            add_node_fn = np.vectorize(self.graph.add_node, otypes=[self.node_index_type])
            first_node_id = add_node_fn(change_counts, change_block_ids)[0]
        else:
            first_node_id = self.graph.add_node(graph_object.data.size, self.object_ids[graph_object])

        return first_node_id

    def _test_types_and_set_inf_cap(self):

        # Test if flow type is valid.
        shrdr.qpbo(
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
        if self.expected_blocks == 0:
            object_max_block_ids = [np.max(go.block_ids) for go in self.objects if go.block_ids is not None]
            if object_max_block_ids and len(object_max_block_ids) != len(self.objects):
                raise ValueError(
                    "Some objects are missing block_ids. Either all graph objects should have block_ids set or none of them should."
                )
            self.expected_blocks = max(object_max_block_ids) + 1 if object_max_block_ids else len(self.objects)

        self.graph = shrdr.parallel_qpbo(
            self.estimated_nodes,
            self.estimated_edges,
            expect_nonsubmodular=self.expect_nonsubmodular,
            expected_blocks=self.expected_blocks,
            capacity_type=self.capacity_type,
            arc_index_type=self.arc_index_type,
            node_index_type=self.node_index_type,
        )

        if self.num_threads > 0:
            self.graph.set_num_threads(self.num_threads)

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
        i, e0, e1 = self.broadcast_terms([i], [e0, e1])

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
        i, j, e00, e01, e10, e11 = self.broadcast_terms([i, j], [e00, e01, e10, e11])

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
        self.solved = True
        if compute_weak_persistencies:
            self.graph.compute_weak_persistencies()
        return self.graph.get_flow()
