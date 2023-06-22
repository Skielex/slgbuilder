import numpy as np
import shrdr

from .graphobject import GraphObject
from .slgbuilder import SLGBuilder


class PBKBuilder(SLGBuilder):
    def __init__(
        self,
        estimated_nodes=0,
        estimated_edges=0,
        expected_blocks=0,
        capacity_type=np.int32,
        arc_index_type=np.uint32,
        node_index_type=np.uint32,
        jit_build=True,
        num_threads=-1,
    ):
        """TODO"""
        flow_type = np.int64 if np.issubdtype(capacity_type, np.integer) else np.float64
        self.num_threads = num_threads
        self.merge_edges = False
        self.expected_blocks = expected_blocks
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
            first_node_id = self.graph.add_node(graph_object.data.size, self.objects.index(graph_object))

        return first_node_id

    def _test_types_and_set_inf_cap(self):

        # Test if flow type is valid.
        shrdr.parallel_bk(
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

        self.graph = shrdr.parallel_bk(
            self.estimated_nodes,
            self.estimated_edges,
            expected_blocks=self.expected_blocks,
            capacity_type=self.capacity_type,
            arc_index_type=self.arc_index_type,
            node_index_type=self.node_index_type,
        )

        if self.num_threads > 0:
            self.graph.set_num_threads(self.num_threads)

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
            first_id = self._add_nodes(graph_object)

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
            self.graph.add_edges(i, j, e01, e10, self.merge_edges)

    def get_labels(self, i):
        return self.what_segments(i)

    def what_segments(self, i):
        if isinstance(i, GraphObject):
            return self.what_segments(self.get_nodeids(i))
        return np.vectorize(self.graph.what_segment, otypes=[np.int8])(i)

    def solve(self):
        self.build_graph()
        flow = self.graph.maxflow()
        self.solve_count += 1
        return flow
