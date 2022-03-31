import numpy as np


class GraphObject:
    def __init__(self, data, sample_points=None, block_ids=None):
        """Creates a GraphObject for storing data and data points.
        """
        self.data = data

        if sample_points is None:
            # If no sample points are provided we assume it's a regular grid.
            sample_points = np.moveaxis(np.indices(data.shape), 0, -1)

        if data.shape != sample_points.shape[:-1]:
            raise ValueError('Shapes of object %s and sample_points %s do not match.' %
                             (data.shape, sample_points.shape[:-1]))

        self.sample_points = sample_points

        if block_ids is not None and data.shape != block_ids.shape:
            raise ValueError('Shapes of object %s and block_ids %s do not match.' % (data.shape, block_ids.shape))

        self.block_ids = block_ids
