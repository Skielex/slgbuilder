import numpy as np

class GraphObject:
    def __init__(self, data, sample_points=None):
        """Creates a GraphObject for storing data and data points.
        """
        self.data = data

        if sample_points is None:
            # If no sample points are provided we assume it's a regular grid.
            sample_points = np.moveaxis(np.indices(data.shape), 0, -1)

        if data.shape != sample_points.shape[:-1]:
            raise ValueError('Shapes of object %s and sample_points %s do not match.' % (data.shape, sample_points.shape[:-1]))

        self.sample_points = sample_points