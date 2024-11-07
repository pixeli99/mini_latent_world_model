import torch

class MSEDist:
    def __init__(self, mode, dims, agg="sum"):
        """
        Custom MSE-based distribution.
        
        Args:
            mode (Tensor): Predicted mode (mean) values.
            dims (int): Number of dimensions to consider as the event shape.
            agg (str): How to aggregate the MSE loss ("sum" or "mean").
        """
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._agg = agg
        self.batch_shape = mode.shape[: len(mode.shape) - dims]
        self.event_shape = mode.shape[len(mode.shape) - dims :]

    def mode(self):
        """Return the mode (mean) of the distribution."""
        return self._mode

    def mean(self):
        """Return the mean of the distribution."""
        return self._mode

    def log_prob(self, value):
        """Compute the negative MSE as the log probability."""
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(f"Aggregation method '{self._agg}' is not implemented.")
        return -loss  # Return negative MSE as the log probability