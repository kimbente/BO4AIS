import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models.model import Model


class DistanceWeightedAcquisition(AnalyticAcquisitionFunction):
    """
    Distance-weighted acquisition function for batch BO.
    Works with q > 1 (batch) and penalizes close candidate points.

    α(X) = mean[ base_acq(x_i) * w_i ]
    where w_i = f(distance to other candidates)
    """

    def __init__(self, model: Model, base_acq: AnalyticAcquisitionFunction, lengthscale: float = 0.1):
        super().__init__(model=model)
        self.base_acq = base_acq
        self.lengthscale = lengthscale  # controls spatial weighting

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: shape (batch_shape, q, d)
        """
        q = X.shape[-2]
        base_vals = self.base_acq(X)  # shape (batch_shape, q)

        # Compute pairwise distances within the batch
        # shape: (q, q)
        dist_matrix = torch.cdist(X.squeeze(0), X.squeeze(0), p=2)

        # Distance-based weights (e.g. Gaussian penalty)
        # Small distances → small weight
        # w_ij = exp(-d_ij / l)
        weights = torch.exp(-dist_matrix / self.lengthscale)

        # Each point's weight = average distance to others (excluding itself)
        # Avoid self-weighting (set diagonal to 0)
        weights.fill_diagonal_(0.0)
        w_i = 1 - weights.mean(dim=-1)  # shape (q,)

        # Normalize weights
        w_i = w_i / w_i.max()

        # Combine with base acquisition
        weighted_acq = base_vals.squeeze(-1) * w_i
        return weighted_acq.mean()  # scalar for batch evaluation
