import torch

def ps_to_norm(X, x_min = 1_300_000, x_max = 2_700_000, y_min = - 1_600_000, y_max = - 200_000):
    """
    Normalize polar stereographic coordinates (x, y) into [0, 1] box domain.

    Parameters
    ----------
    X : torch.Tensor
        Shape [batch, 2], with columns [x, y].
    x_min, x_max, y_min, y_max : float
        Bounding box limits for the PS projection.

    Returns
    -------
    X_norm : torch.Tensor
        Normalized coordinates of shape [batch, 2] in [0, 1].
    """
    # unpack
    x, y = X[:, 0], X[:, 1]

    # linear rescale to [0, 1]
    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = (y - y_min) / (y_max - y_min)

    X_norm = torch.stack([x_norm, y_norm], dim = -1)
    return X_norm


def norm_to_ps(X_norm, x_min = 1_300_000, x_max = 2_700_000, y_min = - 1_600_000, y_max = - 200_000):
    """
    Denormalize [0, 1] box-domain coordinates back to polar stereographic (x, y).

    Parameters
    ----------
    X_norm : torch.Tensor
        Shape [batch, 2], with columns [x_norm, y_norm] in [0, 1].
    x_min, x_max, y_min, y_max : float
        Bounding box limits for the PS projection.

    Returns
    -------
    X : torch.Tensor
        Denormalized coordinates of shape [batch, 2], with columns [x, y].
    """
    # unpack
    x_norm, y_norm = X_norm[:, 0], X_norm[:, 1]

    # invert linear rescaling
    x = x_norm * (x_max - x_min) + x_min
    y = y_norm * (y_max - y_min) + y_min

    X = torch.stack([x, y], dim = -1)
    return X

def ps_to_speed(X, 
                x_min = 1_298_150, x_max = 2_701_700, 
                y_max = - 1_601_900, y_min = - 198_350):
    """
    Project polar stereographic coordinates (x, y) into [-1, 1] box domain
    for use with grid_sample interpolation (speed grid domain).
    NOTE:
    grid_samples assumes y = -1 is the top row so y_max and y_min were swapped.

    Parameters
    ----------
    X : torch.Tensor
        Shape [batch, 2], with columns [x, y] in polar stereographic coords.
    x_min, x_max, y_min, y_max : float
        Bounding box limits for the speed grid.

    Returns
    -------
    X_grid : torch.Tensor
        Normalized coordinates of shape [batch, 2] in [-1, 1].
    """
    # unpack
    x, y = X[:, 0], X[:, 1]

    # linear rescale to [-1, 1]
    x_grid = 2 * (x - x_min) / (x_max - x_min) - 1
    y_grid = 2 * (y - y_min) / (y_max - y_min) - 1

    X_grid = torch.stack([x_grid, y_grid], dim=-1)
    return X_grid