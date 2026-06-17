import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_RF_POINTS_32 = [(4, 6), (7, 5), (20, 30), (2, 10), (3, 16)]


def make_rf_weight_map(height, width, rf_points=DEFAULT_RF_POINTS_32, source_size=32, device=None, dtype=None):
    weight = torch.ones((height, width), device=device, dtype=dtype)
    for row, col in rf_points:
        y = int(round(row / source_size * height))
        x = int(round(col / source_size * width))
        y = max(0, min(height - 1, y))
        x = max(0, min(width - 1, x))
        for dy, dx, value in [(0, 0, 5.0), (-1, 0, 3.0), (1, 0, 3.0), (0, -1, 3.0), (0, 1, 3.0)]:
            yy = max(0, min(height - 1, y + dy))
            xx = max(0, min(width - 1, x + dx))
            weight[yy, xx] = value
    return weight.view(1, 1, height, width)


def make_gaussian_rf_weight_map(
    height,
    width,
    rf_centers,
    source_size=None,
    sigma=3.0,
    baseline=1.0,
    peak=5.0,
    normalize_mean=True,
    device=None,
    dtype=None,
):
    """Build the paper-style RF weight matrix from one RF center per neuron.

    Args:
        height, width: output image size.
        rf_centers: iterable/tensor shaped [n_cells, 2], as (row, col) in source coordinates.
        source_size: scalar or (height, width) for rf_centers coordinates. If None, uses output size.
        sigma: Gaussian sigma in source-coordinate pixels.
        baseline: base weight outside RFs.
        peak: approximate peak contribution scale after normalizing summed Gaussians to max 1.
        normalize_mean: normalize final weight map to mean 1 so loss scale stays stable.
    """
    centers = torch.as_tensor(rf_centers, device=device, dtype=dtype or torch.float32)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(f"rf_centers must have shape [n_cells, 2], got {tuple(centers.shape)}")

    if source_size is None:
        source_h, source_w = float(height), float(width)
    elif isinstance(source_size, (tuple, list)):
        source_h, source_w = float(source_size[0]), float(source_size[1])
    else:
        source_h = source_w = float(source_size)

    yy = torch.arange(height, device=device, dtype=centers.dtype)
    xx = torch.arange(width, device=device, dtype=centers.dtype)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    grid_y = grid_y / max(height - 1, 1) * max(source_h - 1.0, 1.0)
    grid_x = grid_x / max(width - 1, 1) * max(source_w - 1.0, 1.0)

    rf_sum = torch.zeros((height, width), device=device, dtype=centers.dtype)
    sigma = max(float(sigma), 1e-6)
    for row, col in centers:
        dist2 = (grid_y - row).pow(2) + (grid_x - col).pow(2)
        rf_sum = rf_sum + torch.exp(-0.5 * dist2 / (sigma * sigma))

    rf_sum = rf_sum / rf_sum.max().clamp_min(1e-8)
    weight = baseline + (peak - baseline) * rf_sum
    if normalize_mean:
        weight = weight / weight.mean().clamp_min(1e-8)
    return weight.view(1, 1, height, width)


def gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = torch.outer(g, g)
    return window.expand(channels, 1, window_size, window_size).contiguous()


def _ssim_map(x, y, data_range=1.0, window_size=11, sigma=1.5):
    channels = x.shape[1]
    window = gaussian_window(window_size, sigma, channels, x.device, x.dtype)
    padding = window_size // 2
    mu_x = F.conv2d(x, window, padding=padding, groups=channels)
    mu_y = F.conv2d(y, window, padding=padding, groups=channels)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, window, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    return ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )


def ssim_map_2d(x, y, data_range=1.0, window_size=11, sigma=1.5):
    return _ssim_map(x, y, data_range, window_size, sigma)


def ssim_index(x, y, data_range=1.0, window_size=11, sigma=1.5):
    return _ssim_map(x, y, data_range, window_size, sigma).flatten(1).mean(dim=1)


class AdaptiveReconstructionLoss(nn.Module):
    def __init__(
        self,
        mode="mse_ssim",
        image_size=(64, 64),
        ssim_weight=0.05,
        weighted_mse_weight=0.0,
    ):
        super().__init__()
        self.mode = mode
        self.ssim_weight = ssim_weight
        self.weighted_mse_weight = weighted_mse_weight
        weight_map = make_rf_weight_map(image_size[0], image_size[1], dtype=torch.float32)
        self.register_buffer("weight_map", weight_map)

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        if self.mode == "mse":
            return mse

        loss = mse
        if self.mode in {"weighted_mse", "mse_ssim", "weighted_mse_ssim"} and self.weighted_mse_weight > 0:
            weighted_mse = (((pred - target) ** 2) * self.weight_map.to(pred.device, pred.dtype)).mean()
            loss = loss + self.weighted_mse_weight * weighted_mse

        if self.mode in {"mse_ssim", "weighted_mse_ssim"} and self.ssim_weight > 0:
            loss = loss + self.ssim_weight * (1.0 - ssim_index(pred, target).mean())

        return loss
