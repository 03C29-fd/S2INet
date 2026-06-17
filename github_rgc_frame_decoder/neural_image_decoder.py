"""Neural Image Decoder: RF-template guided reconstruction from RGC responses.

Pipeline:
  spike response [B, repeats, history, n_cells]
  → average repeats → weighted by Gaussian RF templates → Neural Image [B, history, H, W]
  → UNet decoder → reconstructed frame [B, 1, H, W]

Loss modes:
  A) ssim_mse:    L = 0.8 * (1-SSIM) + 0.2 * RF-weighted MSE
  B) ssim_grad:   L = 0.7 * (1-SSIM) + 0.2 * RF-MSE + 0.1 * gradient_loss
  C) adversarial: B + PatchGAN discriminator (hinge loss)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rf_templates(rf_centers, image_size, sigma=3.0, source_size=None):
    """Build per-cell 2D Gaussian RF templates.

    Args:
        rf_centers: [n_cells, 2] as (row, col) in source coordinates
        image_size: int, output frame side length
        sigma: float, Gaussian sigma in pixel units
        source_size: scalar, original coordinate scale. If None, assumes same as image_size.

    Returns:
        templates: [n_cells, 1, image_size, image_size]
    """
    centers = torch.as_tensor(rf_centers, dtype=torch.float32)
    n_cells = centers.shape[0]
    H = W = image_size
    yy = torch.arange(H, dtype=torch.float32)
    xx = torch.arange(W, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    if source_size is not None:
        scale = float(image_size) / float(source_size)
        centers = centers * scale
    templates = torch.zeros(n_cells, 1, H, W)
    sigma = max(float(sigma), 1e-6)
    for i, (row, col) in enumerate(centers):
        dist2 = (grid_y - float(row)) ** 2 + (grid_x - float(col)) ** 2
        g = torch.exp(-0.5 * dist2 / (sigma * sigma))
        g = g / g.sum().clamp_min(1e-8)
        templates[i, 0] = g
    return templates


def response_to_neural_image(x, templates):
    """Convert spike responses to neural image.

    Args:
        x: [B, repeats, history_bins, n_cells] or [B, history_bins, n_cells]
        templates: [n_cells, 1, H, W]

    Returns:
        neural_image: [B, history_bins, H, W]
    """
    if x.ndim == 4:
        x = x.mean(dim=1)  # average repeats → [B, history, cells]
    templates_flat = templates.squeeze(1)  # [n_cells, H, W]
    neural = torch.einsum("btc,chw->bthw", x, templates_flat)
    return neural


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class NeuralImageUNet(nn.Module):
    def __init__(self, in_channels, image_size=64, base_channels=64, dropout=0.15):
        super().__init__()
        self.image_size = image_size
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)
        self.bottleneck = nn.Sequential(
            nn.Dropout2d(dropout),
            DoubleConv(base_channels * 8, base_channels * 8),
        )
        self.dec4 = UpBlock(base_channels * 16, base_channels * 4)
        self.dec3 = UpBlock(base_channels * 8, base_channels * 2)
        self.dec2 = UpBlock(base_channels * 4, base_channels)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, 1, 3, padding=1),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        out = self.dec1(torch.cat([d2, e1], dim=1))
        if out.shape[-2:] != (self.image_size, self.image_size):
            out = F.interpolate(out, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return out


class NeuralImageFrameDecoder(nn.Module):
    """RF-template guided frame decoder.

    Compatible with existing RGCFrameDecoder interface:
    forward(x, return_single=False) → (logits, latents, pooled)
    forward(x, return_single=True) → (logits, single_logits, latents, pooled)
    """

    def __init__(self, n_cells, history_bins, image_size=64, rf_centers=None, rf_sigma=3.0,
                 base_channels=64, dropout=0.15, rf_source_size=None):
        super().__init__()
        self.n_cells = n_cells
        self.history_bins = history_bins
        self.image_size = image_size
        if rf_centers is None:
            rf_centers = torch.rand(n_cells, 2) * image_size
        templates = build_rf_templates(rf_centers, image_size, sigma=rf_sigma, source_size=rf_source_size)
        self.register_buffer("rf_templates", templates)
        self.unet = NeuralImageUNet(
            in_channels=history_bins,
            image_size=image_size,
            base_channels=base_channels,
            dropout=dropout,
        )

    def _neural_image_from_response(self, x):
        return response_to_neural_image(x, self.rf_templates)

    def forward(self, x, return_single=False):
        B, K, H_bins, C = x.shape
        pooled_response = x.mean(dim=1)
        neural = self._neural_image_from_response(pooled_response)
        logits = self.unet(neural)
        dummy = torch.zeros(B, 1, device=logits.device)

        if return_single:
            single_logits = []
            for k in range(K):
                single_neural = self._neural_image_from_response(x[:, k])
                single_logits.append(self.unet(single_neural).unsqueeze(1))
            single_logits = torch.cat(single_logits, dim=1)
            return logits, single_logits, dummy, dummy
        return logits, dummy, dummy


class PatchGANDiscriminator(nn.Module):
    """70×70 PatchGAN discriminator."""

    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 8, 1, 4, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = torch.outer(g, g)
    return window.expand(channels, 1, window_size, window_size).contiguous()


def ssim_map(x, y, data_range=1.0, window_size=11, sigma=1.5):
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
    return ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))


def ssim_loss(pred, target):
    return 1.0 - ssim_map(pred, target).flatten(1).mean(dim=1).mean()


def gradient_loss(pred, target):
    """Sobel-based gradient (edge) L1 loss."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    gx_pred = F.conv2d(pred, sobel_x.expand(pred.shape[1], 1, 3, 3), padding=1, groups=pred.shape[1])
    gy_pred = F.conv2d(pred, sobel_y.expand(pred.shape[1], 1, 3, 3), padding=1, groups=pred.shape[1])
    gx_true = F.conv2d(target, sobel_x.expand(target.shape[1], 1, 3, 3), padding=1, groups=target.shape[1])
    gy_true = F.conv2d(target, sobel_y.expand(target.shape[1], 1, 3, 3), padding=1, groups=target.shape[1])
    return F.l1_loss(gx_pred, gx_true) + F.l1_loss(gy_pred, gy_true)


def rf_weighted_mse(pred, target, rf_weight):
    """RF-weighted MSE (spatially modulated)."""
    w = rf_weight.to(pred.device, pred.dtype)
    w_norm = w.mean().clamp_min(1e-8)
    return (((pred - target) ** 2) * w).mean() / w_norm


def rf_weighted_gradient_loss(pred, target, rf_weight):
    """RF-weighted gradient loss."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    gx_pred = F.conv2d(pred, sobel_x.expand(pred.shape[1], 1, 3, 3), padding=1, groups=pred.shape[1])
    gy_pred = F.conv2d(pred, sobel_y.expand(pred.shape[1], 1, 3, 3), padding=1, groups=pred.shape[1])
    gx_true = F.conv2d(target, sobel_x.expand(target.shape[1], 1, 3, 3), padding=1, groups=target.shape[1])
    gy_true = F.conv2d(target, sobel_y.expand(target.shape[1], 1, 3, 3), padding=1, groups=target.shape[1])
    w = rf_weight.to(pred.device, pred.dtype)
    w_norm = w.mean().clamp_min(1e-8)
    gx_diff = ((gx_pred - gx_true).abs() * w).mean() / w_norm
    gy_diff = ((gy_pred - gy_true).abs() * w).mean() / w_norm
    return gx_diff + gy_diff


class MixedSSIMMSELoss(nn.Module):
    """L = 0.8 * (1-SSIM) + 0.2 * RF-weighted MSE."""

    def __init__(self, rf_weight, ssim_weight=0.8, mse_weight=0.2):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.mse_weight = mse_weight
        self.register_buffer("rf_weight", rf_weight)

    def forward(self, pred, target):
        l_ssim = ssim_loss(pred, target)
        l_mse = rf_weighted_mse(pred, target, self.rf_weight)
        return self.ssim_weight * l_ssim + self.mse_weight * l_mse


class SSIMGradientLoss(nn.Module):
    """L = 0.7 * (1-SSIM) + 0.2 * RF-MSE + 0.1 * gradient_loss."""

    def __init__(self, rf_weight, ssim_weight=0.7, mse_weight=0.2, grad_weight=0.1):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.mse_weight = mse_weight
        self.grad_weight = grad_weight
        self.register_buffer("rf_weight", rf_weight)

    def forward(self, pred, target):
        l_ssim = ssim_loss(pred, target)
        l_mse = rf_weighted_mse(pred, target, self.rf_weight)
        l_grad = rf_weighted_gradient_loss(pred, target, self.rf_weight)
        return self.ssim_weight * l_ssim + self.mse_weight * l_mse + self.grad_weight * l_grad


class AdversarialReconstructionLoss(nn.Module):
    """GAN-based loss: L_recon + λ_adv * L_hinge.

    Call `discriminator_loss(real, fake)` separately for D update.
    """

    def __init__(self, rf_weight, ssim_weight=0.7, mse_weight=0.2, grad_weight=0.1,
                 adv_weight=0.01):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.mse_weight = mse_weight
        self.grad_weight = grad_weight
        self.adv_weight = adv_weight
        self.register_buffer("rf_weight", rf_weight)

    def reconstruction_loss(self, pred, target):
        l_ssim = ssim_loss(pred, target)
        l_mse = rf_weighted_mse(pred, target, self.rf_weight)
        l_grad = rf_weighted_gradient_loss(pred, target, self.rf_weight)
        return self.ssim_weight * l_ssim + self.mse_weight * l_mse + self.grad_weight * l_grad

    def generator_loss(self, pred, target, discriminator):
        l_recon = self.reconstruction_loss(pred, target)
        fake_logits = discriminator(pred)
        l_adv = F.relu(1.0 - fake_logits).mean()
        return l_recon + self.adv_weight * l_adv

    def forward(self, pred, target):
        return self.reconstruction_loss(pred, target)


def discriminator_loss(discriminator, real, fake):
    """Hinge loss for PatchGAN discriminator."""
    real_logits = discriminator(real)
    fake_logits = discriminator(fake.detach())
    l_real = F.relu(1.0 - real_logits).mean()
    l_fake = F.relu(1.0 + fake_logits).mean()
    return (l_real + l_fake) * 0.5
