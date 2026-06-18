import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_loss import ssim_index
from neural_image_decoder import build_rf_templates
from wisa_model import (
    AttentionSpatialFrameDecoder,
    AttentionTemporalEncoder,
    MultiScaleTemporalEncoder,
    SpatialFrameDecoder,
)


def gaussian_blur_2d(image, sigma):
    if sigma is None or float(sigma) <= 0:
        return image
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    coords = torch.arange(kernel_size, dtype=image.dtype, device=image.device) - kernel_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = torch.outer(g, g).view(1, 1, kernel_size, kernel_size)
    kernel = kernel.expand(image.shape[1], 1, kernel_size, kernel_size)
    return F.conv2d(image, kernel, padding=kernel_size // 2, groups=image.shape[1])


def sobel_gradients(image):
    kx = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3) / 4.0
    ky = torch.tensor(
        [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3) / 4.0
    gx = F.conv2d(image, kx, padding=1)
    gy = F.conv2d(image, ky, padding=1)
    return gx, gy


def laplacian_response(image):
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3)
    return F.conv2d(image, kernel, padding=1)


def build_rf_support_mask(rf_centers, image_size, rf_sigma=3.0, source_size=40):
    if rf_centers is None:
        return torch.ones(1, 1, image_size, image_size, dtype=torch.float32)
    templates = build_rf_templates(
        rf_centers,
        image_size=image_size,
        sigma=rf_sigma,
        source_size=source_size,
    )
    support = templates.sum(dim=0, keepdim=True)
    support = support / support.max().clamp_min(1e-8)
    return support


class DetailResidualFrameDecoder(nn.Module):
    """Coarse reconstruction plus spike-conditioned residual detail path."""

    outputs_are_images = True

    def __init__(
        self,
        n_cells,
        history_bins,
        image_size=(64, 64),
        encoder="multiscale",
        repeat_pool="attention",
        latent_dim=384,
        temporal_channels=128,
        base_channels=96,
        attention_heads=4,
        attention_layers=1,
        dropout=0.25,
        base_frame=None,
        base_frame_weight=0.35,
        low_aux_sigma=2.0,
        low_aux_weight=0.12,
        low_aux_ssim_weight=0.02,
        edge_loss_weight=0.08,
        laplacian_loss_weight=0.05,
        highfreq_loss_weight=0.08,
        highfreq_sigma=2.0,
        residual_gate="rf_scalar",
        residual_weight=0.15,
        gate_l1_weight=0.01,
        rf_centers=None,
        rf_sigma=3.0,
        rf_source_size=40,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.history_bins = history_bins
        self.repeat_pool = repeat_pool
        self.low_aux_sigma = float(low_aux_sigma)
        self.low_aux_weight = float(low_aux_weight)
        self.low_aux_ssim_weight = float(low_aux_ssim_weight)
        self.edge_loss_weight = float(edge_loss_weight)
        self.laplacian_loss_weight = float(laplacian_loss_weight)
        self.highfreq_loss_weight = float(highfreq_loss_weight)
        self.highfreq_sigma = float(highfreq_sigma)
        self.residual_gate = residual_gate
        self.gate_l1_weight = float(gate_l1_weight)

        if encoder == "attention":
            self.encoder = AttentionTemporalEncoder(
                n_cells,
                latent_dim=latent_dim,
                channels=temporal_channels,
                dropout=dropout,
                num_heads=attention_heads,
                num_layers=attention_layers,
            )
        else:
            self.encoder = MultiScaleTemporalEncoder(n_cells, latent_dim, temporal_channels, dropout)

        if repeat_pool == "attention":
            self.repeat_attention = nn.Sequential(nn.Linear(latent_dim, 1), nn.Softmax(dim=1))
        else:
            self.repeat_attention = None

        self.coarse_decoder = AttentionSpatialFrameDecoder(latent_dim, image_size, base_channels, dropout)
        self.residual_decoder = SpatialFrameDecoder(latent_dim, image_size, max(base_channels // 2, 32), dropout)
        self.gate_head = nn.Sequential(
            nn.Linear(latent_dim, max(latent_dim // 2, 32)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(latent_dim // 2, 32), 1),
        )
        self.base_mix_logit = nn.Parameter(torch.tensor(math.log(base_frame_weight / max(1.0 - base_frame_weight, 1e-6))))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_weight)))

        if base_frame is None:
            base_frame = torch.zeros(1, 1, image_size[0], image_size[1], dtype=torch.float32)
        self.register_buffer("base_frame", base_frame.float().clone())
        self.register_buffer(
            "rf_support_mask",
            build_rf_support_mask(rf_centers, image_size[0], rf_sigma=rf_sigma, source_size=rf_source_size),
        )
        self._last_aux = None

    def _pool_latents(self, latents):
        if self.repeat_attention is None:
            return latents.mean(dim=1)
        weights = self.repeat_attention(latents)
        return (latents * weights).sum(dim=1)

    def _decode_from_latent(self, latent):
        coarse = torch.sigmoid(self.coarse_decoder(latent))
        prior_mix = torch.sigmoid(self.base_mix_logit)
        base = prior_mix * self.base_frame.to(latent.device, latent.dtype) + (1.0 - prior_mix) * coarse

        residual_raw = torch.tanh(self.residual_decoder(latent))
        residual_scale = torch.tanh(self.residual_scale)
        if self.residual_gate == "none":
            gate = torch.ones(latent.shape[0], 1, 1, 1, device=latent.device, dtype=latent.dtype)
            support = 1.0
        else:
            gate = torch.sigmoid(self.gate_head(latent)).view(latent.shape[0], 1, 1, 1)
            support = self.rf_support_mask.to(latent.device, latent.dtype)
        gated_residual = gate * support * residual_scale * residual_raw
        pred = (base + gated_residual).clamp(0.0, 1.0)
        aux = {
            "coarse": coarse,
            "base": base,
            "gated_residual": gated_residual,
            "gate": gate,
            "prior_mix": prior_mix.detach(),
            "pred": pred,
        }
        return pred, aux

    def forward(self, x, return_single=False):
        b, k, h, c = x.shape
        latents = self.encoder(x.reshape(b * k, h, c)).reshape(b, k, -1)
        pooled = self._pool_latents(latents)
        pred, aux = self._decode_from_latent(pooled)
        self._last_aux = aux
        if return_single:
            single_pred = []
            for single_latent in latents.unbind(dim=1):
                s_pred, _ = self._decode_from_latent(single_latent)
                single_pred.append(s_pred.unsqueeze(1))
            single_pred = torch.cat(single_pred, dim=1)
            return pred, single_pred, latents, pooled
        return pred, latents, pooled

    def detail_loss(self, target):
        if self._last_aux is None:
            raise RuntimeError("detail_loss() requires a prior forward pass.")
        pred = self._last_aux["pred"]
        base = self._last_aux["base"]
        gate = self._last_aux["gate"]
        gated_residual = self._last_aux["gated_residual"]

        low_target = gaussian_blur_2d(target, self.low_aux_sigma)
        low_loss = F.mse_loss(base, low_target)
        if self.low_aux_ssim_weight > 0:
            low_loss = low_loss + self.low_aux_ssim_weight * (1.0 - ssim_index(base, low_target).mean())

        pred_gx, pred_gy = sobel_gradients(pred)
        target_gx, target_gy = sobel_gradients(target)
        edge_loss = F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)

        lap_loss = F.l1_loss(laplacian_response(pred), laplacian_response(target))

        pred_high = pred - gaussian_blur_2d(pred, self.highfreq_sigma)
        target_high = target - gaussian_blur_2d(target, self.highfreq_sigma)
        highfreq_loss = F.l1_loss(pred_high, target_high) + 0.5 * F.l1_loss(gated_residual, target_high)

        gate_reg = gate.mean() + 0.5 * gated_residual.abs().mean()
        return (
            self.low_aux_weight * low_loss
            + self.edge_loss_weight * edge_loss
            + self.laplacian_loss_weight * lap_loss
            + self.highfreq_loss_weight * highfreq_loss
            + self.gate_l1_weight * gate_reg
        )

    def auxiliary_summary(self):
        return {
            "base_frame_weight": float(torch.sigmoid(self.base_mix_logit).detach().cpu()),
            "low_aux_sigma": self.low_aux_sigma,
            "low_aux_weight": self.low_aux_weight,
            "low_aux_ssim_weight": self.low_aux_ssim_weight,
            "edge_loss_weight": self.edge_loss_weight,
            "laplacian_loss_weight": self.laplacian_loss_weight,
            "highfreq_loss_weight": self.highfreq_loss_weight,
            "highfreq_sigma": self.highfreq_sigma,
            "residual_gate": self.residual_gate,
            "gate_l1_weight": self.gate_l1_weight,
        }
