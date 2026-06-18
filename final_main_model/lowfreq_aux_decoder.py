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


class LowFrequencyAuxResidualDecoder(nn.Module):
    """Main decoder with low-frequency auxiliary regularization and optional RF gate.

    Main path:
      repeat-aware temporal encoder -> spatial decoder -> main reconstruction logits

    Auxiliary path:
      pooled latent -> linear low-frequency head
      optimized only through an auxiliary loss against a low-pass target

    Optional residual gate:
      pooled latent -> residual decoder -> gated residual logits
      gate can be multiplied by a fixed RF support mask.
    """

    def __init__(
        self,
        n_cells,
        history_bins,
        image_size=(64, 64),
        encoder="multiscale",
        repeat_pool="attention",
        latent_dim=384,
        temporal_channels=96,
        base_channels=96,
        attention_heads=4,
        attention_layers=1,
        dropout=0.25,
        low_aux_sigma=0.0,
        low_aux_weight=0.10,
        low_aux_ssim_weight=0.0,
        residual_gate="rf_scalar",
        residual_weight=0.10,
        gate_l1_weight=0.02,
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
        self.residual_gate = residual_gate
        self.residual_weight = float(residual_weight)
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

        self.main_decoder = AttentionSpatialFrameDecoder(latent_dim, image_size, base_channels, dropout)
        self.low_head = nn.Linear(latent_dim, image_size[0] * image_size[1])
        self.residual_decoder = SpatialFrameDecoder(
            latent_dim,
            image_size,
            max(base_channels // 2, 32),
            dropout,
        )
        self.gate_head = nn.Sequential(
            nn.Linear(latent_dim, max(latent_dim // 2, 32)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(latent_dim // 2, 32), 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(residual_weight))
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
        main_logits = self.main_decoder(latent)
        low_logits = self.low_head(latent).reshape(latent.shape[0], 1, main_logits.shape[-2], main_logits.shape[-1])
        residual_logits = self.residual_decoder(latent)

        if self.residual_gate == "none":
            gate = torch.ones(latent.shape[0], 1, 1, 1, device=latent.device, dtype=latent.dtype)
            gated_residual = torch.tanh(self.residual_scale) * residual_logits
        else:
            gate = torch.sigmoid(self.gate_head(latent)).view(latent.shape[0], 1, 1, 1)
            support = self.rf_support_mask.to(latent.device, latent.dtype) if self.residual_gate == "rf_scalar" else 1.0
            gated_residual = gate * support * torch.tanh(self.residual_scale) * residual_logits

        final_logits = main_logits + gated_residual
        aux = {
            "low_logits": low_logits,
            "main_logits": main_logits,
            "residual_logits": residual_logits,
            "gated_residual": gated_residual,
            "gate": gate,
        }
        return final_logits, aux

    def forward(self, x, return_single=False):
        b, k, h, c = x.shape
        latents = self.encoder(x.reshape(b * k, h, c)).reshape(b, k, -1)
        pooled = self._pool_latents(latents)
        logits, aux = self._decode_from_latent(pooled)
        self._last_aux = aux
        if return_single:
            single_logits = []
            for single_latent in latents.unbind(dim=1):
                s_logits, _ = self._decode_from_latent(single_latent)
                single_logits.append(s_logits.unsqueeze(1))
            single_logits = torch.cat(single_logits, dim=1)
            return logits, single_logits, latents, pooled
        return logits, latents, pooled

    def auxiliary_loss(self, target):
        if self._last_aux is None:
            raise RuntimeError("auxiliary_loss() requires a prior forward pass.")
        low_pred = torch.sigmoid(self._last_aux["low_logits"])
        low_target = gaussian_blur_2d(target, self.low_aux_sigma)
        low_loss = F.mse_loss(low_pred, low_target)
        if self.low_aux_ssim_weight > 0:
            low_loss = low_loss + self.low_aux_ssim_weight * (1.0 - ssim_index(low_pred, low_target).mean())
        gate_reg = self._last_aux["gate"].mean()
        residual_reg = self._last_aux["gated_residual"].abs().mean()
        return self.low_aux_weight * low_loss + self.gate_l1_weight * (gate_reg + residual_reg)

    def auxiliary_summary(self):
        return {
            "low_aux_sigma": self.low_aux_sigma,
            "low_aux_weight": self.low_aux_weight,
            "low_aux_ssim_weight": self.low_aux_ssim_weight,
            "residual_gate": self.residual_gate,
            "residual_weight": self.residual_weight,
            "gate_l1_weight": self.gate_l1_weight,
        }
