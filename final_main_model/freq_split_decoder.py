"""Frequency-Split Decoder: low-frequency ridge + gated residual reconstruction.

Pipeline:
  response [B, repeats, history, n_cells]
  → Low-Freq Path (Ridge Regression) → frame_low in [0, 1]
  → Mid-Freq Path (Temporal CNN + Decoder) → residual_mid
  → High-Freq Path (small CNN decoder) → residual_high
  → gate = sigmoid(gate_head(latent))
  → frame = clamp(low + mid + gate * high, 0, 1)

Evaluation:
  RF coverage mask for per-region metrics (rf_covered / non_rf_covered / global)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_blur_2d(image, sigma, kernel_size=None):
    """Apply Gaussian blur to a batch of images."""
    if sigma is None or float(sigma) <= 0:
        return image
    if kernel_size is None:
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
    coords = torch.arange(kernel_size, dtype=image.dtype, device=image.device) - kernel_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = torch.outer(g, g).view(1, 1, kernel_size, kernel_size)
    kernel = kernel.expand(image.shape[1], 1, kernel_size, kernel_size)
    return F.conv2d(image, kernel, padding=kernel_size // 2, groups=image.shape[1])


def sobel_gradient(image):
    """Return Sobel x/y gradients for grayscale image batches."""
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    return F.conv2d(image, kx, padding=1), F.conv2d(image, ky, padding=1)


def gaussian_blur_2d_np(image, sigma, kernel_size=None):
    """NumPy Gaussian blur for fitting (no GPU needed)."""
    if sigma is None or float(sigma) <= 0:
        return image
    import scipy.ndimage
    return scipy.ndimage.gaussian_filter(image, sigma=sigma, mode="reflect")


def make_rf_radius_mask(rf_centers, image_size, source_size=40, radius=8.0,
                        softness=2.0, dtype=torch.float32):
    """Soft high-frequency support mask around RF centers."""
    centers = torch.as_tensor(rf_centers, dtype=dtype)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(f"rf_centers must have shape [n_cells,2], got {tuple(centers.shape)}")
    h = w = int(image_size)
    if isinstance(source_size, (tuple, list)):
        source_h, source_w = float(source_size[0]), float(source_size[1])
    else:
        source_h = source_w = float(source_size)
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=dtype),
        torch.arange(w, dtype=dtype),
        indexing="ij",
    )
    yy = yy / max(h - 1, 1) * max(source_h - 1.0, 1.0)
    xx = xx / max(w - 1, 1) * max(source_w - 1.0, 1.0)
    radius = float(radius)
    softness = max(float(softness), 1e-6)
    mask = torch.zeros(h, w, dtype=dtype)
    for row, col in centers:
        dist = torch.sqrt((yy - row).pow(2) + (xx - col).pow(2))
        mask = torch.maximum(mask, torch.sigmoid((radius - dist) / softness))
    return mask.view(1, 1, h, w)


class LowFreqRidge:
    """Ridge regression from averaged responses to Gaussian-blurred frames."""

    def __init__(self, alpha=1.0, blur_sigma=2.0):
        self.alpha = alpha
        self.blur_sigma = blur_sigma
        self.W = None
        self.bias = None

    def fit(self, responses_mean, frames, device="cpu"):
        """Fit Ridge regression.

        Args:
            responses_mean: [N, n_features] — averaged spike responses (flattened)
            frames: [N, H, W] — ground truth frames
            device: torch device for computation
        """
        N = responses_mean.shape[0]
        H, W = frames.shape[1], frames.shape[2]

        frames_np = frames.cpu().numpy() if isinstance(frames, torch.Tensor) else frames
        low_frames = np.stack([
            gaussian_blur_2d_np(f, self.blur_sigma) for f in frames_np
        ], axis=0)
        flat_target = low_frames.reshape(N, -1).astype(np.float64)

        X = responses_mean.cpu().numpy().astype(np.float64) if isinstance(responses_mean, torch.Tensor) else responses_mean.astype(np.float64)

        X_mean = X.mean(axis=0, keepdims=True)
        X_centered = X - X_mean
        Y_mean = flat_target.mean(axis=0, keepdims=True)
        Y_centered = flat_target - Y_mean

        XTX = X_centered.T @ X_centered
        reg = self.alpha * np.eye(XTX.shape[0])
        W = np.linalg.solve(XTX + reg, X_centered.T @ Y_centered)

        self.W = torch.from_numpy(W.astype(np.float32)).to(device)
        self.bias = torch.from_numpy((Y_mean - X_mean @ W).astype(np.float32)).to(device)
        self._x_mean = torch.from_numpy(X_mean.astype(np.float32)).to(device)

    def predict(self, responses_mean):
        """Predict low-frequency frame component.

        Args:
            responses_mean: [B, n_features]

        Returns:
            low_frame: [B, 1, H, W] image-space values. These are not logits.
        """
        if self.W is None:
            raise RuntimeError("LowFreqRidge must be fit() before predict().")
        x = responses_mean.to(self.W.device, dtype=self.W.dtype)
        pred = x @ self.W + self.bias
        B = x.shape[0]
        side = int(np.sqrt(pred.shape[-1]))
        return pred.reshape(B, 1, side, side)


class LowFreqRFTemplateRidge:
    """Ridge model constrained to a per-cell RF-template image basis."""

    def __init__(self, rf_centers, image_size=64, source_size=40, rf_sigma=3.0,
                 blur_sigma=2.0, alpha=1.0, template_alpha=1e-3):
        self.rf_centers = rf_centers
        self.image_size = image_size
        self.source_size = source_size
        self.rf_sigma = rf_sigma
        self.blur_sigma = blur_sigma
        self.alpha = alpha
        self.template_alpha = template_alpha
        self.templates = None
        self.W = None
        self.bias = None

    def _build_templates(self, device=None):
        from neural_image_decoder import build_rf_templates

        templates = build_rf_templates(
            self.rf_centers,
            self.image_size,
            sigma=self.rf_sigma,
            source_size=self.source_size,
        ).squeeze(1)
        templates = templates / templates.flatten(1).norm(dim=1).clamp_min(1e-8).view(-1, 1, 1)
        return templates.to(device=device, dtype=torch.float32)

    def fit(self, responses_mean, frames, device="cpu"):
        n = responses_mean.shape[0]
        frames_np = frames.cpu().numpy() if isinstance(frames, torch.Tensor) else frames
        low_frames = np.stack([gaussian_blur_2d_np(f, self.blur_sigma) for f in frames_np], axis=0)
        y = low_frames.reshape(n, -1).astype(np.float64)

        templates = self._build_templates(device="cpu")
        basis = templates.flatten(1).numpy().astype(np.float64)  # [cells, pixels]
        gram = basis @ basis.T
        coeff = y @ basis.T @ np.linalg.inv(gram + self.template_alpha * np.eye(gram.shape[0]))

        x = responses_mean.cpu().numpy().astype(np.float64) if isinstance(responses_mean, torch.Tensor) else responses_mean.astype(np.float64)
        x_mean = x.mean(axis=0, keepdims=True)
        x_centered = x - x_mean
        c_mean = coeff.mean(axis=0, keepdims=True)
        c_centered = coeff - c_mean
        xtx = x_centered.T @ x_centered
        w = np.linalg.solve(xtx + self.alpha * np.eye(xtx.shape[0]), x_centered.T @ c_centered)

        self.templates = templates.to(device)
        self.W = torch.from_numpy(w.astype(np.float32)).to(device)
        self.bias = torch.from_numpy((c_mean - x_mean @ w).astype(np.float32)).to(device)

    def predict(self, responses_mean):
        if self.W is None or self.templates is None:
            raise RuntimeError("LowFreqRFTemplateRidge must be fit() before predict().")
        x = responses_mean.to(self.W.device, dtype=self.W.dtype)
        coeff = x @ self.W + self.bias
        basis = self.templates.flatten(1)
        pred = coeff @ basis
        return pred.reshape(x.shape[0], 1, self.image_size, self.image_size)


class FreqSplitDecoder(nn.Module):
    """Complete frequency-split reconstruction model.

    Low-freq path: pre-fitted Ridge regression in image space (frozen).
    Mid/high paths: temporal encoder + gated residual decoders (trainable).
    """

    def __init__(self, n_cells, history_bins, image_size=64,
                 temporal_channels=128, latent_dim=256, base_channels=64,
                 dropout=0.25, blur_sigma=2.0, ridge_alpha=1.0,
                 low_freq_ridge=None, pretrain_epochs=5, pretrain_lr=1e-3,
                 low_mode="ridge", rf_centers=None, rf_source_size=40,
                 rf_sigma=3.0, template_alpha=1e-3, hybrid_template_weight=0.5,
                 high_rf_radius=0.0, high_rf_softness=2.0):
        super().__init__()
        self.outputs_are_images = True
        self.n_cells = n_cells
        self.history_bins = history_bins
        self.image_size = image_size
        self.blur_sigma = blur_sigma
        self.low_mode = low_mode
        self.hybrid_template_weight = float(hybrid_template_weight)
        self.high_rf_radius = float(high_rf_radius)

        if low_freq_ridge is not None:
            self.low_freq = low_freq_ridge
        else:
            self.low_freq = LowFreqRidge(alpha=ridge_alpha, blur_sigma=blur_sigma)
        if low_mode in {"rf_template", "hybrid"} and rf_centers is not None:
            self.template_low_freq = LowFreqRFTemplateRidge(
                rf_centers,
                image_size=image_size,
                source_size=rf_source_size,
                rf_sigma=rf_sigma,
                blur_sigma=blur_sigma,
                alpha=ridge_alpha,
                template_alpha=template_alpha,
            )
        else:
            self.template_low_freq = None

        from wisa_model import MultiScaleTemporalEncoder, SpatialFrameDecoder
        self.encoder = MultiScaleTemporalEncoder(
            n_cells, latent_dim=latent_dim, channels=temporal_channels, dropout=dropout
        )
        self.mid_decoder = SpatialFrameDecoder(
            latent_dim=latent_dim, image_size=(image_size, image_size),
            base_channels=base_channels, dropout=dropout,
        )
        self.high_decoder = SpatialFrameDecoder(
            latent_dim=latent_dim, image_size=(image_size, image_size),
            base_channels=max(base_channels // 2, 16), dropout=dropout,
        )
        self.gate_head = nn.Sequential(
            nn.Linear(latent_dim, max(latent_dim // 2, 32)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(latent_dim // 2, 32), 1),
        )
        self.mid_scale = nn.Parameter(torch.tensor(0.15))
        self.high_scale = nn.Parameter(torch.tensor(0.05))
        if rf_centers is not None and self.high_rf_radius > 0:
            high_mask = make_rf_radius_mask(
                rf_centers,
                image_size=image_size,
                source_size=rf_source_size,
                radius=high_rf_radius,
                softness=high_rf_softness,
            )
            self.high_mask_style = "soft_rf_radius"
        else:
            high_mask = torch.ones(1, 1, image_size, image_size, dtype=torch.float32)
            self.high_mask_style = "global"
        self.register_buffer("high_rf_mask", high_mask)

        self._low_fitted = False
        self._pretrain_epochs = pretrain_epochs
        self._pretrain_lr = pretrain_lr
        self._last_components = None

    def fit_low_freq(self, train_dataset, device="cpu"):
        """Gather training data and fit low-frequency Ridge model."""
        n_samples = len(train_dataset)
        if n_samples == 0:
            raise ValueError("train_dataset is empty.")

        n_feat = self.n_cells * self.history_bins
        all_responses = np.zeros((n_samples, n_feat), dtype=np.float64)
        all_frames = np.zeros((n_samples, self.image_size, self.image_size), dtype=np.float32)
        for i in range(n_samples):
            x, y, _ = train_dataset[i]
            all_responses[i] = x.mean(dim=0).flatten().numpy().astype(np.float64)
            all_frames[i] = y[0].numpy()

        self.low_freq.fit(
            torch.from_numpy(all_responses).float(),
            torch.from_numpy(all_frames).float(),
            device=device,
        )
        if self.template_low_freq is not None:
            self.template_low_freq.fit(
                torch.from_numpy(all_responses).float(),
                torch.from_numpy(all_frames).float(),
                device=device,
            )
        self._low_fitted = True

    def pretrain_high_path(self, train_dataset, device="cuda", batch_size=64):
        from torch.utils.data import DataLoader
        epochs = self._pretrain_epochs
        if epochs <= 0:
            return
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.encoder.to(device)
        self.mid_decoder.to(device)
        self.high_decoder.to(device)
        self.gate_head.to(device)
        optimizer = torch.optim.AdamW(
            list(self.encoder.parameters())
            + list(self.mid_decoder.parameters())
            + list(self.high_decoder.parameters())
            + list(self.gate_head.parameters()),
            lr=self._pretrain_lr, weight_decay=1e-3,
        )
        self.train()
        for epoch in range(epochs):
            total_loss = 0.0
            count = 0
            for x, y, _ in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                B, K, H_bins, C = x.shape
                pooled = x.mean(dim=1).reshape(B, -1)
                pred, _, _ = self(x)
                loss = F.mse_loss(pred, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * y.shape[0]
                count += y.shape[0]
            avg_loss = total_loss / max(count, 1)
            print(f"  [Pretrain High Path] epoch {epoch+1}/{epochs} MSE: {avg_loss:.6f}")

    def _pooled_response(self, x):
        B, K, H_bins, C = x.shape
        return x.mean(dim=1).reshape(B, -1)

    def _decode_components(self, response_window):
        B = response_window.shape[0]
        flat = response_window.reshape(B, -1)
        ridge_low = self.low_freq.predict(flat).to(response_window.device, response_window.dtype)
        if self.template_low_freq is None:
            low = ridge_low
            template_low = None
        else:
            template_low = self.template_low_freq.predict(flat).to(response_window.device, response_window.dtype)
            if self.low_mode == "rf_template":
                low = template_low
            elif self.low_mode == "hybrid":
                w = float(np.clip(self.hybrid_template_weight, 0.0, 1.0))
                low = (1.0 - w) * ridge_low + w * template_low
            else:
                low = ridge_low
        low = low.clamp(0.0, 1.0)
        latent = self.encoder(response_window)
        mid_raw = self.mid_decoder(latent)
        high_raw = self.high_decoder(latent)
        mid = torch.tanh(self.mid_scale) * torch.tanh(mid_raw)
        mask = self.high_rf_mask.to(response_window.device, response_window.dtype)
        high = torch.tanh(self.high_scale) * torch.tanh(high_raw) * mask
        gate = torch.sigmoid(self.gate_head(latent)).view(B, 1, 1, 1)
        pred = (low + mid + gate * high).clamp(0.0, 1.0)
        components = {
            "low": low,
            "ridge_low": ridge_low.clamp(0.0, 1.0),
            "template_low": template_low.clamp(0.0, 1.0) if template_low is not None else None,
            "mid": mid,
            "high": high,
            "gated_high": gate * high,
            "gate": gate,
            "high_mask": mask,
            "latent": latent,
        }
        return pred, components

    def forward(self, x, return_single=False):
        B, K, H_bins, C = x.shape
        pooled = x.mean(dim=1)  # [B, history, cells]

        pred, components = self._decode_components(pooled)
        self._last_components = components

        if return_single:
            single_preds = []
            single_latents = []
            for k in range(K):
                s_pooled = x[:, k]
                s_pred, s_components = self._decode_components(s_pooled)
                single_preds.append(s_pred.unsqueeze(1))
                single_latents.append(s_components["latent"].unsqueeze(1))
            single_preds = torch.cat(single_preds, dim=1)
            single_latents = torch.cat(single_latents, dim=1)
            return pred, single_preds, single_latents, components["latent"]
        return pred, components["latent"], components["latent"]

    def frequency_loss(self, pred, target):
        """SSIM-aware loss for image-space low/mid/high decomposition."""
        from adaptive_loss import ssim_index, ssim_map_2d

        if self._last_components is None:
            raise RuntimeError("frequency_loss() requires a prior forward pass.")
        low = self._last_components["low"]
        mid = self._last_components["mid"]
        gated_high = self._last_components["gated_high"]
        gate = self._last_components["gate"]
        high_mask = self._last_components["high_mask"]

        if self.blur_sigma <= 0:
            target_low = target
            target_mid = torch.zeros_like(target)
            target_high = torch.zeros_like(target)
        else:
            target_low = gaussian_blur_2d(target, sigma=self.blur_sigma)
            target_mid_base = gaussian_blur_2d(target, sigma=max(self.blur_sigma * 0.5, 1.0))
            target_mid = target_mid_base - target_low
            target_high = target - target_mid_base

        ssim_loss = 1.0 - ssim_index(pred, target).mean()
        low_loss = F.mse_loss(low, target_low)
        mid_loss = F.l1_loss(mid, target_mid)

        difficulty = (1.0 - ssim_map_2d(low.detach(), target).detach()).clamp(0.0, 2.0)
        high_weight = (0.25 + difficulty) * gate.detach() * high_mask.detach()
        high_loss = (high_weight * (gated_high - target_high).abs()).mean()
        residual_reg = mid.abs().mean() + gated_high.abs().mean()

        pred_gx, pred_gy = sobel_gradient(pred)
        target_gx, target_gy = sobel_gradient(target)
        grad_loss = F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)

        gate_loss = gate.mean() + 0.1 * gate.square().mean()
        return (
            0.45 * ssim_loss
            + 0.25 * low_loss
            + 0.15 * mid_loss
            + 0.05 * high_loss
            + 0.05 * grad_loss
            + 0.03 * residual_reg
            + 0.02 * gate_loss
        )

    def get_freq_components(self, x):
        """Return low and high frequency components separately (for analysis)."""
        pred = self.forward(x)[0]
        components = self._last_components
        return components["low"], pred


def build_rf_coverage_mask(rf_centers, image_size, sigma=3.0, threshold=None):
    """Build binary RF coverage mask from RF centers.

    Args:
        rf_centers: [n_cells, 2]
        image_size: int
        sigma: Gaussian sigma
        threshold: if None, use median of non-zero values

    Returns:
        mask: [1, 1, H, W] float tensor
    """
    centers = torch.as_tensor(rf_centers, dtype=torch.float32)
    H = W = image_size
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    coverage = torch.zeros(H, W)
    sigma = max(float(sigma), 1e-6)
    for row, col in centers:
        dist2 = (yy - float(row)) ** 2 + (xx - float(col)) ** 2
        coverage = coverage + torch.exp(-0.5 * dist2 / (sigma * sigma))
    coverage = coverage / coverage.max().clamp_min(1e-8)

    if threshold is None:
        nonzero = coverage[coverage > 0.01 * coverage.max()]
        threshold = nonzero.median().item() if len(nonzero) > 0 else 0.5

    mask = (coverage >= threshold).float()
    return mask.view(1, 1, H, W)


def compute_masked_metrics(pred, target, mask):
    """Compute metrics within a binary mask.

    Args:
        pred: [B, 1, H, W]
        target: [B, 1, H, W]
        mask: [1, 1, H, W]

    Returns:
        dict with mse, psnr, ssim
    """
    m = mask.to(pred.device, pred.dtype)
    pred_m = pred * m
    target_m = target * m
    n_pixels = m.sum().clamp_min(1)

    mse_val = float(((pred_m - target_m) ** 2).sum() / n_pixels)

    from adaptive_loss import ssim_index
    ssim_val = float(ssim_index(pred_m, target_m).mean())

    from train_rgc_frame_decoder import psnr_from_mse
    psnr_val = float(psnr_from_mse(mse_val))

    return {"mse": mse_val, "psnr": psnr_val, "ssim": ssim_val}


def evaluate_masked(model, dataset, device, rf_centers, image_size=64, rf_sigma=3.0,
                    batch_size=64):
    """Evaluate with both RF-covered and global metrics."""
    from torch.utils.data import DataLoader
    from adaptive_loss import ssim_index
    from train_rgc_frame_decoder import psnr_from_mse

    mask = build_rf_coverage_mask(rf_centers, image_size, sigma=rf_sigma)
    mask = mask.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    total_rf = {"mse": 0.0, "ssim": 0.0, "n": 0}
    total_global = {"mse": 0.0, "ssim": 0.0, "n": 0}

    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            raw = model(x)[0]
            pred = raw if getattr(model, "outputs_are_images", False) else torch.sigmoid(raw)

            rm = compute_masked_metrics(pred, y, mask)
            total_rf["mse"] += rm["mse"] * y.shape[0]
            total_rf["ssim"] += rm["ssim"] * y.shape[0]
            total_rf["n"] += y.shape[0]

            mse_global = float(F.mse_loss(pred, y))
            ssim_global = float(ssim_index(pred, y).mean())
            total_global["mse"] += mse_global * y.shape[0]
            total_global["ssim"] += ssim_global * y.shape[0]
            total_global["n"] += y.shape[0]

    n_rf = total_rf["n"]
    n_g = total_global["n"]
    result = {}
    if n_rf > 0:
        mse_rf = total_rf["mse"] / n_rf
        result["rf_mse"] = float(mse_rf)
        result["rf_psnr"] = float(psnr_from_mse(mse_rf))
        result["rf_ssim"] = float(total_rf["ssim"] / n_rf)
    if n_g > 0:
        mse_g = total_global["mse"] / n_g
        result["global_mse"] = float(mse_g)
        result["global_psnr"] = float(psnr_from_mse(mse_g))
        result["global_ssim"] = float(total_global["ssim"] / n_g)
    return result
