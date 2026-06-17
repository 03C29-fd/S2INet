"""Frequency-Split Decoder: Low-freq (linear) + High-freq (CNN) reconstruction.

Pipeline:
  response [B, repeats, history, n_cells]
  → Low-Freq Path (Ridge Regression) → frame_low [B, 1, H, W]
  → High-Freq Path (Temporal CNN + Decoder) → frame_high [B, 1, H, W]
  → frame = sigmoid(low + high)

Evaluation:
  RF coverage mask for per-region metrics (rf_covered / non_rf_covered / global)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_blur_2d(image, sigma, kernel_size=None):
    """Apply Gaussian blur to a batch of images."""
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


def gaussian_blur_2d_np(image, sigma, kernel_size=None):
    """NumPy Gaussian blur for fitting (no GPU needed)."""
    import scipy.ndimage
    return scipy.ndimage.gaussian_filter(image, sigma=sigma, mode="reflect")


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
            low_frame: [B, 1, H, W] logit-style values (not sigmoided)
        """
        if self.W is None:
            raise RuntimeError("LowFreqRidge must be fit() before predict().")
        x = responses_mean.to(self.W.device, dtype=self.W.dtype)
        pred = x @ self.W + self.bias
        B = x.shape[0]
        side = int(np.sqrt(pred.shape[-1]))
        return pred.reshape(B, 1, side, side)


class FreqSplitDecoder(nn.Module):
    """Complete frequency-split reconstruction model.

    Low-freq path: pre-fitted Ridge regression (frozen, no gradient).
    High-freq path: temporal encoder + spatial decoder (trainable).
    """

    def __init__(self, n_cells, history_bins, image_size=64,
                 temporal_channels=128, latent_dim=512, base_channels=128,
                 dropout=0.15, blur_sigma=2.0, ridge_alpha=1.0,
                 low_freq_ridge=None, pretrain_epochs=5, pretrain_lr=1e-3):
        super().__init__()
        self.n_cells = n_cells
        self.history_bins = history_bins
        self.image_size = image_size
        self.blur_sigma = blur_sigma

        if low_freq_ridge is not None:
            self.low_freq = low_freq_ridge
        else:
            self.low_freq = LowFreqRidge(alpha=ridge_alpha, blur_sigma=blur_sigma)

        from wisa_model import MultiScaleTemporalEncoder, SpatialFrameDecoder
        self.high_encoder = MultiScaleTemporalEncoder(
            n_cells, latent_dim=latent_dim, channels=temporal_channels, dropout=dropout
        )
        self.high_decoder = SpatialFrameDecoder(
            latent_dim=latent_dim, image_size=(image_size, image_size),
            base_channels=base_channels, dropout=dropout,
        )

        self._low_fitted = False
        self._pretrain_epochs = pretrain_epochs
        self._pretrain_lr = pretrain_lr

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
        self._low_fitted = True

    def pretrain_high_path(self, train_dataset, device="cuda", batch_size=64):
        from torch.utils.data import DataLoader
        epochs = self._pretrain_epochs
        if epochs <= 0:
            return
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.high_encoder.to(device)
        self.high_decoder.to(device)
        optimizer = torch.optim.AdamW(
            list(self.high_encoder.parameters()) + list(self.high_decoder.parameters()),
            lr=self._pretrain_lr, weight_decay=1e-5,
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
                with torch.no_grad():
                    low = self.low_freq.predict(pooled).to(device, y.dtype)
                high_latent = self.high_encoder(x.mean(dim=1))
                high_logits = self.high_decoder(high_latent)
                logits = low + high_logits
                pred = torch.sigmoid(logits)
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

    def forward(self, x, return_single=False):
        B, K, H_bins, C = x.shape
        pooled = x.mean(dim=1)  # [B, history, cells]

        low = self.low_freq.predict(pooled.reshape(B, -1))

        high_latent = self.high_encoder(pooled)
        high_logits = self.high_decoder(high_latent)

        logits = low.to(high_logits.device, high_logits.dtype) + high_logits

        if return_single:
            single_logits = []
            for k in range(K):
                s_pooled = x[:, k]
                s_low = self.low_freq.predict(s_pooled.reshape(B, -1))
                s_latent = self.high_encoder(s_pooled)
                s_high = self.high_decoder(s_latent)
                single_logits.append((s_low.to(s_high.device, s_high.dtype) + s_high).unsqueeze(1))
            single_logits = torch.cat(single_logits, dim=1)
            return logits, single_logits, high_latent, high_latent
        return logits, high_latent, high_latent

    def get_freq_components(self, x):
        """Return low and high frequency components separately (for analysis)."""
        B, K, H_bins, C = x.shape
        pooled = x.mean(dim=1)

        low = self.low_freq.predict(pooled.reshape(B, -1))

        high_latent = self.high_encoder(pooled)
        high = self.high_decoder(high_latent)

        return torch.sigmoid(low), torch.sigmoid(low + high)


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
            pred = torch.sigmoid(model(x)[0])

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
