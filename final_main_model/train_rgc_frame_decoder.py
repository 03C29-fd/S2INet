import argparse
import csv
import json
import math
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["figure.dpi"] = 140
matplotlib.rcParams["savefig.dpi"] = 180

import matplotlib.pyplot as plt

from adaptive_loss import make_gaussian_rf_weight_map, make_rf_weight_map, ssim_index, ssim_map_2d
from train_movie_wisa import MOVIE_FILES, choose_device, read_movie_frames, set_seed
from wisa_model import (
    AttentionSpatialFrameDecoder,
    AttentionTemporalEncoder,
    MultiScaleTemporalEncoder,
    SpatialFrameDecoder,
)
from neural_image_decoder import (
    NeuralImageFrameDecoder,
    PatchGANDiscriminator,
    MixedSSIMMSELoss,
    SSIMGradientLoss,
    AdversarialReconstructionLoss,
    discriminator_loss,
    build_rf_templates,
)
from freq_split_decoder import (
    FreqSplitDecoder,
    build_rf_coverage_mask,
    compute_masked_metrics,
    evaluate_masked,
)
from lowfreq_aux_decoder import LowFrequencyAuxResidualDecoder
from detail_residual_decoder import DetailResidualFrameDecoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="RGC population response to movie-frame reconstruction: use post-stimulus response windows and repeat-level latent pooling."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--spikes-mat", type=Path, default=Path("data/movieBinnedSpiking.mat"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs_rgc_frame"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--split", type=str, default="scene", choices=["by_movie", "time", "scene", "leave_one_movie_out"])
    parser.add_argument("--val-movies", type=str, default="5")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--embargo", type=int, default=30)
    parser.add_argument("--scene-length", type=int, default=120)
    parser.add_argument("--response-lag", type=str, default="3", help="Global int or comma-separated per-movie lags (e.g. '3,3,5,2,4').")
    parser.add_argument("--history-bins", type=int, default=11)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache_movie_frames"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-valid-frames", type=int, default=1, help="Min valid frames per movie; warn/error if fewer.")

    parser.add_argument("--train-repeats", type=int, default=8, help="Repeats sampled per frame during training; 0 means all.")
    parser.add_argument("--eval-repeats", type=int, default=0, help="Repeats used per frame during eval; 0 means all.")
    parser.add_argument("--repeat-sampling", type=str, default="random", choices=["random", "stratified", "all"], help="How to select repeats per sample.")
    parser.add_argument("--normalize-inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--norm-samples", type=int, default=50000)

    parser.add_argument("--encoder", type=str, default="multiscale", choices=["multiscale", "attention"])
    parser.add_argument("--decoder", type=str, default="attention", choices=["linear", "conv", "attention", "dense_ae", "neural_image", "freq_split", "lowfreq_aux", "detail_residual"])
    parser.add_argument("--repeat-pool", type=str, default="attention", choices=["mean", "attention"])
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--temporal-channels", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.25)

    parser.add_argument("--loss-mode", type=str, default="rf_ssim_wlmse_gabor", choices=["mse_ssim", "rf_ssim_wlmse_gabor", "ssim_mse", "ssim_grad", "adversarial", "freq_split_ssim_aware"])
    parser.add_argument("--loss-mu", type=float, default=0.1, help="L = mu * LSSIM + (1 - mu) * WLMSE.")
    parser.add_argument("--gabor-loss-weight", type=float, default=0.02)
    parser.add_argument("--ssim-loss-weight", type=float, default=0.05, help="Used for mse_ssim mode.")
    parser.add_argument("--gradient-loss-weight", type=float, default=0.1, help="Weight for Sobel gradient loss.")
    parser.add_argument("--adv-weight", type=float, default=0.01, help="Adversarial loss weight.")
    parser.add_argument("--rf-sigma", type=float, default=3.0, help="Gaussian sigma for RF coverage and neural-image templates.")
    parser.add_argument("--freq-blur-sigma", type=float, default=4.0, help="Gaussian blur sigma for low-freq decomposition.")
    parser.add_argument("--freq-ridge-alpha", type=float, default=1.0, help="Ridge alpha for low-freq linear regressor.")
    parser.add_argument("--freq-pretrain-epochs", type=int, default=5, help="Warm-up epochs for freq-split residual paths.")
    parser.add_argument("--freq-pretrain-lr", type=float, default=1e-3, help="Warm-up learning rate for freq-split residual paths.")
    parser.add_argument("--freq-low-mode", type=str, default="ridge", choices=["ridge", "rf_template", "hybrid"], help="Low-frequency path for freq-split decoder.")
    parser.add_argument("--freq-template-alpha", type=float, default=1e-3, help="Regularization for fitting frame lows into RF-template basis.")
    parser.add_argument("--freq-hybrid-template-weight", type=float, default=0.5, help="Template low-path weight when --freq-low-mode hybrid.")
    parser.add_argument("--freq-high-rf-radius", type=float, default=0.0, help="If >0, restrict freq-split high residuals to soft disks around RF centers.")
    parser.add_argument("--freq-high-rf-softness", type=float, default=2.0, help="Soft edge width for freq-split RF-radius high-frequency mask.")
    parser.add_argument("--low-aux-sigma", type=float, default=0.0, help="Gaussian blur sigma for auxiliary low-frequency target; 0 disables blur.")
    parser.add_argument("--low-aux-weight", type=float, default=0.10, help="Weight for low-frequency auxiliary regularization.")
    parser.add_argument("--low-aux-ssim-weight", type=float, default=0.0, help="Optional SSIM term inside the low-frequency auxiliary loss.")
    parser.add_argument("--base-frame-prior-weight", type=float, default=0.35, help="Mix weight for global mean-frame prior in detail_residual decoder.")
    parser.add_argument("--edge-loss-weight", type=float, default=0.08, help="Weight for edge-preservation loss in detail_residual decoder.")
    parser.add_argument("--laplacian-loss-weight", type=float, default=0.05, help="Weight for Laplacian detail loss in detail_residual decoder.")
    parser.add_argument("--highfreq-loss-weight", type=float, default=0.08, help="Weight for high-frequency residual loss in detail_residual decoder.")
    parser.add_argument("--highfreq-sigma", type=float, default=2.0, help="Blur sigma used to define high-frequency residual targets.")
    parser.add_argument("--residual-gate", type=str, default="rf_scalar", choices=["none", "rf_scalar"], help="Optional gate for residual branch in lowfreq_aux decoder.")
    parser.add_argument("--residual-weight", type=float, default=0.10, help="Initial residual branch scale for lowfreq_aux decoder.")
    parser.add_argument("--gate-l1-weight", type=float, default=0.02, help="Regularization on residual gate magnitude and residual activation.")
    parser.add_argument("--rf-centers-path", type=Path, default=None, help="Optional CSV/NPY/MAT with RF centers shaped [n_cells,2].")
    parser.add_argument("--rf-checkerboard-path", type=Path, default=Path("data/binaryCheckerboard.mat"))
    parser.add_argument("--rf-estimate-from-checkerboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rf-max-frames", type=int, default=50000)
    parser.add_argument("--rf-source-size", type=int, default=40)
    parser.add_argument("--rf-baseline", type=float, default=1.0)
    parser.add_argument("--rf-peak", type=float, default=5.0)
    parser.add_argument("--consistency-weight", type=float, default=0.02)
    parser.add_argument("--single-repeat-loss-weight", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--estimate-batches", type=int, default=10)
    parser.add_argument("--viz-samples", type=int, default=8)
    parser.add_argument("--save-last", action="store_true")
    return parser.parse_args()


def psnr_from_mse(mse, data_range=1.0):
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10((data_range**2) / mse)


def image_from_model_output(model, output):
    return output if getattr(model, "outputs_are_images", False) else torch.sigmoid(output)


def make_gabor_bank(kernel_size=15, sigmas=(2.0, 4.0), lambdas=(4.0, 8.0), thetas=None):
    if thetas is None:
        thetas = [0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0]
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    filters = []
    for sigma in sigmas:
        for wavelength in lambdas:
            for theta in thetas:
                x_theta = xx * math.cos(theta) + yy * math.sin(theta)
                y_theta = -xx * math.sin(theta) + yy * math.cos(theta)
                envelope = torch.exp(-(x_theta.pow(2) + y_theta.pow(2)) / (2.0 * sigma**2))
                carrier = torch.cos(2.0 * math.pi * x_theta / wavelength)
                kernel = envelope * carrier
                kernel = kernel - kernel.mean()
                kernel = kernel / (kernel.abs().sum() + 1e-8)
                filters.append(kernel)
    return torch.stack(filters, dim=0).unsqueeze(1)


def load_rf_centers(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        centers = np.load(path)
    elif path.suffix.lower() in {".csv", ".txt"}:
        centers = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None)
    elif path.suffix.lower() == ".mat":
        mat = loadmat(path)
        candidates = ["rf_centers", "RF_centers", "centers", "rfCenters", "RF"]
        key = next((name for name in candidates if name in mat), None)
        if key is None:
            visible = [name for name in mat if not name.startswith("__")]
            raise KeyError(f"No RF center variable found in {path}. Tried {candidates}; available={visible}")
        centers = mat[key]
    else:
        raise ValueError(f"Unsupported RF center file type: {path.suffix}")
    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(f"RF centers must have shape [n_cells, 2], got {centers.shape} from {path}")
    return centers


def estimate_rf_centers_from_checkerboard(path, n_cells, max_frames=50000):
    mat = loadmat(path)
    if "binaryCheckerboard" not in mat or "stimulusFrames" not in mat:
        raise KeyError(f"{path} must contain binaryCheckerboard and stimulusFrames to estimate RF centers.")
    responses = np.asarray(mat["binaryCheckerboard"], dtype=np.float32)
    frames = np.asarray(mat["stimulusFrames"], dtype=np.float32)
    if responses.shape[0] != n_cells and responses.shape[1] == n_cells:
        responses = responses.T
    if responses.shape[0] != n_cells:
        raise ValueError(f"Expected {n_cells} cells in checkerboard responses, got {responses.shape}.")
    if frames.ndim != 3:
        raise ValueError(f"Expected stimulusFrames shaped [H,W,T], got {frames.shape}.")

    n_time = min(responses.shape[1], frames.shape[2], int(max_frames))
    responses = responses[:, :n_time]
    frames = frames[:, :, :n_time]
    height, width, _ = frames.shape

    stim = frames.reshape(height * width, n_time)
    stim = stim - stim.mean(axis=1, keepdims=True)
    resp = responses - responses.mean(axis=1, keepdims=True)
    sta = stim @ resp.T / max(n_time, 1)
    max_idx = np.argmax(np.abs(sta), axis=0)
    rows = max_idx // width
    cols = max_idx % width
    return np.stack([rows, cols], axis=1).astype(np.float32)


def resolve_rf_centers(args, n_cells):
    if args.rf_centers_path is not None:
        centers = load_rf_centers(args.rf_centers_path)
        source = str(args.rf_centers_path)
    elif args.rf_estimate_from_checkerboard and args.rf_checkerboard_path.exists():
        centers = estimate_rf_centers_from_checkerboard(args.rf_checkerboard_path, n_cells, args.rf_max_frames)
        source = f"estimated_from:{args.rf_checkerboard_path}"
    else:
        centers = None
        source = "fallback_fixed_points"
    if centers is not None and centers.shape[0] != n_cells:
        raise ValueError(f"RF centers count {centers.shape[0]} does not match n_cells={n_cells}.")
    return centers, source


class AlignReport:
    """Per-movie bin/lag/frame alignment diagnostic."""
    def __init__(self):
        self.records = []

    def add(self, name, n_frames_orig, n_resp_bins, n_reps, n_trunc, lag, h_bins,
            valid_start, valid_stop, n_train, n_val, skipped):
        self.records.append({
            "movie": name, "n_frames_orig": n_frames_orig, "n_resp_bins": n_resp_bins,
            "n_reps": n_reps, "n_trunc": n_trunc, "response_lag": lag,
            "history_bins": h_bins, "valid_start": valid_start, "valid_stop": valid_stop,
            "n_train": n_train, "n_val": n_val, "skipped": skipped,
        })

    def dump(self):
        if not self.records:
            print("[AlignReport] No movie records.")
            return
        header = (
            f"{'Movie':>6s} {'orig_f':>7s} {'rsp_bin':>7s} {'n_reps':>6s} "
            f"{'trunc':>5s} {'lag':>3s} {'h_bin':>4s} "
            f"{'v_start':>7s} {'v_stop':>7s} "
            f"{'n_train':>7s} {'n_val':>7s} {'skipped':>7s}"
        )
        sep = "-" * len(header) + "-" * 2
        print("\n[AlignReport] Per-movie alignment diagnostics")
        print(sep)
        print(header)
        print(sep)
        for r in self.records:
            skipped = "YES" if r["skipped"] else "no"
            print(
                f"{r['movie']:>6s} {r['n_frames_orig']:>7d} {r['n_resp_bins']:>7d} "
                f"{r['n_reps']:>6d} {r['n_trunc']:>5d} "
                f"{r['response_lag']:>3d} {r['history_bins']:>4d} "
                f"{r['valid_start']:>7d} {r['valid_stop']:>7d} "
                f"{r['n_train']:>7d} {r['n_val']:>7d} {skipped:>7s}"
            )
        print(sep)
        n_skipped = sum(1 for r in self.records if r["skipped"])
        total_train = sum(r["n_train"] for r in self.records)
        total_val = sum(r["n_val"] for r in self.records)
        print(f"[AlignReport] Total train frames: {total_train}, val frames: {total_val}")
        if n_skipped:
            print(f"[AlignReport] WARNING: {n_skipped} movie(s) skipped due to insufficient valid frames.")
        return total_train, total_val, n_skipped


def parse_response_lags(lag_str, n_movies):
    parts = [x.strip() for x in str(lag_str).split(",")]
    if len(parts) == 1:
        lag = int(parts[0])
        return [lag] * n_movies
    lags = [int(x) for x in parts]
    if len(lags) != n_movies:
        raise ValueError(f"Per-movie response_lag count ({len(lags)}) != n_movies ({n_movies}).")
    return lags


class RFWeightedGaborLoss(nn.Module):
    def __init__(
        self,
        image_size,
        mu=0.1,
        gabor_weight=0.02,
        rf_centers=None,
        rf_source_size=40,
        rf_sigma=3.0,
        rf_baseline=1.0,
        rf_peak=5.0,
    ):
        super().__init__()
        self.mu = mu
        self.gabor_weight = gabor_weight
        self.rf_style = "gaussian_per_neuron" if rf_centers is not None else "fallback_fixed_points"
        if rf_centers is not None:
            weight = make_gaussian_rf_weight_map(
                image_size[0],
                image_size[1],
                rf_centers,
                source_size=rf_source_size,
                sigma=rf_sigma,
                baseline=rf_baseline,
                peak=rf_peak,
                dtype=torch.float32,
            )
        else:
            weight = make_rf_weight_map(image_size[0], image_size[1], dtype=torch.float32)
        self.register_buffer("rf_weight", weight)
        self.register_buffer("rf_norm", weight.mean().clamp_min(1e-8))
        self.register_buffer("gabor_bank", make_gabor_bank())

    def forward(self, pred, target):
        rf_weight = self.rf_weight.to(pred.device, pred.dtype)
        rf_norm = self.rf_norm.to(pred.device, pred.dtype)
        weighted_mse = (((pred - target) ** 2) * rf_weight).mean() / rf_norm

        ssim_map = ssim_map_2d(pred, target)
        weighted_ssim = (ssim_map * rf_weight).mean() / rf_norm
        ssim_loss = 1.0 - weighted_ssim

        loss = self.mu * ssim_loss + (1.0 - self.mu) * weighted_mse
        if self.gabor_weight > 0:
            bank = self.gabor_bank.to(pred.device, pred.dtype)
            pred_gabor = F.conv2d(pred, bank, padding=bank.shape[-1] // 2)
            target_gabor = F.conv2d(target, bank, padding=bank.shape[-1] // 2)
            loss = loss + self.gabor_weight * F.l1_loss(pred_gabor, target_gabor)
        return loss


class RGCFrameResponseDataset(Dataset):
    def __init__(
        self,
        data_dir,
        spikes_mat,
        image_size,
        split,
        val_movies,
        val_ratio,
        embargo,
        scene_length,
        response_lags,
        history_bins,
        cache_dir,
        sample_repeats,
        train,
        max_samples=None,
        seed=42,
        repeat_sampling="random",
        min_valid_frames=1,
        align_report=None,
    ):
        mat = loadmat(spikes_mat)
        binned = mat["binned"]
        nreps = mat["nreps"].reshape(-1).astype(int)

        if not isinstance(response_lags, (list, tuple)):
            response_lags = [int(response_lags)] * len(MOVIE_FILES)
        response_lags = [int(l) for l in response_lags]

        self.frames = []
        self.responses = []
        self.samples = []
        self.per_movie_lags = []
        self.requested_sample_repeats = sample_repeats
        self.sample_repeats = sample_repeats
        self.train = train
        self.history_bins = history_bins
        self.repeat_sampling = repeat_sampling
        self.rng = np.random.default_rng(seed + (0 if train else 10000))
        self.input_mean = None
        self.input_std = None
        self._repeat_epoch_counter = 0
        self.movie_global_indices = []

        val_movie_ids = {int(item) - 1 for item in str(val_movies).split(",") if item.strip()}
        for movie_idx, filename in enumerate(MOVIE_FILES):
            frames_orig = read_movie_frames(data_dir / filename, image_size, cache_dir)
            reps = int(nreps[movie_idx])
            resp_orig = binned[:reps, :, :, movie_idx].astype(np.float32)
            n_frames_orig = frames_orig.shape[0]
            n_resp_bins_orig = resp_orig.shape[1]
            n_time = min(n_frames_orig, n_resp_bins_orig)

            movie_lag = response_lags[movie_idx]
            required_len = history_bins + movie_lag
            if n_time < required_len:
                if align_report is not None:
                    align_report.add(
                        MOVIE_FILES[movie_idx], n_frames_orig, n_resp_bins_orig, reps,
                        n_time, movie_lag, history_bins, -1, -1, 0, 0, skipped=True,
                    )
                continue

            frames = frames_orig[:n_time]
            resp = resp_orig[:, :n_time, :]

            start = max(0, history_bins - 1 - movie_lag)
            stop = n_time - movie_lag
            if stop <= start:
                if align_report is not None:
                    align_report.add(
                        MOVIE_FILES[movie_idx], n_frames_orig, n_resp_bins_orig, reps,
                        n_time, movie_lag, history_bins, start, stop, 0, 0, skipped=True,
                    )
                continue

            if split in ("by_movie", "leave_one_movie_out"):
                is_val_movie = movie_idx in val_movie_ids
                if is_val_movie != (not train):
                    if align_report is not None:
                        align_report.add(
                            MOVIE_FILES[movie_idx], n_frames_orig, n_resp_bins_orig, reps,
                            n_time, movie_lag, history_bins, start, stop,
                            n_train=0, n_val=0, skipped=True,
                        )
                    continue
                time_indices = np.arange(start, stop, dtype=np.int64)
            elif split == "scene":
                time_indices = self._scene_split_indices(
                    start=start, stop=stop, val_ratio=val_ratio,
                    embargo=embargo, scene_length=scene_length,
                    train=train, seed=seed + movie_idx * 9973,
                )
            else:
                val_size = max(1, int(round((stop - start) * val_ratio)))
                val_start = stop - val_size
                train_end = max(start, val_start - max(embargo, movie_lag))
                if train:
                    time_indices = np.arange(start, train_end, dtype=np.int64)
                else:
                    time_indices = np.arange(val_start, stop, dtype=np.int64)

            n_valid = len(time_indices)
            if n_valid < min_valid_frames:
                if align_report is not None:
                    align_report.add(
                        MOVIE_FILES[movie_idx], n_frames_orig, n_resp_bins_orig, reps,
                        n_time, movie_lag, history_bins, start, stop,
                        n_train=n_valid if train else 0, n_val=n_valid if not train else 0,
                        skipped=True,
                    )
                continue

            local_movie_idx = len(self.frames)
            self.frames.append(frames)
            self.responses.append(resp)
            self.per_movie_lags.append(movie_lag)
            self.movie_global_indices.append(movie_idx)
            for t in time_indices:
                self.samples.append((local_movie_idx, int(t)))

            if align_report is not None:
                align_report.add(
                    MOVIE_FILES[movie_idx], n_frames_orig, n_resp_bins_orig, reps,
                    n_time, movie_lag, history_bins, start, stop,
                    n_train=n_valid if train else 0, n_val=n_valid if not train else 0,
                    skipped=False,
                )

        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise ValueError("No samples created. Check split/lag/history settings.")

        self.samples = np.asarray(self.samples, dtype=np.int64)
        self.n_cells = self.responses[0].shape[2]
        if self.sample_repeats <= 0:
            self.sample_repeats = min(resp.shape[0] for resp in self.responses)
        self.input_dim = self.n_cells * history_bins
        self.target_size = (image_size, image_size)

    @staticmethod
    def _scene_split_indices(start, stop, val_ratio, embargo, scene_length, train, seed):
        scene_length = max(1, int(scene_length))
        blocks = []
        for block_start in range(start, stop, scene_length):
            block_stop = min(stop, block_start + scene_length)
            if block_stop > block_start:
                blocks.append((block_start, block_stop))
        if len(blocks) < 2:
            return np.arange(start, stop, dtype=np.int64)

        rng = np.random.default_rng(seed)
        n_val = max(1, int(round(len(blocks) * val_ratio)))
        n_val = min(n_val, len(blocks) - 1)
        val_block_ids = set(rng.choice(len(blocks), size=n_val, replace=False).tolist())
        val_ranges = [blocks[idx] for idx in val_block_ids]

        indices = []
        for block_idx, (block_start, block_stop) in enumerate(blocks):
            if train:
                if block_idx in val_block_ids:
                    continue
                keep_start, keep_stop = block_start, block_stop
                for val_start, val_stop in val_ranges:
                    if keep_stop <= val_start - embargo or keep_start >= val_stop + embargo:
                        continue
                    if keep_start < val_start:
                        keep_stop = min(keep_stop, val_start - embargo)
                    else:
                        keep_start = max(keep_start, val_stop + embargo)
                if keep_stop > keep_start:
                    indices.append(np.arange(keep_start, keep_stop, dtype=np.int64))
            elif block_idx in val_block_ids:
                indices.append(np.arange(block_start, block_stop, dtype=np.int64))

        if not indices:
            return np.asarray([], dtype=np.int64)
        out = np.concatenate(indices)
        rng.shuffle(out)
        return out.astype(np.int64)

    def __len__(self):
        return len(self.samples)

    def _select_repeats(self, n_reps):
        if self.sample_repeats <= 0 or self.repeat_sampling == "all":
            return np.arange(n_reps, dtype=np.int64)
        if self.sample_repeats >= n_reps:
            return np.arange(n_reps, dtype=np.int64)
        if self.repeat_sampling == "stratified":
            if self.train:
                self._repeat_epoch_counter += 1
                rng = np.random.default_rng(self._repeat_epoch_counter * 10007 + hash(str(n_reps)) % 9973)
            else:
                rng = np.random.default_rng(98765)
            return rng.choice(n_reps, size=self.sample_repeats, replace=False).astype(np.int64)
        if self.train:
            return self.rng.choice(n_reps, size=self.sample_repeats, replace=False).astype(np.int64)
        return np.linspace(0, n_reps - 1, self.sample_repeats, dtype=np.int64)

    def _repeat_window(self, movie_idx, rep_idx, t):
        resp = self.responses[movie_idx]
        movie_lag = self.per_movie_lags[movie_idx]
        spike_t0 = t + movie_lag - self.history_bins + 1
        spike_t1 = t + movie_lag + 1
        if spike_t0 < 0 or spike_t1 > resp.shape[1]:
            raise IndexError(
                f"Invalid lag window for frame_t={t}: spike_t0={spike_t0}, "
                f"spike_t1={spike_t1}, response_time={resp.shape[1]}."
            )
        x = resp[rep_idx, spike_t0:spike_t1, :].astype(np.float32)
        if self.input_mean is not None and self.input_std is not None:
            x = (x - self.input_mean) / self.input_std
        return x

    def fit_input_normalization(self, max_features=50000):
        rng = np.random.default_rng(123)
        total = np.zeros((self.history_bins, self.n_cells), dtype=np.float64)
        total_sq = np.zeros_like(total)
        count = 0
        n_draws = min(max_features, len(self.samples))
        sample_ids = rng.choice(len(self.samples), size=n_draws, replace=False)
        for sample_id in tqdm(sample_ids, desc="fit norm", leave=False):
            movie_idx, t = self.samples[int(sample_id)]
            n_reps = self.responses[int(movie_idx)].shape[0]
            rep_idx = int(rng.integers(0, n_reps))
            x = self._repeat_window(int(movie_idx), rep_idx, int(t))
            total += x
            total_sq += x * x
            count += 1
        mean = total / count
        var = np.maximum(total_sq / count - mean * mean, 1e-12)
        self.input_mean = mean.astype(np.float32)
        self.input_std = np.sqrt(var).astype(np.float32)

    def set_input_normalization(self, mean, std):
        self.input_mean = mean
        self.input_std = std

    def __getitem__(self, idx):
        movie_idx, t = self.samples[int(idx)]
        resp = self.responses[int(movie_idx)]
        repeat_ids = self._select_repeats(resp.shape[0])
        x = np.stack([self._repeat_window(int(movie_idx), int(rep), int(t)) for rep in repeat_ids], axis=0)
        y = self.frames[int(movie_idx)][int(t)][None, :, :].astype(np.float32)
        prev_t = max(0, int(t) - 1)
        prev_y = self.frames[int(movie_idx)][prev_t][None, :, :].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(prev_y)


class DenseAEFrameRefiner(nn.Module):
    def __init__(self, latent_dim=512, image_size=(64, 64), base_channels=64, dropout=0.25):
        super().__init__()
        self.image_size = image_size
        height, width = image_size
        self.coarse = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, height * width),
        )
        self.enc1 = self._down_block(1, base_channels, kernel_size=7, dropout=0.0)
        self.enc2 = self._down_block(base_channels, base_channels * 2, kernel_size=5, dropout=dropout)
        self.enc3 = self._down_block(base_channels * 2, base_channels * 4, kernel_size=3, dropout=dropout)
        self.enc4 = self._down_block(base_channels * 4, base_channels * 4, kernel_size=3, dropout=dropout)
        self.dec1 = self._up_block(base_channels * 4, base_channels * 4, kernel_size=3, dropout=dropout)
        self.dec2 = self._up_block(base_channels * 4, base_channels * 2, kernel_size=3, dropout=dropout)
        self.dec3 = self._up_block(base_channels * 2, base_channels, kernel_size=5, dropout=dropout)
        self.dec4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels, 1, kernel_size=7, padding=3),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _down_block(in_channels, out_channels, kernel_size, dropout):
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

    @staticmethod
    def _up_block(in_channels, out_channels, kernel_size, dropout):
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, latent):
        height, width = self.image_size
        coarse_logits = self.coarse(latent).reshape(latent.shape[0], 1, height, width)
        coarse_frame = torch.sigmoid(coarse_logits)
        out = self.enc1(coarse_frame)
        out = self.enc2(out)
        out = self.enc3(out)
        out = self.enc4(out)
        out = self.dec1(out)
        out = self.dec2(out)
        out = self.dec3(out)
        residual_logits = self.dec4(out)
        if residual_logits.shape[-2:] != self.image_size:
            residual_logits = F.interpolate(residual_logits, size=self.image_size, mode="bilinear", align_corners=False)
        return coarse_logits + torch.tanh(self.residual_scale) * residual_logits


class LinearFrameDecoder(nn.Module):
    def __init__(self, latent_dim=512, image_size=(64, 64)):
        super().__init__()
        self.image_size = image_size
        self.fc = nn.Linear(latent_dim, image_size[0] * image_size[1])

    def forward(self, latent):
        out = self.fc(latent)
        return out.reshape(latent.shape[0], 1, self.image_size[0], self.image_size[1])


class RGCFrameDecoder(nn.Module):
    def __init__(
        self,
        n_cells,
        history_bins,
        image_size=(64, 64),
        encoder="multiscale",
        decoder="attention",
        repeat_pool="attention",
        latent_dim=512,
        temporal_channels=128,
        base_channels=128,
        attention_heads=4,
        attention_layers=1,
        dropout=0.25,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.history_bins = history_bins
        self.repeat_pool = repeat_pool
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

        if decoder == "linear":
            self.decoder = LinearFrameDecoder(latent_dim, image_size)
        elif decoder == "dense_ae":
            self.decoder = DenseAEFrameRefiner(latent_dim, image_size, max(base_channels // 2, 32), dropout)
        elif decoder == "attention":
            self.decoder = AttentionSpatialFrameDecoder(latent_dim, image_size, base_channels, dropout)
        else:
            self.decoder = SpatialFrameDecoder(latent_dim, image_size, base_channels, dropout)

    def forward(self, x, return_single=False):
        # x: [B, repeats, history, cells]
        b, k, h, c = x.shape
        latents = self.encoder(x.reshape(b * k, h, c)).reshape(b, k, -1)
        if self.repeat_attention is None:
            pooled = latents.mean(dim=1)
        else:
            weights = self.repeat_attention(latents)
            pooled = (latents * weights).sum(dim=1)
        recon_logits = self.decoder(pooled)
        if return_single:
            single_logits = self.decoder(latents.reshape(b * k, -1)).reshape(b, k, 1, *recon_logits.shape[-2:])
            return recon_logits, single_logits, latents, pooled
        return recon_logits, latents, pooled


def make_datasets(args):
    response_lags = parse_response_lags(args.response_lag, len(MOVIE_FILES))
    common = dict(
        data_dir=args.data_dir,
        spikes_mat=args.spikes_mat,
        image_size=args.image_size,
        split=args.split,
        val_movies=args.val_movies,
        val_ratio=args.val_ratio,
        embargo=args.embargo,
        scene_length=args.scene_length,
        response_lags=response_lags,
        history_bins=args.history_bins,
        cache_dir=args.cache_dir,
        max_samples=args.max_samples,
        repeat_sampling=args.repeat_sampling,
        min_valid_frames=args.min_valid_frames,
    )
    align_report = AlignReport()
    train_dataset = RGCFrameResponseDataset(
        **common,
        sample_repeats=args.train_repeats,
        train=True,
        seed=args.seed,
        align_report=align_report,
    )
    val_dataset = RGCFrameResponseDataset(
        **common,
        sample_repeats=args.eval_repeats,
        train=False,
        seed=args.seed,
        align_report=align_report,
    )
    if args.normalize_inputs:
        train_dataset.fit_input_normalization(max_features=args.norm_samples)
        val_dataset.set_input_normalization(train_dataset.input_mean, train_dataset.input_std)
    return train_dataset, val_dataset, align_report


def build_model(args, dataset, device=None):
    if args.decoder == "neural_image":
        rf_centers, _ = resolve_rf_centers(args, dataset.n_cells)
        return NeuralImageFrameDecoder(
            n_cells=dataset.n_cells,
            history_bins=args.history_bins,
            image_size=dataset.target_size[0],
            rf_centers=rf_centers,
            rf_sigma=args.rf_sigma,
            rf_source_size=args.rf_source_size,
            base_channels=args.base_channels,
            dropout=args.dropout,
        )
    if args.decoder == "freq_split":
        rf_centers, _ = resolve_rf_centers(args, dataset.n_cells)
        model = FreqSplitDecoder(
            n_cells=dataset.n_cells,
            history_bins=args.history_bins,
            image_size=dataset.target_size[0],
            temporal_channels=args.temporal_channels,
            latent_dim=args.latent_dim,
            base_channels=args.base_channels,
            dropout=args.dropout,
            blur_sigma=args.freq_blur_sigma,
            ridge_alpha=args.freq_ridge_alpha,
            pretrain_epochs=args.freq_pretrain_epochs,
            pretrain_lr=args.freq_pretrain_lr,
            low_mode=args.freq_low_mode,
            rf_centers=rf_centers,
            rf_source_size=args.rf_source_size,
            rf_sigma=args.rf_sigma,
            template_alpha=args.freq_template_alpha,
            hybrid_template_weight=args.freq_hybrid_template_weight,
            high_rf_radius=args.freq_high_rf_radius,
            high_rf_softness=args.freq_high_rf_softness,
        )
        model.fit_low_freq(dataset, device="cpu")
        if device is not None:
            model.pretrain_high_path(dataset, device=device)
        model._rf_centers = rf_centers
        return model
    if args.decoder == "lowfreq_aux":
        rf_centers, _ = resolve_rf_centers(args, dataset.n_cells)
        model = LowFrequencyAuxResidualDecoder(
            n_cells=dataset.n_cells,
            history_bins=args.history_bins,
            image_size=dataset.target_size,
            encoder=args.encoder,
            repeat_pool=args.repeat_pool,
            latent_dim=args.latent_dim,
            temporal_channels=args.temporal_channels,
            base_channels=args.base_channels,
            attention_heads=args.attention_heads,
            attention_layers=args.attention_layers,
            dropout=args.dropout,
            low_aux_sigma=args.low_aux_sigma,
            low_aux_weight=args.low_aux_weight,
            low_aux_ssim_weight=args.low_aux_ssim_weight,
            residual_gate=args.residual_gate,
            residual_weight=args.residual_weight,
            gate_l1_weight=args.gate_l1_weight,
            rf_centers=rf_centers,
            rf_sigma=args.rf_sigma,
            rf_source_size=args.rf_source_size,
        )
        model._rf_centers = rf_centers
        return model
    if args.decoder == "detail_residual":
        rf_centers, _ = resolve_rf_centers(args, dataset.n_cells)
        model = DetailResidualFrameDecoder(
            n_cells=dataset.n_cells,
            history_bins=args.history_bins,
            image_size=dataset.target_size,
            encoder=args.encoder,
            repeat_pool=args.repeat_pool,
            latent_dim=args.latent_dim,
            temporal_channels=args.temporal_channels,
            base_channels=args.base_channels,
            attention_heads=args.attention_heads,
            attention_layers=args.attention_layers,
            dropout=args.dropout,
            base_frame=compute_global_mean_frame(dataset),
            base_frame_weight=args.base_frame_prior_weight,
            low_aux_sigma=args.low_aux_sigma,
            low_aux_weight=args.low_aux_weight,
            low_aux_ssim_weight=args.low_aux_ssim_weight,
            edge_loss_weight=args.edge_loss_weight,
            laplacian_loss_weight=args.laplacian_loss_weight,
            highfreq_loss_weight=args.highfreq_loss_weight,
            highfreq_sigma=args.highfreq_sigma,
            residual_gate=args.residual_gate,
            residual_weight=args.residual_weight,
            gate_l1_weight=args.gate_l1_weight,
            rf_centers=rf_centers,
            rf_sigma=args.rf_sigma,
            rf_source_size=args.rf_source_size,
        )
        model._rf_centers = rf_centers
        return model
    return RGCFrameDecoder(
        n_cells=dataset.n_cells,
        history_bins=args.history_bins,
        image_size=dataset.target_size,
        encoder=args.encoder,
        decoder=args.decoder,
        repeat_pool=args.repeat_pool,
        latent_dim=args.latent_dim,
        temporal_channels=args.temporal_channels,
        base_channels=args.base_channels,
        attention_heads=args.attention_heads,
        attention_layers=args.attention_layers,
        dropout=args.dropout,
    )


def build_loss(args, dataset):
    if args.loss_mode == "ssim_mse":
        rf_centers, _ = resolve_rf_centers(args, dataset.n_cells)
        if rf_centers is not None:
            rf_weight = make_gaussian_rf_weight_map(
                dataset.target_size[0], dataset.target_size[1],
                rf_centers, source_size=args.rf_source_size,
                sigma=args.rf_sigma, baseline=args.rf_baseline, peak=args.rf_peak,
            )
        else:
            rf_weight = make_rf_weight_map(dataset.target_size[0], dataset.target_size[1])
        return MixedSSIMMSELoss(rf_weight, ssim_weight=0.8, mse_weight=0.2)

    if args.loss_mode == "ssim_grad":
        rf_centers, _ = resolve_rf_centers(args, dataset.n_cells)
        if rf_centers is not None:
            rf_weight = make_gaussian_rf_weight_map(
                dataset.target_size[0], dataset.target_size[1],
                rf_centers, source_size=args.rf_source_size,
                sigma=args.rf_sigma, baseline=args.rf_baseline, peak=args.rf_peak,
            )
        else:
            rf_weight = make_rf_weight_map(dataset.target_size[0], dataset.target_size[1])
        return SSIMGradientLoss(
            rf_weight,
            ssim_weight=0.7, mse_weight=0.2,
            grad_weight=args.gradient_loss_weight,
        )

    if args.loss_mode == "adversarial":
        rf_centers, _ = resolve_rf_centers(args, dataset.n_cells)
        if rf_centers is not None:
            rf_weight = make_gaussian_rf_weight_map(
                dataset.target_size[0], dataset.target_size[1],
                rf_centers, source_size=args.rf_source_size,
                sigma=args.rf_sigma, baseline=args.rf_baseline, peak=args.rf_peak,
            )
        else:
            rf_weight = make_rf_weight_map(dataset.target_size[0], dataset.target_size[1])
        criterion = AdversarialReconstructionLoss(
            rf_weight, ssim_weight=0.7, mse_weight=0.2,
            grad_weight=args.gradient_loss_weight, adv_weight=args.adv_weight,
        )
        discriminator = PatchGANDiscriminator(in_channels=1, base_channels=64)
        criterion.discriminator = discriminator
        return criterion

    if args.loss_mode == "rf_ssim_wlmse_gabor":
        rf_centers, rf_source = resolve_rf_centers(args, dataset.n_cells)
        criterion = RFWeightedGaborLoss(
            dataset.target_size,
            mu=args.loss_mu,
            gabor_weight=args.gabor_loss_weight,
            rf_centers=rf_centers,
            rf_source_size=args.rf_source_size,
            rf_sigma=args.rf_sigma,
            rf_baseline=args.rf_baseline,
            rf_peak=args.rf_peak,
        )
        criterion.rf_source = rf_source
        return criterion

    if args.loss_mode == "freq_split_ssim_aware":
        if args.decoder != "freq_split":
            raise ValueError("--loss-mode freq_split_ssim_aware requires --decoder freq_split.")
        return None

    def mse_ssim_loss(pred, target):
        loss = F.mse_loss(pred, target)
        if args.ssim_loss_weight > 0:
            loss = loss + args.ssim_loss_weight * (1.0 - ssim_index(pred, target).mean())
        return loss
    return mse_ssim_loss


def compute_per_movie_mean_frames(dataset):
    per_movie_means = {}
    for local_idx, frames in enumerate(dataset.frames):
        global_idx = int(dataset.movie_global_indices[local_idx])
        mean_frame = frames.mean(axis=0, keepdims=True)
        per_movie_means[global_idx] = torch.from_numpy(mean_frame)[None, :, :]
    if not per_movie_means:
        raise ValueError("Cannot compute mean frames: dataset.frames is empty.")
    all_means = torch.stack(list(per_movie_means.values()))
    global_mean = all_means.mean(dim=0)
    return per_movie_means, global_mean


def compute_global_mean_frame(dataset):
    frames = [torch.from_numpy(movie_frames.mean(axis=0, keepdims=True))[None, :, :] for movie_frames in dataset.frames]
    if not frames:
        raise ValueError("Cannot compute global mean frame: dataset.frames is empty.")
    return torch.stack(frames).mean(dim=0)


def compute_temporal_smoothness(model, dataset, device):
    """Detect if model has collapsed to predicting near-constant frames.
    Returns mean pairwise SSIM between consecutive frame predictions."""
    n_samples = min(128, len(dataset))
    selected = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)
    loader = DataLoader(torch.utils.data.Subset(dataset, selected.tolist()), batch_size=n_samples, shuffle=False)
    model.eval()
    with torch.no_grad():
        x, y, _prev_y = next(iter(loader))
        raw = model(x.to(device))[0]
        pred = image_from_model_output(model, raw).cpu()
    inter_frame_sim = []
    for i in range(1, len(pred)):
        ssim_val = ssim_index(pred[i:i+1], pred[i-1:i], data_range=1.0, window_size=11, sigma=1.5).item()
        inter_frame_sim.append(ssim_val)
    if inter_frame_sim:
        avg = float(np.mean(inter_frame_sim))
        std = float(np.std(inter_frame_sim))
        max_sim = float(np.max(inter_frame_sim))
        gt_sim = []
        for i in range(1, len(y)):
            gt_sim.append(ssim_index(y[i:i+1], y[i-1:i]).item())
        gt_avg = float(np.mean(gt_sim)) if gt_sim else 0.0
        result = {
            "inter_frame_ssim_mean": avg,
            "inter_frame_ssim_std": std,
            "inter_frame_ssim_max": max_sim,
            "gt_inter_frame_ssim_mean": gt_avg,
            "n_pairs": len(inter_frame_sim),
        }
        if avg > 0.98:
            result["warning"] = "HIGH inter-frame prediction similarity; model may predict nearly constant frames."
        return result
    return {"inter_frame_ssim_mean": 0.0, "warning": "insufficient samples"}


def run_epoch(model, loader, optimizer, device, train, epoch, args, criterion, disc_optimizer=None):
    model.train(train)
    total_loss = total_mse = total_ssim = total_consistency = 0.0
    total_samples = 0
    estimate = None
    batch_times = []
    is_adversarial = hasattr(criterion, "discriminator") and criterion.discriminator is not None
    if is_adversarial:
        criterion.discriminator.train(train)
    context = torch.enable_grad() if train else torch.no_grad()
    desc = f"epoch {epoch:03d} {'train' if train else 'val'}"
    with context:
        pbar = tqdm(loader, desc=desc, leave=False)
        for batch_idx, (x, y, _prev_y) in enumerate(pbar, start=1):
            start = time.perf_counter()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)

            logits, single_logits, latents, pooled = model(x, return_single=True)
            pred = image_from_model_output(model, logits)

            if is_adversarial and train:
                loss = criterion.generator_loss(pred, y, criterion.discriminator)
            elif args.loss_mode == "freq_split_ssim_aware":
                loss = model.frequency_loss(pred, y)
            else:
                loss = criterion(pred, y)

            if hasattr(model, "auxiliary_loss"):
                loss = loss + model.auxiliary_loss(y)
            if hasattr(model, "detail_loss"):
                loss = loss + model.detail_loss(y)

            consistency = ((latents - pooled[:, None, :]) ** 2).mean()
            if args.consistency_weight > 0:
                loss = loss + args.consistency_weight * consistency

            if args.single_repeat_loss_weight > 0:
                single_pred = image_from_model_output(model, single_logits)
                single_target = y[:, None, :, :, :].expand_as(single_pred)
                if args.loss_mode == "freq_split_ssim_aware":
                    single_loss = F.mse_loss(single_pred.reshape(-1, *y.shape[1:]), single_target.reshape(-1, *y.shape[1:]))
                else:
                    single_loss = criterion(single_pred.reshape(-1, *y.shape[1:]), single_target.reshape(-1, *y.shape[1:]))
                loss = loss + args.single_repeat_loss_weight * single_loss

            if train:
                loss.backward()
                optimizer.step()
                if is_adversarial and disc_optimizer is not None:
                    disc_optimizer.zero_grad(set_to_none=True)
                    d_loss = discriminator_loss(criterion.discriminator, y, pred.detach())
                    d_loss.backward()
                    disc_optimizer.step()

            with torch.no_grad():
                mse = F.mse_loss(pred, y, reduction="mean")
                ssim = ssim_index(pred, y).mean()

            batch_size = x.shape[0]
            total_loss += loss.item() * batch_size
            total_mse += mse.item() * batch_size
            total_ssim += ssim.item() * batch_size
            total_consistency += consistency.item() * batch_size
            total_samples += batch_size
            pbar.set_postfix(loss=f"{loss.item():.5f}", mse=f"{mse.item():.5f}", ssim=f"{ssim.item():.4f}")

            elapsed = time.perf_counter() - start
            if train and epoch == 1 and batch_idx <= args.estimate_batches:
                batch_times.append(elapsed)
                if batch_idx == min(args.estimate_batches, len(loader)):
                    avg = float(np.mean(batch_times))
                    estimate = {
                        "seconds_per_batch": avg,
                        "seconds_per_epoch": avg * len(loader),
                        "seconds_total_training": avg * len(loader) * args.epochs,
                        "batches_used": len(batch_times),
                    }
                    print(f"Time estimate: {estimate['seconds_per_epoch']:.1f}s/epoch")

    mse = total_mse / total_samples
    return {
        "loss": total_loss / total_samples,
        "mse": mse,
        "psnr": psnr_from_mse(mse),
        "ssim": total_ssim / total_samples,
        "consistency": total_consistency / total_samples,
        "time_estimate": estimate,
    }


def evaluate_prediction(model, loader, device, mode, per_movie_means=None, global_mean=None, mean_response_input=None, zero_response_input=None, per_movie=False, dataset=None):
    model.eval()
    total_mse = total_ssim = total_samples = 0.0
    with torch.no_grad():
        for x, y, prev_y in tqdm(loader, desc=f"eval {mode}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            prev_y = prev_y.to(device, non_blocking=True)
            if mode == "normal":
                pred = image_from_model_output(model, model(x)[0])
            elif mode == "shuffle_response":
                perm = torch.randperm(x.shape[0], device=device)
                pred = image_from_model_output(model, model(x[perm])[0])
            elif mode == "mean_response":
                if mean_response_input is None:
                    control_x = torch.zeros_like(x)
                else:
                    control_x = mean_response_input.to(device).view(1, 1, *x.shape[2:]).expand_as(x)
                pred = image_from_model_output(model, model(control_x)[0])
            elif mode == "zero_response":
                if zero_response_input is None:
                    control_x = torch.zeros_like(x)
                else:
                    control_x = zero_response_input.to(device).view(1, 1, *x.shape[2:]).expand_as(x)
                pred = image_from_model_output(model, model(control_x)[0])
            elif mode == "mean_frame":
                pred = global_mean.to(device).expand(y.shape[0], -1, -1, -1)
            elif mode == "previous_frame":
                pred = prev_y
            else:
                raise ValueError(mode)
            mse = F.mse_loss(pred, y, reduction="mean")
            ssim = ssim_index(pred, y).mean()
            total_mse += mse.item() * y.shape[0]
            total_ssim += ssim.item() * y.shape[0]
            total_samples += y.shape[0]
    mse = total_mse / total_samples
    result = {"mse": mse, "psnr": psnr_from_mse(mse), "ssim": total_ssim / total_samples, "n": total_samples}
    if per_movie and dataset is not None:
        result["per_movie"] = evaluate_per_movie(model, dataset, device, mode, per_movie_means, global_mean, mean_response_input, zero_response_input)
    return result


def evaluate_per_movie(model, dataset, device, mode, per_movie_means, global_mean, mean_response_input=None, zero_response_input=None):
    movie_to_indices = {}
    for idx, (movie_id, _t) in enumerate(dataset.samples):
        movie_to_indices.setdefault(int(movie_id), []).append(idx)
    pm = {}
    for movie_id in sorted(movie_to_indices.keys()):
        indices = movie_to_indices[movie_id]
        subset = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=0)
        total_mse = total_ssim = total_n = 0.0
        global_idx = int(dataset.movie_global_indices[int(movie_id)])
        with torch.no_grad():
            for x, y, prev_y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                prev_y = prev_y.to(device, non_blocking=True)
                if mode == "normal":
                    pred = image_from_model_output(model, model(x)[0])
                elif mode == "mean_frame":
                    current_mean = per_movie_means.get(global_idx, global_mean)
                    pred = current_mean.to(device).expand(y.shape[0], -1, -1, -1)
                elif mode == "previous_frame":
                    pred = prev_y
                else:
                    continue
                mse = F.mse_loss(pred, y, reduction="mean")
                total_mse += mse.item() * y.shape[0]
                total_ssim += ssim_index(pred, y).mean().item() * y.shape[0]
                total_n += y.shape[0]
        if total_n > 0:
            pm[f"movie{movie_id}_mse"] = total_mse / total_n
            pm[f"movie{movie_id}_psnr"] = psnr_from_mse(total_mse / total_n)
            pm[f"movie{movie_id}_ssim"] = total_ssim / total_n
            pm[f"movie{movie_id}_n"] = total_n
    return pm


def save_visualization(model, dataset, device, out_path, n_images, per_movie_means, global_mean):
    n_images = min(n_images, len(dataset))
    selected = np.linspace(0, len(dataset) - 1, n_images, dtype=int)
    loader = DataLoader(torch.utils.data.Subset(dataset, selected.tolist()), batch_size=n_images, shuffle=False)
    model.eval()
    with torch.no_grad():
        x, y, prev_y = next(iter(loader))
        pred = image_from_model_output(model, model(x.to(device))[0]).cpu()
    err = (pred - y).abs()
    sample_mse = ((pred - y) ** 2).flatten(1).mean(dim=1).numpy()
    fig, axes = plt.subplots(n_images, 5, figsize=(12.5, 2.4 * n_images), squeeze=False)
    for i in range(n_images):
        local_movie_idx = int(dataset.samples[selected[i], 0])
        global_idx = int(dataset.movie_global_indices[local_movie_idx])
        current_mean = per_movie_means.get(global_idx, global_mean).cpu()
        panels = [
            (y[i, 0], "GT", "gray", 0, 1),
            (prev_y[i, 0], "Previous frame", "gray", 0, 1),
            (current_mean[0, 0], "Mean frame", "gray", 0, 1),
            (pred[i, 0], f"Recon | MSE {sample_mse[i]:.4f}", "gray", 0, 1),
            (err[i, 0], "Abs error", "magma", 0, 1),
        ]
        for j, (image, title, cmap, vmin, vmax) in enumerate(panels):
            axes[i, j].imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[i, j].set_title(title)
            axes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_config(path, args, train_dataset, val_dataset, criterion=None, align_report=None, temporal_smoothness=None, per_movie_metrics=None):
    response_lags = parse_response_lags(args.response_lag, len(MOVIE_FILES))
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        {
            "task_definition": (
                "Given a short post-stimulus RGC population response window, reconstruct the natural movie "
                "frame that elicited that response."
            ),
            "input_alignment": (
                "For frame t, input response window is response[t + response_lag - history_bins + 1 : "
                "t + response_lag + 1, neurons]. Note: response_lag is per-movie, see per_movie_lags."
            ),
            "target_definition": "target is movie_frame[t], normalized to [0, 1].",
            "model_family": (
                "frequency-split decoder: frozen image-space ridge low-frequency path plus gated mid/high residual paths"
                if args.decoder == "freq_split"
                else
                "main frame decoder with low-frequency auxiliary regularization and optional RF-aware residual gate"
                if args.decoder == "lowfreq_aux"
                else
                "coarse-plus-residual decoder with mean-frame prior, explicit edge/laplacian/high-frequency losses, and optional RF-aware residual gate"
                if args.decoder == "detail_residual"
                else
                "temporal RGC encoder with repeat-level latent pooling and an image decoder; "
                "dense_ae decoder uses dense coarse reconstruction followed by convolutional AE refinement"
                if args.decoder == "dense_ae"
                else "temporal RGC encoder with repeat-level latent pooling and spatial image decoder"
            ),
            "input_dim": train_dataset.input_dim,
            "n_cells": train_dataset.n_cells,
            "target_size": train_dataset.target_size,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "train_effective_repeats": train_dataset.sample_repeats,
            "val_effective_repeats": val_dataset.sample_repeats,
            "repeat_sampling": args.repeat_sampling,
            "per_movie_lags": response_lags,
            "leakage_control": (
                "by_movie/leave_one_movie_out split holds out entire movies"
                if args.split in ("by_movie", "leave_one_movie_out")
                else "scene split uses random non-overlapping contiguous scenes with train embargo"
                if args.split == "scene"
                else "time split uses contiguous temporal blocks with embargo"
            ),
            "trial_handling": (
                "single-trial repeats are retained; multiple repeats from the same frame are encoded separately "
                "and aggregated only in latent space"
            ),
            "control_protocol": (
                "normal response is compared with shuffled response, mean response, zero response, mean frame, "
                "and previous frame baselines to test neural dependence and video-prior strength."
            ),
            "rf_weighting": {
                "style": getattr(criterion, "rf_style", None),
                "source": getattr(criterion, "rf_source", None),
                "sigma": args.rf_sigma,
                "baseline": args.rf_baseline,
                "peak": args.rf_peak,
                "source_size": args.rf_source_size,
            }
            if criterion is not None
            else None,
            "frequency_split": {
                "enabled": args.decoder == "freq_split",
                "low_path": "ridge predicts low-frequency image-space values in [0,1], not logits",
                "low_mode": args.freq_low_mode,
                "composition": "pred = clamp(low + mid + gate * high, 0, 1)",
                "blur_sigma": args.freq_blur_sigma,
                "ridge_alpha": args.freq_ridge_alpha,
                "template_alpha": args.freq_template_alpha,
                "hybrid_template_weight": args.freq_hybrid_template_weight,
                "high_rf_radius": args.freq_high_rf_radius,
                "high_rf_softness": args.freq_high_rf_softness,
                "loss": (
                    "0.45*SSIM + 0.25*low + 0.15*mid + 0.05*high + 0.05*gradient + 0.03*residual + 0.02*gate"
                    if args.loss_mode == "freq_split_ssim_aware"
                    else None
                ),
            },
            "low_frequency_aux": {
                "enabled": args.decoder == "lowfreq_aux",
                "design": "main reconstruction logits + optional gated residual branch; low-frequency head is auxiliary regularization only",
                "low_aux_sigma": args.low_aux_sigma,
                "low_aux_weight": args.low_aux_weight,
                "low_aux_ssim_weight": args.low_aux_ssim_weight,
                "residual_gate": args.residual_gate,
                "residual_weight": args.residual_weight,
                "gate_l1_weight": args.gate_l1_weight,
            },
            "detail_residual": {
                "enabled": args.decoder == "detail_residual",
                "design": "pred = base/coarse + spike-conditioned residual; optimize coarse structure separately from high-frequency detail",
                "base_frame_prior_weight": args.base_frame_prior_weight,
                "low_aux_sigma": args.low_aux_sigma,
                "low_aux_weight": args.low_aux_weight,
                "low_aux_ssim_weight": args.low_aux_ssim_weight,
                "edge_loss_weight": args.edge_loss_weight,
                "laplacian_loss_weight": args.laplacian_loss_weight,
                "highfreq_loss_weight": args.highfreq_loss_weight,
                "highfreq_sigma": args.highfreq_sigma,
                "residual_gate": args.residual_gate,
                "residual_weight": args.residual_weight,
                "gate_l1_weight": args.gate_l1_weight,
            },
            "align_report": [
                {k: v for k, v in r.items()} for r in (align_report.records if align_report else [])
            ],
            "temporal_smoothness": temporal_smoothness,
            "per_movie_val_metrics": per_movie_metrics,
        }
    )
    path.write_text(json.dumps(config, indent=2))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    run_name = args.run_name or datetime.now().strftime("rgc_frame_decoder_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, align_report = make_datasets(args)
    align_report.dump()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args, train_dataset, device=device).to(device)
    criterion = build_loss(args, train_dataset)
    if isinstance(criterion, nn.Module):
        criterion = criterion.to(device)
    disc_optimizer = None
    is_adversarial = hasattr(criterion, "discriminator") and criterion.discriminator is not None
    if is_adversarial:
        criterion.discriminator = criterion.discriminator.to(device)
        disc_optimizer = torch.optim.AdamW(criterion.discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    write_config(run_dir / "config.json", args, train_dataset, val_dataset, criterion, align_report=align_report)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    per_movie_means, global_mean = compute_per_movie_mean_frames(train_dataset)
    mean_response_input = torch.zeros(args.history_bins, train_dataset.n_cells, dtype=torch.float32)
    zero_response = np.zeros((args.history_bins, train_dataset.n_cells), dtype=np.float32)
    if train_dataset.input_mean is not None and train_dataset.input_std is not None:
        zero_response = (zero_response - train_dataset.input_mean) / train_dataset.input_std
    zero_response_input = torch.from_numpy(zero_response.astype(np.float32))
    mean_frame_metrics = evaluate_prediction(model, val_loader, device, "mean_frame", global_mean=global_mean)

    print(f"\nRun directory: {run_dir}")
    print(
        f"Train {len(train_dataset)} frames | val {len(val_dataset)} frames | "
        f"train repeats/sample {train_dataset.sample_repeats} | eval repeats/sample {val_dataset.sample_repeats}"
    )
    print(f"Repeat sampling: {args.repeat_sampling}")
    per_movie_lags = parse_response_lags(args.response_lag, len(MOVIE_FILES))
    print(f"Per-movie response lags: {per_movie_lags}")
    print(f"Mean frame baseline PSNR {mean_frame_metrics['psnr']:.2f} SSIM {mean_frame_metrics['ssim']:.4f}")

    metrics_path = run_dir / "metrics.csv"
    best_path = run_dir / "best_model.pt"
    last_path = run_dir / "last_model.pt"
    viz_path = run_dir / "val_gt_vs_recon.png"
    best_val_mse = float("inf")
    bad_epochs = 0

    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_mse",
                "train_psnr",
                "train_ssim",
                "train_consistency",
                "val_loss",
                "val_mse",
                "val_psnr",
                "val_ssim",
                "val_consistency",
                "mean_frame_mse",
                "mean_frame_psnr",
                "mean_frame_ssim",
                "epoch_seconds",
                "lr",
            ],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            train_metrics = run_epoch(model, train_loader, optimizer, device, True, epoch, args, criterion, disc_optimizer)
            val_metrics = run_epoch(model, val_loader, optimizer, device, False, epoch, args, criterion)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_mse": train_metrics["mse"],
                "train_psnr": train_metrics["psnr"],
                "train_ssim": train_metrics["ssim"],
                "train_consistency": train_metrics["consistency"],
                "val_loss": val_metrics["loss"],
                "val_mse": val_metrics["mse"],
                "val_psnr": val_metrics["psnr"],
                "val_ssim": val_metrics["ssim"],
                "val_consistency": val_metrics["consistency"],
                "mean_frame_mse": mean_frame_metrics["mse"],
                "mean_frame_psnr": mean_frame_metrics["psnr"],
                "mean_frame_ssim": mean_frame_metrics["ssim"],
                "epoch_seconds": time.perf_counter() - start,
                "lr": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            f.flush()
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train MSE {row['train_mse']:.6f} SSIM {row['train_ssim']:.4f} | "
                f"val MSE {row['val_mse']:.6f} PSNR {row['val_psnr']:.2f} SSIM {row['val_ssim']:.4f} | "
                f"mean PSNR {mean_frame_metrics['psnr']:.2f} SSIM {mean_frame_metrics['ssim']:.4f}"
            )
            if val_metrics["mse"] < best_val_mse:
                best_val_mse = val_metrics["mse"]
                bad_epochs = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_mse": best_val_mse,
                        "config": json.loads((run_dir / "config.json").read_text()),
                    },
                    best_path,
                )
            else:
                bad_epochs += 1
            if args.save_last:
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, last_path)
            if args.patience > 0 and bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch:03d}; best val MSE {best_val_mse:.6f}.")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    save_visualization(model, val_dataset, device, viz_path, args.viz_samples, per_movie_means, global_mean)

    temporal_smoothness = compute_temporal_smoothness(model, val_dataset, device)
    if temporal_smoothness.get("warning"):
        print(f"[Temporal Smoothness] {temporal_smoothness['warning']}")
    print(
        f"[Temporal Smoothness] Inter-frame pred SSIM: {temporal_smoothness.get('inter_frame_ssim_mean', 0):.4f} "
        f"(GT: {temporal_smoothness.get('gt_inter_frame_ssim_mean', 0):.4f})"
    )

    controls = ["normal", "shuffle_response", "mean_response", "zero_response", "mean_frame", "previous_frame"]
    control_path = run_dir / "control_summary.csv"
    with control_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["control", "mse", "psnr", "ssim", "n"])
        writer.writeheader()
        for control in controls:
            metrics = evaluate_prediction(
                model,
                val_loader,
                device,
                control,
                global_mean=global_mean,
                mean_response_input=mean_response_input,
                zero_response_input=zero_response_input,
            )
            writer.writerow({"control": control, **metrics})

    per_movie_metrics = evaluate_prediction(
        model, val_loader, device, "normal",
        per_movie=True, dataset=val_dataset,
        per_movie_means=per_movie_means, global_mean=global_mean,
    )
    per_movie_path = run_dir / "per_movie_val_metrics.csv"
    if "per_movie" in per_movie_metrics:
        pm_data = per_movie_metrics["per_movie"]
        movie_ids = sorted(set(
            int(k.replace("movie", "").replace("_mse", "").replace("_psnr", "").replace("_ssim", "").replace("_n", ""))
            for k in pm_data if k.startswith("movie")
        ))
        with per_movie_path.open("w", newline="") as f:
            pm_writer = csv.DictWriter(f, fieldnames=["movie_id", "movie_name", "mse", "psnr", "ssim", "n"])
            pm_writer.writeheader()
            for mi in movie_ids:
                name = MOVIE_FILES[mi] if mi < len(MOVIE_FILES) else f"movie_{mi}"
                pm_writer.writerow({
                    "movie_id": str(mi),
                    "movie_name": name,
                    "mse": pm_data.get(f"movie{mi}_mse", ""),
                    "psnr": pm_data.get(f"movie{mi}_psnr", ""),
                    "ssim": pm_data.get(f"movie{mi}_ssim", ""),
                    "n": pm_data.get(f"movie{mi}_n", ""),
                })

    pm_summary = per_movie_metrics.get("per_movie", {}) if "per_movie" in per_movie_metrics else None

    rf_masked_metrics = {}
    rf_centers_for_mask, _ = resolve_rf_centers(args, train_dataset.n_cells)
    if rf_centers_for_mask is not None:
        rf_masked_path = run_dir / "rf_masked_metrics.csv"
        rf_masked_metrics = evaluate_masked(model, val_dataset, device, rf_centers_for_mask,
                                            image_size=args.image_size, rf_sigma=args.rf_sigma)
        print(
            f"[RF-Masked] RF-covered: PSNR {rf_masked_metrics.get('rf_psnr', 0):.2f} "
            f"SSIM {rf_masked_metrics.get('rf_ssim', 0):.4f} | "
            f"Global: PSNR {rf_masked_metrics.get('global_psnr', 0):.2f} "
            f"SSIM {rf_masked_metrics.get('global_ssim', 0):.4f}"
        )
        with rf_masked_path.open("w", newline="") as f:
            mw = csv.DictWriter(f, fieldnames=["region", "mse", "psnr", "ssim"])
            mw.writeheader()
            for region in ["rf", "global"]:
                mw.writerow({
                    "region": f"{region}_covered" if region == "rf" else region,
                    "mse": rf_masked_metrics.get(f"{region}_mse", ""),
                    "psnr": rf_masked_metrics.get(f"{region}_psnr", ""),
                    "ssim": rf_masked_metrics.get(f"{region}_ssim", ""),
                })

    write_config(run_dir / "config.json", args, train_dataset, val_dataset, criterion,
                 align_report=align_report, temporal_smoothness=temporal_smoothness,
                 per_movie_metrics=pm_summary)

    print(f"Best checkpoint: {best_path}")
    print(f"Metrics CSV: {metrics_path}")
    print(f"Validation visualization: {viz_path}")
    print(f"Control summary: {control_path}")
    print(f"Per-movie metrics: {per_movie_path}")
    if rf_masked_metrics:
        print(f"RF-masked metrics: {rf_masked_path}")

    return model, train_dataset, val_dataset, align_report


if __name__ == "__main__":
    main()
