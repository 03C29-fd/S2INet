import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["figure.dpi"] = 140
matplotlib.rcParams["savefig.dpi"] = 180

import matplotlib.pyplot as plt

from wisa_model import WISAAttentionDecoder, WISALiteDecoder
from adaptive_loss import ssim_index


MOVIE_FILES = [
    "MultipleMoviesStim_1_tree.avi",
    "MultipleMoviesStim_2_water.avi",
    "MultipleMoviesStim_3_grasses.avi",
    "MultipleMoviesStim_4_fish.avi",
    "MultipleMoviesStim_5_opticflow.avi",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train WISA-lite movie reconstruction from retinal responses.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--spikes-mat", type=Path, default=Path("data/movieBinnedSpiking.mat"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs_wisa"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--model", type=str, default="wisa", choices=["wisa", "wisa_attn"])
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--temporal-channels", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--ssim-loss-weight", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--split", type=str, default="by_movie", choices=["by_movie", "time"])
    parser.add_argument("--val-movies", type=str, default="5", help="1-based movie ids for validation, comma-separated.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Used only for --split time.")
    parser.add_argument("--embargo", type=int, default=30, help="Frame gap for time split.")
    parser.add_argument("--response-lag", type=int, default=0, help="Use response at t + lag for movie frame t.")
    parser.add_argument("--history-bins", type=int, default=3)
    parser.add_argument("--rep-mode", type=str, default="mean", choices=["mean", "all"])
    parser.add_argument("--normalize-inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--estimate-batches", type=int, default=10)
    parser.add_argument("--viz-samples", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache_movie_frames"))
    parser.add_argument("--save-last", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def choose_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def read_movie_frames(path, image_size, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{path.stem}_{image_size}x{image_size}_gray.npy"
    if cache_path.exists():
        return np.load(cache_path)

    backend = os.environ.get("MOVIE_FRAME_BACKEND", "ffmpeg")
    if backend == "ffmpeg":
        arr = read_movie_frames_ffmpeg(path, image_size)
    elif backend == "imageio":
        arr = read_movie_frames_imageio(path, image_size)
    elif backend == "opencv":
        arr = read_movie_frames_opencv(path, image_size)
    else:
        raise ValueError(f"Unknown MOVIE_FRAME_BACKEND={backend!r}. Use ffmpeg, imageio, or opencv.")

    np.save(cache_path, arr)
    return arr


def read_movie_frames_ffmpeg(path, image_size):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return read_movie_frames_imageio(path, image_size)

    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"scale={image_size}:{image_size},format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame_pixels = image_size * image_size
    if raw.size == 0 or raw.size % frame_pixels != 0:
        raise ValueError(f"ffmpeg returned invalid frame bytes for {path}: {raw.size}.")
    return raw.reshape(-1, image_size, image_size).astype(np.float32) / 255.0


def read_movie_frames_imageio(path, image_size):
    try:
        import imageio.v2 as imageio
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Install imageio, imageio-ffmpeg, and pillow, or provide ffmpeg on PATH.") from exc

    frames = []
    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        for frame in reader:
            if frame.ndim == 3:
                # ITU-R BT.601 luma approximation.
                frame = 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
            img = Image.fromarray(frame.astype(np.uint8), mode="L")
            img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
            frames.append(np.asarray(img, dtype=np.float32) / 255.0)
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"No frames could be read from {path}.")
    return np.stack(frames, axis=0)


def read_movie_frames_opencv(path, image_size):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Please install opencv-python-headless to use MOVIE_FRAME_BACKEND=opencv.") from exc
    cv2.setNumThreads(0)

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
        frames.append(gray.astype(np.float32) / 255.0)
    cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from {path}.")
    return np.stack(frames, axis=0)


class MovieReconstructionDataset(Dataset):
    def __init__(
        self,
        data_dir,
        spikes_mat,
        image_size,
        split,
        val_movies,
        val_ratio,
        embargo,
        response_lag,
        history_bins,
        rep_mode,
        cache_dir,
        max_samples=None,
        train=True,
    ):
        mat = loadmat(spikes_mat)
        binned = mat["binned"]
        nreps = mat["nreps"].reshape(-1).astype(int)

        if response_lag < 0:
            raise ValueError("--response-lag must be non-negative.")
        if history_bins < 1:
            raise ValueError("--history-bins must be at least 1.")

        self.response_lag = response_lag
        self.history_bins = history_bins
        self.rep_mode = rep_mode
        self.input_mean = None
        self.input_std = None
        self.frames = []
        self.responses = []

        for movie_idx, filename in enumerate(MOVIE_FILES):
            movie_path = data_dir / filename
            movie_frames = read_movie_frames(movie_path, image_size, cache_dir)
            reps = int(nreps[movie_idx])
            resp = binned[:reps, :, :, movie_idx].astype(np.float32)
            n_time = min(resp.shape[1], movie_frames.shape[0])
            self.responses.append(resp[:, :n_time])
            self.frames.append(movie_frames[:n_time])

        val_movie_ids = {int(x) - 1 for x in val_movies.split(",") if x.strip()}
        samples = []
        for movie_idx, resp in enumerate(self.responses):
            n_time = resp.shape[1]
            start = max(0, history_bins - 1 - response_lag)
            stop = n_time - response_lag
            if stop <= start:
                continue

            if split == "by_movie":
                use_movie = movie_idx in val_movie_ids
                if use_movie != (not train):
                    continue
                frame_indices = np.arange(start, stop, dtype=np.int64)
            else:
                val_size = max(1, int(round((stop - start) * val_ratio)))
                val_start = stop - val_size
                train_end = max(start, val_start - max(embargo, response_lag))
                if train:
                    frame_indices = np.arange(start, train_end, dtype=np.int64)
                else:
                    frame_indices = np.arange(val_start, stop, dtype=np.int64)

            if rep_mode == "mean":
                for t in frame_indices:
                    samples.append((movie_idx, -1, int(t)))
            else:
                for rep in range(resp.shape[0]):
                    for t in frame_indices:
                        samples.append((movie_idx, rep, int(t)))

        if max_samples is not None:
            samples = samples[:max_samples]
        if not samples:
            raise ValueError("No samples were created. Check split/lag/history settings.")

        self.samples = np.asarray(samples, dtype=np.int64)
        self.input_dim = self.responses[0].shape[2] * history_bins
        self.target_size = (image_size, image_size)
        self.split_name = "train" if train else "val"

    def __len__(self):
        return len(self.samples)

    def _feature(self, movie_idx, rep, t):
        resp = self.responses[movie_idx]
        chunks = []
        for offset in range(self.history_bins - 1, -1, -1):
            rt = t + self.response_lag - offset
            if rep < 0:
                chunks.append(resp[:, rt, :].mean(axis=0))
            else:
                chunks.append(resp[rep, rt, :])
        return np.concatenate(chunks, axis=0).astype(np.float32)

    def fit_input_normalization(self, chunk_size=8192):
        total = np.zeros((self.input_dim,), dtype=np.float64)
        total_sq = np.zeros((self.input_dim,), dtype=np.float64)
        count = 0
        for start in range(0, len(self.samples), chunk_size):
            batch = [self._feature(*row) for row in self.samples[start : start + chunk_size]]
            arr = np.stack(batch, axis=0)
            total += arr.sum(axis=0)
            total_sq += (arr * arr).sum(axis=0)
            count += arr.shape[0]
        mean = total / count
        var = np.maximum(total_sq / count - mean * mean, 1e-12)
        self.input_mean = mean.astype(np.float32)
        self.input_std = np.sqrt(var).astype(np.float32)

    def set_input_normalization(self, mean, std):
        self.input_mean = mean
        self.input_std = std

    def __getitem__(self, idx):
        movie_idx, rep, t = self.samples[idx]
        x = self._feature(int(movie_idx), int(rep), int(t))
        if self.input_mean is not None and self.input_std is not None:
            x = (x - self.input_mean) / self.input_std
        y = self.frames[int(movie_idx)][int(t)][None, :, :].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def psnr_from_mse(mse, data_range=1.0):
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10((data_range**2) / mse)


def build_model(args, input_dim, target_size):
    if input_dim % args.history_bins != 0:
        raise ValueError(f"input_dim={input_dim} is not divisible by history_bins={args.history_bins}.")
    n_cells = input_dim // args.history_bins
    model_cls = WISAAttentionDecoder if args.model == "wisa_attn" else WISALiteDecoder
    kwargs = {}
    if args.model == "wisa_attn":
        kwargs = {"num_heads": args.attention_heads, "num_layers": args.attention_layers}
    return model_cls(
        n_cells=n_cells,
        history_bins=args.history_bins,
        image_size=target_size,
        latent_dim=args.latent_dim,
        temporal_channels=args.temporal_channels,
        base_channels=args.base_channels,
        dropout=args.dropout,
        **kwargs,
    )


def run_epoch(model, loader, optimizer, device, train, epoch, args):
    model.train(train)
    criterion = nn.MSELoss()
    total_loss = total_mse = total_ssim = total_samples = 0.0
    estimate = None
    batch_times = []
    context = torch.enable_grad() if train else torch.no_grad()
    desc = f"epoch {epoch:03d} {'train' if train else 'val'}"
    with context:
        pbar = tqdm(loader, desc=desc, leave=False)
        for batch_idx, (x, y) in enumerate(pbar, start=1):
            start = time.perf_counter()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            pred = torch.sigmoid(model(x, x.shape[0]))
            loss = criterion(pred, y)
            if args.ssim_loss_weight > 0:
                loss = loss + args.ssim_loss_weight * (1.0 - ssim_index(pred, y).mean())
            if train:
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                mse = F.mse_loss(pred, y, reduction="mean")
                ssim = ssim_index(pred, y).mean()
            batch_size = x.shape[0]
            total_loss += loss.item() * batch_size
            total_mse += mse.item() * batch_size
            total_ssim += ssim.item() * batch_size
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
                    print(
                        f"Time estimate: {estimate['seconds_per_epoch']:.1f}s/epoch, "
                        f"{estimate['seconds_total_training'] / 60:.1f}m total"
                    )
    mse = total_mse / total_samples
    return {
        "loss": total_loss / total_samples,
        "mse": mse,
        "psnr": psnr_from_mse(mse),
        "ssim": total_ssim / total_samples,
        "time_estimate": estimate,
    }


def save_visualization(model, dataset, device, out_path, n_images):
    n_images = min(n_images, len(dataset))
    chosen = np.linspace(0, len(dataset) - 1, n_images, dtype=int)
    loader = DataLoader(torch.utils.data.Subset(dataset, chosen.tolist()), batch_size=n_images, shuffle=False)
    model.eval()
    with torch.no_grad():
        x, y = next(iter(loader))
        pred = torch.sigmoid(model(x.to(device), x.shape[0])).cpu()
    errors = (pred - y).abs()
    sample_mse = ((pred - y) ** 2).flatten(1).mean(dim=1).numpy()
    fig, axes = plt.subplots(n_images, 3, figsize=(7.5, 2.4 * n_images), squeeze=False)
    for row in range(n_images):
        axes[row, 0].imshow(y[row, 0], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title("GT")
        axes[row, 1].imshow(pred[row, 0], cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"Recon | MSE {sample_mse[row]:.4f}")
        axes[row, 2].imshow(errors[row, 0], cmap="magma", vmin=0, vmax=1)
        axes[row, 2].set_title("Abs error")
        for col in range(3):
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_config(path, args, train_dataset, val_dataset, device):
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        {
            "data_dir": str(args.data_dir),
            "spikes_mat": str(args.spikes_mat),
            "device_resolved": str(device),
            "input_dim": train_dataset.input_dim,
            "target_size": train_dataset.target_size,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "leakage_control": (
                "by_movie split holds out entire movies"
                if args.split == "by_movie"
                else "time split assigns frame blocks before expanding repetitions"
            ),
            "input_normalization": "fit on train split only" if train_dataset.input_mean is not None else None,
        }
    )
    with path.open("w") as f:
        json.dump(config, f, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    run_name = args.run_name or datetime.now().strftime("movie_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    common_dataset_args = dict(
        data_dir=args.data_dir,
        spikes_mat=args.spikes_mat,
        image_size=args.image_size,
        split=args.split,
        val_movies=args.val_movies,
        val_ratio=args.val_ratio,
        embargo=args.embargo,
        response_lag=args.response_lag,
        history_bins=args.history_bins,
        rep_mode=args.rep_mode,
        cache_dir=args.cache_dir,
        max_samples=args.max_samples,
    )
    train_dataset = MovieReconstructionDataset(**common_dataset_args, train=True)
    val_dataset = MovieReconstructionDataset(**common_dataset_args, train=False)

    if args.normalize_inputs:
        train_dataset.fit_input_normalization()
        val_dataset.set_input_normalization(train_dataset.input_mean, train_dataset.input_std)

    write_config(run_dir / "config.json", args, train_dataset, val_dataset, device)
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

    model = build_model(args, train_dataset.input_dim, train_dataset.target_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metrics_path = run_dir / "metrics.csv"
    best_path = run_dir / "best_model.pt"
    last_path = run_dir / "last_model.pt"
    timing_path = run_dir / "timing.json"
    viz_path = run_dir / "val_gt_vs_recon.png"

    print(f"Run directory: {run_dir}")
    print(
        f"Train {len(train_dataset)} | val {len(val_dataset)} | input {train_dataset.input_dim} | "
        f"target {train_dataset.target_size} | device {device}"
    )
    print(f"Split: {args.split} | val_movies={args.val_movies} | rep_mode={args.rep_mode}")

    best_val_mse = float("inf")
    bad_epochs = 0
    time_estimate = None
    start_all = time.perf_counter()
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_mse",
                "train_psnr",
                "train_ssim",
                "val_loss",
                "val_mse",
                "val_psnr",
                "val_ssim",
                "epoch_seconds",
                "lr",
            ],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            train_metrics = run_epoch(model, train_loader, optimizer, device, True, epoch, args)
            if train_metrics["time_estimate"] is not None:
                time_estimate = train_metrics["time_estimate"]
            val_metrics = run_epoch(model, val_loader, optimizer, device, False, epoch, args)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_mse": train_metrics["mse"],
                "train_psnr": train_metrics["psnr"],
                "train_ssim": train_metrics["ssim"],
                "val_loss": val_metrics["loss"],
                "val_mse": val_metrics["mse"],
                "val_psnr": val_metrics["psnr"],
                "val_ssim": val_metrics["ssim"],
                "epoch_seconds": time.perf_counter() - epoch_start,
                "lr": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            f.flush()
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train MSE {row['train_mse']:.6f} PSNR {row['train_psnr']:.2f} SSIM {row['train_ssim']:.4f} | "
                f"val MSE {row['val_mse']:.6f} PSNR {row['val_psnr']:.2f} SSIM {row['val_ssim']:.4f} | "
                f"{row['epoch_seconds']:.1f}s"
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

    timing = {"observed_total_seconds": time.perf_counter() - start_all, "initial_estimate": time_estimate}
    with timing_path.open("w") as f:
        json.dump(timing, f, indent=2)
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    save_visualization(model, val_dataset, device, viz_path, args.viz_samples)
    print(f"Best checkpoint: {best_path}")
    print(f"Metrics CSV: {metrics_path}")
    print(f"Validation visualization: {viz_path}")
    print(f"Timing: {timing_path}")


if __name__ == "__main__":
    main()
