import csv
import json
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_rgc_frame_decoder import (
    compute_per_movie_mean_frames,
    evaluate_prediction,
    make_datasets,
    parse_args,
    psnr_from_mse,
    save_visualization,
    ssim_index,
    set_seed,
    choose_device,
)


class PureLinearRGCDecoder(nn.Module):
    def __init__(self, n_cells, history_bins, image_size):
        super().__init__()
        self.n_cells = n_cells
        self.history_bins = history_bins
        self.image_size = image_size
        self.fc = nn.Linear(n_cells * history_bins, image_size[0] * image_size[1])

    def forward(self, x, return_single=False):
        # x: [B, repeats, history, cells]. Average repeats at input level for a true linear baseline.
        pooled = x.mean(dim=1)
        flat = pooled.reshape(pooled.shape[0], -1)
        logits = self.fc(flat).reshape(flat.shape[0], 1, self.image_size[0], self.image_size[1])
        if return_single:
            b, k, h, c = x.shape
            single_logits = self.fc(x.reshape(b * k, h * c)).reshape(b, k, 1, self.image_size[0], self.image_size[1])
            dummy_latents = flat[:, None, :]
            return logits, single_logits, dummy_latents, flat
        return logits, None, flat


def plain_mse_ssim_loss(pred, target, ssim_weight=0.0):
    loss = F.mse_loss(pred, target)
    if ssim_weight > 0:
        loss = loss + ssim_weight * (1.0 - ssim_index(pred, target).mean())
    return loss


def run_epoch(model, loader, optimizer, device, train, epoch, args):
    model.train(train)
    total_loss = total_mse = total_ssim = 0.0
    total_samples = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y, _prev_y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)

            logits, single_logits, _latents, _pooled = model(x, return_single=True)
            pred = torch.sigmoid(logits)
            loss = plain_mse_ssim_loss(pred, y, args.ssim_loss_weight)

            if args.single_repeat_loss_weight > 0:
                single_pred = torch.sigmoid(single_logits)
                single_target = y[:, None, :, :, :].expand_as(single_pred)
                loss = loss + args.single_repeat_loss_weight * plain_mse_ssim_loss(
                    single_pred.reshape(-1, *y.shape[1:]),
                    single_target.reshape(-1, *y.shape[1:]),
                    args.ssim_loss_weight,
                )

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

    mse = total_mse / total_samples
    return {
        "loss": total_loss / total_samples,
        "mse": mse,
        "psnr": psnr_from_mse(mse),
        "ssim": total_ssim / total_samples,
    }


def write_config(path, args, train_dataset, val_dataset):
    config = {key: str(value) for key, value in vars(args).items()}
    config.update(
        {
            "task_definition": "Pure linear baseline for RGC response window to movie frame reconstruction.",
            "model_family": "mean repeat response window flattened directly into one linear image projection",
            "input_alignment": (
                "For frame t, input response window is response[t + response_lag - history_bins + 1 : "
                "t + response_lag + 1, neurons]."
            ),
            "input_dim": train_dataset.input_dim,
            "n_cells": train_dataset.n_cells,
            "target_size": train_dataset.target_size,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "train_effective_repeats": train_dataset.sample_repeats,
            "val_effective_repeats": val_dataset.sample_repeats,
        }
    )
    path.write_text(json.dumps(config, indent=2))


def main():
    args = parse_args()
    args.output_dir = args.output_dir
    set_seed(args.seed)
    device = choose_device(args.device)
    run_name = args.run_name or datetime.now().strftime("rgc_pure_linear_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, _align_report = make_datasets(args)
    write_config(run_dir / "config.json", args, train_dataset, val_dataset)

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

    model = PureLinearRGCDecoder(train_dataset.n_cells, args.history_bins, train_dataset.target_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    per_movie_means, global_mean = compute_per_movie_mean_frames(train_dataset)
    mean_frame_metrics = evaluate_prediction(model, val_loader, device, "mean_frame", global_mean=global_mean)
    mean_response_input = torch.zeros(args.history_bins, train_dataset.n_cells, dtype=torch.float32)
    zero_response_raw = np.zeros((args.history_bins, train_dataset.n_cells), dtype=np.float32)
    if train_dataset.input_mean is not None and train_dataset.input_std is not None:
        zero_response_raw = (zero_response_raw - train_dataset.input_mean) / train_dataset.input_std
    zero_response = torch.from_numpy(zero_response_raw.astype("float32"))

    print(f"Run directory: {run_dir}")
    print(f"Pure linear baseline | train {len(train_dataset)} | val {len(val_dataset)} | device {device}")

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
                "val_loss",
                "val_mse",
                "val_psnr",
                "val_ssim",
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
            train_metrics = run_epoch(model, train_loader, optimizer, device, True, epoch, args)
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
                f"val MSE {row['val_mse']:.6f} PSNR {row['val_psnr']:.2f} SSIM {row['val_ssim']:.4f}"
            )
            if val_metrics["mse"] < best_val_mse:
                best_val_mse = val_metrics["mse"]
                bad_epochs = 0
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "best_val_mse": best_val_mse}, best_path)
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

    control_path = run_dir / "control_summary.csv"
    with control_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["control", "mse", "psnr", "ssim", "n"])
        writer.writeheader()
        for control in ["normal", "shuffle_response", "mean_response", "zero_response", "mean_frame", "previous_frame"]:
            metrics = evaluate_prediction(
                model,
                val_loader,
                device,
                control,
                global_mean=global_mean,
                mean_response_input=mean_response_input,
                zero_response_input=zero_response,
            )
            writer.writerow({"control": control, **metrics})

    print(f"Best checkpoint: {best_path}")
    print(f"Metrics CSV: {metrics_path}")
    print(f"Validation visualization: {viz_path}")
    print(f"Control summary: {control_path}")


if __name__ == "__main__":
    main()
