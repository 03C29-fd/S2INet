import argparse
import csv
import json
from pathlib import Path


def read_best_metrics(metrics_path):
    if not metrics_path.exists():
        return {}
    with metrics_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    best = min(rows, key=lambda row: float(row["val_mse"]))
    last = rows[-1]
    return {
        "epochs_done": last.get("epoch", ""),
        "best_epoch": best.get("epoch", ""),
        "best_val_mse": best.get("val_mse", ""),
        "best_val_psnr": best.get("val_psnr", ""),
        "best_val_ssim": best.get("val_ssim", ""),
        "last_val_mse": last.get("val_mse", ""),
        "last_val_psnr": last.get("val_psnr", ""),
        "last_val_ssim": last.get("val_ssim", ""),
        "mean_frame_mse": best.get("mean_frame_mse", ""),
        "mean_frame_psnr": best.get("mean_frame_psnr", ""),
        "mean_frame_ssim": best.get("mean_frame_ssim", ""),
    }


def read_controls(control_path):
    controls = {}
    if not control_path.exists():
        return controls
    with control_path.open(newline="") as f:
        for row in csv.DictReader(f):
            name = row["control"]
            controls[f"{name}_mse"] = row.get("mse", "")
            controls[f"{name}_psnr"] = row.get("psnr", "")
            controls[f"{name}_ssim"] = row.get("ssim", "")
    return controls


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def add_deltas(row):
    normal_mse = safe_float(row.get("normal_mse"))
    normal_psnr = safe_float(row.get("normal_psnr"))
    normal_ssim = safe_float(row.get("normal_ssim"))
    for control in ["shuffle_response", "mean_response", "zero_response", "mean_frame", "previous_frame"]:
        mse = safe_float(row.get(f"{control}_mse"))
        psnr = safe_float(row.get(f"{control}_psnr"))
        ssim = safe_float(row.get(f"{control}_ssim"))
        if normal_mse is not None and mse is not None and mse != 0:
            row[f"normal_vs_{control}_mse_reduction_pct"] = f"{(mse - normal_mse) / mse * 100.0:.3f}"
        else:
            row[f"normal_vs_{control}_mse_reduction_pct"] = ""
        row[f"normal_vs_{control}_psnr_gain"] = f"{normal_psnr - psnr:.3f}" if normal_psnr is not None and psnr is not None else ""
        row[f"normal_vs_{control}_ssim_gain"] = f"{normal_ssim - ssim:.4f}" if normal_ssim is not None and ssim is not None else ""
    return row


def summarize(root, prefix=None):
    runs = []
    for run_dir in sorted(Path(root).iterdir()):
        if not run_dir.is_dir():
            continue
        if prefix and not run_dir.name.startswith(prefix):
            continue
        config_path = run_dir / "config.json"
        config = {}
        if config_path.exists():
            config = json.loads(config_path.read_text())
        row = {
            "run": run_dir.name,
            "split": config.get("split", ""),
            "seed": config.get("seed", ""),
            "response_lag": config.get("response_lag", ""),
            "history_bins": config.get("history_bins", ""),
            "train_repeats": config.get("train_effective_repeats", config.get("train_repeats", "")),
            "eval_repeats": config.get("val_effective_repeats", config.get("eval_repeats", "")),
            "encoder": config.get("encoder", ""),
            "decoder": config.get("decoder", ""),
            "loss_mode": config.get("loss_mode", ""),
            "loss_mu": config.get("loss_mu", ""),
            "gabor_loss_weight": config.get("gabor_loss_weight", ""),
            "viz": str(run_dir / "val_gt_vs_recon.png"),
        }
        row.update(read_best_metrics(run_dir / "metrics.csv"))
        row.update(read_controls(run_dir / "control_summary.csv"))
        runs.append(add_deltas(row))
    return runs


def main():
    parser = argparse.ArgumentParser(description="Summarize RGC frame-decoder runs with neural-dependence controls.")
    parser.add_argument("root", type=Path, nargs="?", default=Path("runs_rgc_frame"))
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = summarize(args.root, args.prefix)
    if not rows:
        print("No runs found.")
        return

    preferred = [
        "run",
        "split",
        "seed",
        "response_lag",
        "history_bins",
        "train_repeats",
        "eval_repeats",
        "encoder",
        "decoder",
        "epochs_done",
        "best_epoch",
        "best_val_mse",
        "best_val_psnr",
        "best_val_ssim",
        "normal_mse",
        "normal_psnr",
        "normal_ssim",
        "shuffle_response_mse",
        "shuffle_response_psnr",
        "shuffle_response_ssim",
        "mean_frame_mse",
        "mean_frame_psnr",
        "mean_frame_ssim",
        "previous_frame_mse",
        "previous_frame_psnr",
        "previous_frame_ssim",
        "normal_vs_shuffle_response_mse_reduction_pct",
        "normal_vs_shuffle_response_psnr_gain",
        "normal_vs_shuffle_response_ssim_gain",
        "normal_vs_mean_frame_mse_reduction_pct",
        "normal_vs_mean_frame_psnr_gain",
        "normal_vs_mean_frame_ssim_gain",
        "normal_vs_previous_frame_mse_reduction_pct",
        "normal_vs_previous_frame_psnr_gain",
        "normal_vs_previous_frame_ssim_gain",
        "loss_mode",
        "loss_mu",
        "gabor_loss_weight",
        "viz",
    ]
    extras = sorted({key for row in rows for key in row if key not in preferred})
    fieldnames = preferred + extras

    out_path = args.out
    if out_path is None and args.prefix:
        out_path = args.root / f"{args.prefix}_summary.csv"
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
