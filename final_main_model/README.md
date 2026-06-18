# RGC Population Response to Natural Movie Frame Reconstruction

This repository contains a reproducible baseline for reconstructing natural movie frames from retinal ganglion cell (RGC) population spike responses.

## Task

For each natural movie frame, the decoder receives a short post-stimulus RGC population response window and predicts the movie frame that elicited that response.

```text
input:  RGC population spike response window
target: corresponding natural movie frame
metrics: MSE, PSNR, SSIM, GT vs reconstruction visualization
split: scene/block split with embargo
controls: shuffled response, mean response, zero response, mean frame, previous frame
```

For frame `t`, the response window is aligned as:

```text
spike_t0 = t + response_lag - history_bins + 1
spike_t1 = t + response_lag + 1
x = response[spike_t0:spike_t1, neurons]
y = frame[t]
```

This tests whether RGC population activity contains decodable information about the eliciting natural image frame.

## Method Summary

The main optimized baseline uses:

- lag-corrected post-stimulus response windows;
- repeat-level encoding instead of directly averaging repetitions into PSTH;
- latent pooling across repeats;
- scene/block validation split with an embargo gap to reduce temporal leakage;
- paper-style RF-weighted MSE, SSIM, and Gabor feature loss;
- neural-dependence controls.

The RF-weighted loss follows the paper-style receptive-field weighting:

```text
each neuron has one RF center
-> place a Gaussian kernel around each RF center
-> sum Gaussian kernels over neurons
-> obtain a full-image RF weight matrix
-> L = 0.1 * LSSIM + 0.9 * RF-weighted MSE + optional Gabor feature loss
```

By default, RF centers are estimated from `data/binaryCheckerboard.mat` using a simple spike-triggered-average peak. You can also pass precomputed centers with `--rf-centers-path`.

Model:

```text
RGC response window
-> temporal encoder
-> repeat-level latent pooling
-> spatial frame decoder
-> reconstructed movie frame
```

An additional Dense-AE ablation is included:

```text
RGC response window
-> temporal encoder
-> repeat-level latent pooling
-> dense coarse image decoder
-> convolutional autoencoder-style refiner
-> reconstructed movie frame
```

The Dense-AE ablation was inspired by dense decoder + convolutional AE image refinement architectures.

## Repository Contents

```text
train_rgc_frame_decoder.py        Main training/evaluation script
wisa_model.py                     Temporal encoders and spatial decoders
adaptive_loss.py                  RF-weighted MSE and SSIM utilities
freq_split_decoder.py             Low/mid/high frequency-split decoder components
lowfreq_aux_decoder.py            Main decoder + low-frequency auxiliary + RF-aware residual gate
detail_residual_decoder.py        Coarse-plus-residual decoder with detail-preserving losses
train_movie_wisa.py               Movie-frame loading utilities and shared helpers
summarize_rgc_frame_runs.py       Summarize metrics and controls across runs
run_rgc_frame_decoder.sh          Main experiment launcher
run_rgc_linear_decoder.sh         Linear image-decoder baseline launcher
run_rgc_pure_linear_baseline.sh   Pure linear spike-window baseline launcher
run_rgc_denseae_decoder.sh        Dense-AE decoder ablation launcher
run_rgc_freq_split_decoder.sh     Frequency-split ridge + gated residual decoder launcher
run_rgc_lowfreq_aux_decoder.sh    Main decoder with auxiliary low-frequency regularization
run_rgc_detail_residual_decoder.sh Detail-preserving coarse + residual decoder launcher
run_rgc_lag_sweep.sh              Response-lag sweep launcher
run_rgc_seed_sweep.sh             Seed robustness launcher
requirements.txt                  Python dependencies
data/README.md                    Expected dataset files
results/README.md                 Where to place optional downloaded results
```

## Installation

Python 3.10+ is recommended. On a GPU machine:

```bash
conda create -n rgc-frame python=3.10 -y
conda activate rgc-frame
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build for your machine if the default pip install does not match your GPU driver.

The code does not require OpenCV. Movie frames are read with `ffmpeg` or `imageio-ffmpeg`.

## Data Setup

Place the following files under `data/`:

```text
data/movieBinnedSpiking.mat
data/binaryCheckerboard.mat
data/MultipleMoviesStim_1_tree.avi
data/MultipleMoviesStim_2_water.avi
data/MultipleMoviesStim_3_grasses.avi
data/MultipleMoviesStim_4_fish.avi
data/MultipleMoviesStim_5_opticflow.avi
```

`binaryCheckerboard.mat` is used to estimate per-neuron RF centers for the RF-weighted loss. These files are not included in this GitHub package because they are dataset artifacts and may be large. If you want the repository to include them, use Git LFS and check the dataset license/permission first.

## Quick Test

Run a small sanity check before full training:

```bash
chmod +x run_rgc_frame_decoder.sh run_rgc_denseae_decoder.sh run_rgc_lag_sweep.sh run_rgc_seed_sweep.sh

MAX_SAMPLES=512 EPOCHS=2 BATCH_SIZE=32 \
RUN_NAME=debug_rgc_frame_decoder \
bash run_rgc_frame_decoder.sh
```

Expected outputs:

```text
runs_rgc_frame/debug_rgc_frame_decoder/metrics.csv
runs_rgc_frame/debug_rgc_frame_decoder/control_summary.csv
runs_rgc_frame/debug_rgc_frame_decoder/val_gt_vs_recon.png
runs_rgc_frame/debug_rgc_frame_decoder/config.json
```

## Main Experiment

```bash
RUN_NAME=rgc_scene_lag3_h11_k8_rf_gabor \
SEED=1 EPOCHS=60 BATCH_SIZE=96 \
RESPONSE_LAG=3 HISTORY_BINS=11 \
bash run_rgc_frame_decoder.sh
```

Inspect:

```bash
cat runs_rgc_frame/rgc_scene_lag3_h11_k8_rf_gabor/control_summary.csv
tail -n 5 runs_rgc_frame/rgc_scene_lag3_h11_k8_rf_gabor/metrics.csv
```

## Low-Frequency Auxiliary Decoder

This variant keeps a normal main reconstruction decoder, then adds:

- a low-frequency auxiliary head trained only as a regularizer;
- an optional RF-aware residual gate to limit where residual corrections are expressed.

Conceptually:

```text
RGC response window
-> temporal encoder
-> repeat-level latent pooling
-> main frame decoder ----------------------> main logits
-> low-frequency auxiliary head -----------> low-pass target regularization
-> residual decoder + optional RF gate ---> gated residual logits
-> final logits = main logits + gated residual logits
```

Recommended first run:

```bash
RUN_NAME=rgc_lowfreqaux_lag3_h11_seed1 \
SEED=1 EPOCHS=60 BATCH_SIZE=96 \
RESPONSE_LAG=3 HISTORY_BINS=11 \
LOW_AUX_SIGMA=0 \
LOW_AUX_WEIGHT=0.10 \
RESIDUAL_GATE=rf_scalar \
bash run_rgc_lowfreq_aux_decoder.sh
```

Inspect:

```bash
cat runs_rgc_frame/rgc_lowfreqaux_lag3_h11_seed1/control_summary.csv
tail -n 5 runs_rgc_frame/rgc_lowfreqaux_lag3_h11_seed1/metrics.csv
```

## Detail-Preserving Residual Decoder

This variant is for the case where MSE/SSIM improve but reconstructions become too smooth. It makes the task explicit:

- `base/coarse` branch handles stable low-frequency structure;
- `residual` branch handles spike-conditioned detail;
- training adds edge, Laplacian, and high-frequency residual losses so the model is penalized for washing out detail.

Conceptually:

```text
pred = coarse_base + spike_conditioned_residual
```

Recommended first run:

```bash
RUN_NAME=rgc_detailres_lag3_h11_seed1 \
SEED=1 EPOCHS=60 BATCH_SIZE=96 \
RESPONSE_LAG=3 HISTORY_BINS=11 \
BASE_FRAME_PRIOR_WEIGHT=0.35 \
LOW_AUX_SIGMA=2.0 \
LOW_AUX_WEIGHT=0.12 \
EDGE_LOSS_WEIGHT=0.08 \
LAPLACIAN_LOSS_WEIGHT=0.05 \
HIGHFREQ_LOSS_WEIGHT=0.08 \
HIGHFREQ_SIGMA=2.0 \
RESIDUAL_GATE=rf_scalar \
bash run_rgc_detail_residual_decoder.sh
```

Inspect:

```bash
cat runs_rgc_frame/rgc_detailres_lag3_h11_seed1/control_summary.csv
tail -n 5 runs_rgc_frame/rgc_detailres_lag3_h11_seed1/metrics.csv
```

## Linear Decoder Baseline

This keeps the same RGC temporal encoder and repeat-level pooling, but replaces the image decoder with a single linear projection from latent space to the image.

```bash
RUN_NAME=rgc_linear_lag3_h11_k8_seed1 \
SEED=1 EPOCHS=60 BATCH_SIZE=96 \
RESPONSE_LAG=3 HISTORY_BINS=11 \
bash run_rgc_linear_decoder.sh
```

Inspect:

```bash
cat runs_rgc_frame/rgc_linear_lag3_h11_k8_seed1/control_summary.csv
```

## Pure Linear Baseline

This is the cleanest baseline: it averages repeats at the input level, flattens the aligned spike-response window, and maps it directly to the image with one linear layer. It does not use the temporal encoder, repeat-attention pooling, or spatial decoder.

```bash
RUN_NAME=rgc_pure_linear_lag3_h11_k8_seed1 \
SEED=1 EPOCHS=60 BATCH_SIZE=96 \
RESPONSE_LAG=3 HISTORY_BINS=11 \
bash run_rgc_pure_linear_baseline.sh
```

Inspect:

```bash
cat runs_rgc_frame/rgc_pure_linear_lag3_h11_k8_seed1/control_summary.csv
tail -n 5 runs_rgc_frame/rgc_pure_linear_lag3_h11_k8_seed1/metrics.csv
```

## Lag Sweep

```bash
EPOCHS=60 BATCH_SIZE=96 \
LAGS="0 1 2 3 4 5 6" \
BASE_RUN_PREFIX=rgc_lag_sweep \
bash run_rgc_lag_sweep.sh
```

Inspect:

```bash
cat runs_rgc_frame/rgc_lag_sweep_summary.csv
```

## Seed Robustness

```bash
EPOCHS=60 BATCH_SIZE=96 \
RESPONSE_LAG=3 \
SEEDS="1 2 3" \
BASE_RUN_PREFIX=rgc_seed_sweep_lag3_fixedseed \
bash run_rgc_seed_sweep.sh
```

Inspect:

```bash
cat runs_rgc_frame/rgc_seed_sweep_lag3_fixedseed_summary.csv
```

## Dense-AE Decoder Ablation

```bash
RUN_NAME=rgc_denseae_lag3_h11_k8_seed1 \
SEED=1 EPOCHS=60 BATCH_SIZE=96 \
RESPONSE_LAG=3 HISTORY_BINS=11 \
bash run_rgc_denseae_decoder.sh
```

Inspect:

```bash
cat runs_rgc_frame/rgc_denseae_lag3_h11_k8_seed1/control_summary.csv
```

## Frequency-Split Decoder Ablation

This decoder explicitly separates low, mid, and high spatial frequencies:

```text
low:  ridge regression predicts blurred image-space values in [0,1]
low-template: optional RF-template basis low = sum_i coeff_i * RF_template_i
mid:  temporal encoder + small spatial decoder predicts residual structure
high: smaller residual decoder predicts gated high-frequency detail
pred: clamp(low + mid + gate * high, 0, 1)
```

The low path is not treated as a logit. This avoids applying sigmoid twice to the stable low-frequency reconstruction.
By default the frequency-split launcher sets `FREQ_BLUR_SIGMA=0`, so the low path is trained as a regularized image-space reconstruction rather than a blurred target. In that no-blur mode, mid/high residuals are regularized toward zero and only correct the low path when they improve SSIM/gradient reconstruction.

Run:

```bash
RUN_NAME=rgc_freqsplit_lag3_h11_seed1 \
SEED=1 EPOCHS=60 BATCH_SIZE=96 \
RESPONSE_LAG=3 HISTORY_BINS=11 \
bash run_rgc_freq_split_decoder.sh
```

Recommended conservative settings are built into the launcher:

```text
low_mode=hybrid
blur_sigma=0
latent_dim=256
temporal_channels=128
base_channels=64
dropout=0.25
weight_decay=1e-3
loss=freq_split_ssim_aware
```

Inspect:

```bash
cat runs_rgc_frame/rgc_freqsplit_lag3_h11_seed1/control_summary.csv
tail -n 5 runs_rgc_frame/rgc_freqsplit_lag3_h11_seed1/metrics.csv
```

Low-mode / blur / RF-radius sweep:

```bash
LOW_MODES="ridge rf_template hybrid" \
BLURS="3.0 5.0 7.0" \
RADII="0 4 6 8 10" \
BASE_RUN_PREFIX=fs_lag3_h11 \
SEED=42 EPOCHS=60 BATCH_SIZE=64 \
RESPONSE_LAG=3 HISTORY_BINS=11 TRAIN_REPEATS=8 \
bash run_rgc_freq_split_sweep.sh
```

## How to Interpret Results

The most important comparison is not only the absolute MSE/PSNR/SSIM, but whether the normal response outperforms non-neural controls:

```text
normal response > shuffled response
normal response > mean response
normal response > zero response
normal response > mean frame
```

The `previous_frame` control is usually much stronger because natural movies are temporally continuous. It is a strong non-neural video prior, not evidence that RGC responses contain no information.

In our experiments, the optimized baseline consistently outperformed shuffled/mean/zero/mean-frame controls, indicating that RGC population responses contain decodable information about the eliciting natural movie frame. However, reconstruction quality remained moderate, and previous-frame prediction remained substantially stronger.

## Main Files for Report Reproduction

For a minimal reproducible submission, include:

```text
train_rgc_frame_decoder.py
wisa_model.py
adaptive_loss.py
train_movie_wisa.py
summarize_rgc_frame_runs.py
run_rgc_frame_decoder.sh
run_rgc_linear_decoder.sh
run_rgc_pure_linear_baseline.sh
run_rgc_lag_sweep.sh
run_rgc_seed_sweep.sh
run_rgc_denseae_decoder.sh
requirements.txt
data/README.md
```

Optional result artifacts for a report:

```text
runs_rgc_frame/*/metrics.csv
runs_rgc_frame/*/control_summary.csv
runs_rgc_frame/*/val_gt_vs_recon.png
runs_rgc_frame/*/config.json
```

Do not commit large checkpoints (`*.pt`) unless necessary. Use release assets or Git LFS for checkpoints.
