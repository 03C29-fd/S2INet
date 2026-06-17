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
train_movie_wisa.py               Movie-frame loading utilities and shared helpers
summarize_rgc_frame_runs.py       Summarize metrics and controls across runs
run_rgc_frame_decoder.sh          Main experiment launcher
run_rgc_linear_decoder.sh         Linear image-decoder baseline launcher
run_rgc_pure_linear_baseline.sh   Pure linear spike-window baseline launcher
run_rgc_denseae_decoder.sh        Dense-AE decoder ablation launcher
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
