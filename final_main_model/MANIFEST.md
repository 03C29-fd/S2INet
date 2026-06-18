# GitHub Package Manifest

## Included Code

```text
train_rgc_frame_decoder.py
wisa_model.py
adaptive_loss.py
train_movie_wisa.py
freq_split_decoder.py
lowfreq_aux_decoder.py
detail_residual_decoder.py
summarize_rgc_frame_runs.py
run_rgc_frame_decoder.sh
run_rgc_linear_decoder.sh
run_rgc_pure_linear_baseline.sh
run_rgc_denseae_decoder.sh
run_rgc_freq_split_decoder.sh
run_rgc_lowfreq_aux_decoder.sh
run_rgc_detail_residual_decoder.sh
run_rgc_freq_split_sweep.sh
run_rgc_lag_sweep.sh
run_rgc_seed_sweep.sh
requirements.txt
requirements_movie.txt
README.md
data/README.md
results/README.md
.gitignore
```

## Required Data Not Included

Place these files under `data/`:

```text
movieBinnedSpiking.mat
binaryCheckerboard.mat
MultipleMoviesStim_1_tree.avi
MultipleMoviesStim_2_water.avi
MultipleMoviesStim_3_grasses.avi
MultipleMoviesStim_4_fish.avi
MultipleMoviesStim_5_opticflow.avi
```

## Recommended Result Artifacts To Download Before Final Report

These are not currently present in this local GitHub package unless you copy/download them from the training machine:

```text
runs_rgc_frame/rgc_lag_sweep_summary.csv
runs_rgc_frame/rgc_seed_sweep_lag3_fixedseed_summary.csv
runs_rgc_frame/<main_run>/metrics.csv
runs_rgc_frame/<main_run>/control_summary.csv
runs_rgc_frame/<main_run>/val_gt_vs_recon.png
runs_rgc_frame/<main_run>/config.json
runs_rgc_frame/<denseae_run>/metrics.csv
runs_rgc_frame/<denseae_run>/control_summary.csv
runs_rgc_frame/<denseae_run>/val_gt_vs_recon.png
runs_rgc_frame/<denseae_run>/config.json
```

Suggested main runs from the experiments discussed:

```text
rgc_lag_sweep_lag0_h11_k8
rgc_lag_sweep_lag1_h11_k8
rgc_lag_sweep_lag3_h11_k8
rgc_seed_sweep_lag3_fixedseed_seed1_h11_k8
rgc_seed_sweep_lag3_fixedseed_seed2_h11_k8
rgc_seed_sweep_lag3_fixedseed_seed3_h11_k8
rgc_denseae_lag3_h11_k8_seed1
```

Do not commit `*.pt` checkpoints unless specifically required.
