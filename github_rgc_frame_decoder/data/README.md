# Data Files

Place the required dataset files in this directory before running training:

```text
movieBinnedSpiking.mat
binaryCheckerboard.mat
MultipleMoviesStim_1_tree.avi
MultipleMoviesStim_2_water.avi
MultipleMoviesStim_3_grasses.avi
MultipleMoviesStim_4_fish.avi
MultipleMoviesStim_5_opticflow.avi
```

The training script expects:

```text
data/movieBinnedSpiking.mat
data/binaryCheckerboard.mat
data/MultipleMoviesStim_1_tree.avi
data/MultipleMoviesStim_2_water.avi
data/MultipleMoviesStim_3_grasses.avi
data/MultipleMoviesStim_4_fish.avi
data/MultipleMoviesStim_5_opticflow.avi
```

`binaryCheckerboard.mat` is used to estimate one RF center per neuron for the Gaussian RF weight matrix. These files are not included by default because they may be large and may have dataset-specific redistribution restrictions.

If you choose to track them in GitHub, use Git LFS:

```bash
git lfs track "*.mat"
git lfs track "*.avi"
```
