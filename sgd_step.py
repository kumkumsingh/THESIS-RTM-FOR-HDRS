# sgd_step.py: run only the fitting steps (grid -> least squares -> SGD), with caching
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from pathlib import Path

import numpy as np
import pandas as pd
from keras import ops

from prfmodel.examples import load_dataset
from prfmodel.fitters import GridFitter, LeastSquaresFitter, SGDFitter, grid_values_2d_prf
from prfmodel.fitters.adapter import Adapter, ParameterTransform
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.models.prf import Gaussian2DPRFModel
from prfmodel.stimuli import PRFStimulus

pd.set_option("display.precision", 3)
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# 1. Data and stimulus (fast: the dataset is already cached on disk)
# ------------------------------------------------------------------
dataset = load_dataset("7t-aot-visual", surface_type="inflated")

stimulus = dataset.stimulus
is_on_screen = stimulus.design.max(axis=0) > 0.0
rows_on_screen = np.flatnonzero(is_on_screen.any(axis=1))
cols_on_screen = np.flatnonzero(is_on_screen.any(axis=0))
rows = slice(rows_on_screen[0], rows_on_screen[-1] + 1)
cols = slice(cols_on_screen[0], cols_on_screen[-1] + 1)
stimulus = PRFStimulus(
    design=stimulus.design[:, rows, cols],
    grid=stimulus.grid[rows, cols],
    frame_duration=stimulus.frame_duration,
    dimension_labels=stimulus.dimension_labels,
)

response_no_baseline = np.asarray(dataset.response) - 100.0
response_is_valid = response_no_baseline.std(axis=1) > 0.0
response_valid = response_no_baseline[response_is_valid]
print("Valid voxels:", response_valid.shape)

# ------------------------------------------------------------------
# 2. pRF model with TR = 0.9 s
# ------------------------------------------------------------------
prf_model = Gaussian2DPRFModel(impulse_model=DerivativeTwoGammaImpulse(resolution=0.9))

# ------------------------------------------------------------------
# 3. Grid search + least squares (computed once, then loaded from disk)
# ------------------------------------------------------------------
grid_file = RESULTS / "grid_params.pkl"
ls_file = RESULTS / "ls_params.pkl"

if ls_file.exists():
    ls_params = pd.read_pickle(ls_file)
    print("Loaded least-squares parameters from", ls_file)
else:
    if grid_file.exists():
        grid_params = pd.read_pickle(grid_file)
        print("Loaded grid parameters from", grid_file)
    else:
        grid_param_ranges = grid_values_2d_prf(
            stimulus, num_mu=21, mu_extent=2.0, num_sigma=10, sigma_range=(0.05, 4.0),
        ) | {"weight_deriv": [-0.5], "baseline": [0.0], "amplitude": [1.0]}
        grid_fitter = GridFitter(model=prf_model, stimulus=stimulus, compile_step=True)
        _, grid_params = grid_fitter.fit(data=response_valid, parameter_values=grid_param_ranges, batch_size=20)
        grid_params.to_pickle(grid_file)
        print("Saved grid parameters to", grid_file)

    ls_fitter = LeastSquaresFitter(model=prf_model, stimulus=stimulus)
    _, ls_params = ls_fitter.fit(
        data=response_valid,
        parameters=grid_params,
        slope_name="amplitude",
        intercept_name="baseline",
        batch_size=200,
    )
    ls_params.to_pickle(ls_file)
    print("Saved least-squares parameters to", ls_file)

# ------------------------------------------------------------------
# 4. SGD (the new part)
# ------------------------------------------------------------------
adapter = Adapter([ParameterTransform(["sigma"], transform_fun=ops.log, inverse_fun=ops.exp)])

sgd_fitter = SGDFitter(
    model=prf_model,
    stimulus=stimulus,
    adapter=adapter,
    compile_step=True,
)

sgd_history, sgd_params = sgd_fitter.fit(
    data=response_valid,
    init_parameters=ls_params,
    fixed_parameters=["weight_deriv"],   # keep weight_deriv at its starting value
    batch_size=20_000,
    num_steps=100,
)

sgd_params.to_pickle(RESULTS / "sgd_params.pkl")
print("SGD finished. Parameters per voxel:")
print(sgd_params)

# Exercise 20

import matplotlib.pyplot as plt
from prfmodel.plotting import plot_2d_prf_centers, plot_grid_parameter_distribution

# ------------------------------------------------------------------
# Load / align the three sets of estimates
# ------------------------------------------------------------------
grid_param_ranges = grid_values_2d_prf(
    stimulus, num_mu=21, mu_extent=2.0, num_sigma=10, sigma_range=(0.05, 4.0),
) | {"weight_deriv": [-0.5], "baseline": [0.0], "amplitude": [1.0]}

grid_params = pd.read_pickle(RESULTS / "grid_params.pkl")

n = len(sgd_params)        # works whether SGD ran on all voxels or on a subset
estimates = {
    "Grid search":   grid_params.iloc[:n].reset_index(drop=True),
    "Least squares": ls_params.iloc[:n].reset_index(drop=True),
    "SGD":           sgd_params.reset_index(drop=True),
}
data_n = response_valid[:n]

# ------------------------------------------------------------------
# 1. Distributions of centres and sizes (as in the hint)
# ------------------------------------------------------------------
x_lim = np.abs(stimulus.grid[..., 1]).max()
y_lim = np.abs(stimulus.grid[..., 0]).max()
mu_bins_x = np.linspace(-8, 8, 41)
mu_bins_y = np.linspace(-6, 6, 31)
sigma_bins = np.logspace(np.log10(0.02), np.log10(20), 40)

fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
for col, (name, p) in enumerate(estimates.items()):
    ax = axes[0, col]
    h = ax.hist2d(p["mu_x"], p["mu_y"], bins=[mu_bins_x, mu_bins_y], cmap="inferno",
                  norm=plt.matplotlib.colors.LogNorm())
    ax.add_patch(plt.Rectangle((-x_lim, -y_lim), 2 * x_lim, 2 * y_lim, fill=False, edgecolor="cyan", linewidth=2))
    ax.set(title=f"{name}: pRF centres", xlabel="mu_x (deg)", ylabel="mu_y (deg)")
    fig.colorbar(h[3], ax=ax, label="Number of voxels")

    ax = axes[1, col]
    ax.hist(p["sigma"].clip(sigma_bins[0], sigma_bins[-1]), bins=sigma_bins, color="gray")
    ax.set_xscale("log")
    ax.set(title=f"{name}: pRF sizes", xlabel="sigma (deg)", ylabel="Number of voxels")
plt.show()

# The practical's own helper for the grid estimates (they lie exactly on grid points)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
plot_2d_prf_centers(estimates["Grid search"], grid_param_ranges, stimulus=stimulus, ax=axes[0])
plot_grid_parameter_distribution(estimates["Grid search"], grid_param_ranges, "sigma", ax=axes[1])
fig.suptitle("Grid search (prfmodel helpers)")
plt.show()

# ------------------------------------------------------------------
# 2. Direct comparison voxel by voxel: before (LS) vs after (SGD)
# ------------------------------------------------------------------
before, after = estimates["Least squares"], estimates["SGD"]
params_to_compare = ["mu_x", "mu_y", "sigma", "baseline", "amplitude"]

fig, axes = plt.subplots(1, len(params_to_compare), figsize=(20, 4), layout="constrained")
for ax, name in zip(axes, params_to_compare):
    ax.scatter(before[name], after[name], s=2, alpha=0.3)
    lo = np.nanpercentile(np.r_[before[name], after[name]], 1)
    hi = np.nanpercentile(np.r_[before[name], after[name]], 99)
    ax.plot([lo, hi], [lo, hi], color="red", linewidth=1)          # identity line = no change
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel=f"{name} (grid + LS)", ylabel=f"{name} (SGD)", title=name)
    if name == "sigma":
        ax.set(xscale="log", yscale="log")
plt.show()

# ------------------------------------------------------------------
# 3. Numbers: how much did SGD change, and did the fit improve?
# ------------------------------------------------------------------
def r_squared(params, data, chunk=5000):
    out = []
    for i in range(0, len(params), chunk):
        pred = np.asarray(prf_model(stimulus, params.iloc[i:i + chunk].reset_index(drop=True)))
        d = data[i:i + chunk]
        ss_res = ((d - pred) ** 2).sum(axis=1)
        ss_tot = ((d - d.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        out.append(1 - ss_res / ss_tot)
    return np.concatenate(out)

r2_ls = r_squared(before, data_n)
r2_sgd = r_squared(after, data_n)

summary = pd.DataFrame({
    "mean |change|": [np.mean(np.abs(after[p] - before[p])) for p in params_to_compare],
    "% voxels changed": [100 * np.mean(~np.isclose(after[p], before[p])) for p in params_to_compare],
    "min (SGD)": [after[p].min() for p in params_to_compare],
    "max (SGD)": [after[p].max() for p in params_to_compare],
}, index=params_to_compare)
print(summary.round(3))

print(f"\nMedian R²  least squares: {np.median(r2_ls):.3f}   SGD: {np.median(r2_sgd):.3f}")
print(f"Voxels with R² > 0.3  least squares: {np.mean(r2_ls > 0.3)*100:.1f}%   SGD: {np.mean(r2_sgd > 0.3)*100:.1f}%")
print(f"Voxels improved by SGD: {np.mean(r2_sgd > r2_ls)*100:.1f}%")
print(f"sigma above old grid max (4°): {np.mean(after['sigma'] > 4.0)*100:.1f}%   "
      f"below old grid min (0.05°): {np.mean(after['sigma'] < 0.05)*100:.1f}%")