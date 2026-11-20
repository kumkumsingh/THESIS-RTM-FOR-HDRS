"""Evaluating the model fit (Exercises 21-23).

Self-contained: does not need common.py.
Needs in the same folder:
  - plotting_helpers.py
  - results/ls_params.pkl   (made by sgd_step.py)
  - results/sgd_params.pkl  (made by sgd_step.py, only for Exercise 23)
"""
import os

os.environ["KERAS_BACKEND"] = "tensorflow"   # must be set before importing prfmodel / keras

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting_helpers import make_surface_plot, make_surface_projection
from prfmodel.examples import load_dataset
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.models.prf import Gaussian2DPRFModel
from prfmodel.plotting import plot_r_squared_hist
from prfmodel.stimuli import PRFStimulus
from prfmodel.utils import batched, calculate_r_squared

pd.set_option("display.precision", 3)

RESULTS = Path("results")   # folder with ls_params.pkl and sgd_params.pkl

# ------------------------------------------------------------------
# Setup: data, cropped stimulus, valid voxels, pRF model
# ------------------------------------------------------------------
dataset = load_dataset("7t-aot-visual", surface_type="inflated")   # cached, so fast

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
time_frames = np.arange(stimulus.design.shape[0])

response_no_baseline = np.asarray(dataset.response) - 100.0
response_is_valid = response_no_baseline.std(axis=1) > 0.0
response_valid = response_no_baseline[response_is_valid]

prf_model = Gaussian2DPRFModel(impulse_model=DerivativeTwoGammaImpulse(resolution=0.9))


def fill_valid_voxels(values: np.ndarray) -> np.ndarray:
    """Fill values for the valid voxels into an array with one value per voxel in the mask."""
    values_full = np.full((response_no_baseline.shape[0],), fill_value=np.nan)
    values_full[response_is_valid] = values
    return values_full


# ------------------------------------------------------------------
# Load the fitted parameters (no refitting)
# ------------------------------------------------------------------
ls_params = pd.read_pickle(RESULTS / "ls_params.pkl")
sgd_file = RESULTS / "sgd_params.pkl"
sgd_params = pd.read_pickle(sgd_file) if sgd_file.exists() else None

predict_batched = batched(prf_model)

# ------------------------------------------------------------------
# Predictions and R-squared for the grid search + least-squares fit
# ------------------------------------------------------------------
pred_response = np.asarray(predict_batched(stimulus, ls_params, batch_size=200))
r_squared = np.asarray(calculate_r_squared(response_valid, pred_response))  # one score per voxel
print("R-squared shape:", r_squared.shape)
print(f"Median R²: {np.median(r_squared):.3f}   max: {r_squared.max():.3f}   "
      f"R² > 0.3: {np.mean(r_squared > 0.3) * 100:.1f}% of voxels")

plot_r_squared_hist(r_squared, log=True)
plt.show()

# ------------------------------------------------------------------
# Exercise 21: R-squared on the inflated surface
# ------------------------------------------------------------------
project_to_surface = make_surface_projection(dataset)
plot_surf_stat_map_helper = make_surface_plot(dataset.mesh)

plot_surf_stat_map_helper(project_to_surface(fill_valid_voxels(r_squared)), vmax=0.6,
                          title="R-squared (grid search + least squares)")
plt.show()

# ------------------------------------------------------------------
# Exercise 22: observed vs predicted for voxels with different fit quality
# ------------------------------------------------------------------
order = np.argsort(r_squared)                       # from worst to best fit
picks = {
    "worst": order[0],
    "low (10th percentile)": order[int(0.10 * len(order))],
    "median": order[len(order) // 2],
    "good (99th percentile)": order[int(0.99 * len(order))],
    "best": order[-1],
}

fig, axes = plt.subplots(len(picks), 1, figsize=(12, 2.4 * len(picks)), sharex=True, layout="constrained")
for ax, (label, v) in zip(axes, picks.items()):
    ax.plot(time_frames, response_valid[v], color="gray", linewidth=1, label="Observed")
    ax.plot(time_frames, pred_response[v], color="tab:blue", linewidth=2, label="Predicted")
    ax.set_title(f"{label}: voxel {v}, R² = {r_squared[v]:.2f}", loc="left", fontsize=10)
    ax.set_ylabel("BOLD (%)")
axes[0].legend(loc="upper right")
axes[-1].set_xlabel("Time frame")
plt.show()

# ------------------------------------------------------------------
# Exercise 23: SGD fit vs grid search + least-squares fit
# ------------------------------------------------------------------
if sgd_params is None:
    print("No results/sgd_params.pkl found: run sgd_step.py first.")
else:
    n = len(sgd_params)  # SGD may have been run on a subset of the voxels
    pred_sgd = np.asarray(predict_batched(stimulus, sgd_params.reset_index(drop=True), batch_size=200))
    r_squared_sgd = np.asarray(calculate_r_squared(response_valid[:n], pred_sgd))
    r_squared_ls = r_squared[:n]

    print(f"\nMedian R²   LS: {np.median(r_squared_ls):.3f}   SGD: {np.median(r_squared_sgd):.3f}")
    print(f"R² > 0.3    LS: {np.mean(r_squared_ls > 0.3) * 100:.1f}%   SGD: {np.mean(r_squared_sgd > 0.3) * 100:.1f}%")
    print(f"Voxels improved by SGD: {np.mean(r_squared_sgd > r_squared_ls) * 100:.1f}%   "
          f"worse: {np.mean(r_squared_sgd < r_squared_ls) * 100:.1f}%")

    fig, ax = plt.subplots(figsize=(5.5, 5.5), layout="constrained")
    ax.scatter(r_squared_ls, r_squared_sgd, s=2, alpha=0.3)
    ax.plot([-0.1, 1.0], [-0.1, 1.0], color="red", linewidth=1)  # points above the line: SGD better
    ax.set(xlabel="R² grid search + least squares", ylabel="R² SGD", title="Fit before vs after SGD",
           xlim=(-0.1, 1), ylim=(-0.1, 1))
    plt.show()

    # The R² distribution is strongly skewed: most voxels have R² close to zero, reflecting grey-matter voxels outside the visual cortex or noisy voxels without stimulus-driven responses. A smaller group of voxels forms a plateau between about 0.2 and 0.8, with the best fits reaching about 0.9. These are voxels whose BOLD response is well explained by a Gaussian pRF and are expected to lie in the visual cortex. There are no negative R² values, because baseline and amplitude were fitted to each voxel’s data with least squares.


from prfmodel.fitters import grid_values_2d_prf

grid_param_ranges = grid_values_2d_prf(
    stimulus,
    num_mu=21,
    mu_extent=2.0,  # Span twice the visual field covered by the stimulus
    num_sigma=10,
    sigma_range=(0.05, 4.0),  # Log-spaced by default
) | {  # Add constant parameters
    "weight_deriv": [-0.5],
    "baseline": [0.0],
    "amplitude": [1.0],
}

grid_param_ranges

mu_x = ls_params["mu_x"].to_numpy()
mu_y = ls_params["mu_y"].to_numpy()
sigma = ls_params["sigma"].to_numpy()

r_squared_threshold = 0.3
edge_tolerance = 1e-3  # in degrees

grid_min_xy, grid_max_xy = stimulus.grid.min(axis=(0, 1)), stimulus.grid.max(axis=(0, 1))

selection_criteria = pd.DataFrame({
    f"r_squared > {r_squared_threshold}": r_squared > r_squared_threshold,
    # The tolerance includes the grid values at the stimulus edges, which can be off due to rounding
    "center inside stimulus": (
        (mu_x >= grid_min_xy[1] - edge_tolerance)
        & (mu_x <= grid_max_xy[1] + edge_tolerance)
        & (mu_y >= grid_min_xy[0] - edge_tolerance)
        & (mu_y <= grid_max_xy[0] + edge_tolerance)
    ),
    # Least squares does not change sigma, so it is always one of the grid values
    "sigma below grid ceiling": ~np.isclose(sigma, np.max(grid_param_ranges["sigma"])),
})

is_selected = selection_criteria.all(axis=1).to_numpy()


def project_selected(voxel_values: np.ndarray) -> np.ndarray:
    """Project values of the selected voxels onto the surface, ignoring all other voxels."""
    return project_to_surface(fill_valid_voxels(np.where(is_selected, voxel_values, np.nan)))


# Fraction of valid voxels that fulfill each criterion and all criteria together
selection_criteria.assign(all_criteria=is_selected).mean()

size_surf = project_selected(sigma)
plot_surf_stat_map_helper(size_surf, vmin=0.0, vmax=5.0, title="pRF size (sigma), R² > 0.3")
plt.show()


# ------------------------------------------------------------------
# Exercise 24: pRF size for different R-squared thresholds
# ------------------------------------------------------------------
inside_stimulus = (
    (mu_x >= grid_min_xy[1] - edge_tolerance)
    & (mu_x <= grid_max_xy[1] + edge_tolerance)
    & (mu_y >= grid_min_xy[0] - edge_tolerance)
    & (mu_y <= grid_max_xy[0] + edge_tolerance)
)
below_ceiling = ~np.isclose(sigma, np.max(grid_param_ranges["sigma"]))


def selection_for(threshold):
    """Voxels passing all three criteria for a given R-squared threshold."""
    return (r_squared > threshold) & inside_stimulus & below_ceiling


for threshold in [0.1, 0.3, 0.5, 0.7]:
    selected = selection_for(threshold)
    print(f"R² > {threshold}: {selected.sum():>6} voxels selected "
          f"({selected.mean() * 100:.1f}%), median sigma = {np.median(sigma[selected]):.2f}°")

    sigma_surf = project_to_surface(fill_valid_voxels(np.where(selected, sigma, np.nan)))
    plot_surf_stat_map_helper(sigma_surf, vmin=0.0, vmax=5.0, title=f"pRF size (sigma), R² > {threshold}")
    plt.show()

# ------------------------------------------------------------------
# Exercise 25: polar angle on the surface
# ------------------------------------------------------------------
from prfmodel.utils import calculate_polar_angle

# 1. Project mu_x and mu_y SEPARATELY (only selected voxels; others are NaN)
mu_x_surf = project_selected(mu_x)
mu_y_surf = project_selected(mu_y)

# 2. Compute the polar angle AFTER projection, from the averaged x and y
polar_angle_surf = np.asarray(calculate_polar_angle(mu_x_surf, mu_y_surf))

# 3. Plot with a cyclic colour map (-pi and +pi get the same colour)
plot_surf_stat_map_helper(polar_angle_surf, vmin=-np.pi, vmax=np.pi, cyclic=True,
                          title="Polar angle of pRF centre (radians)")
plt.show()

# ------------------------------------------------------------------
# Exercise 26: eccentricity on the surface
# ------------------------------------------------------------------
from prfmodel.utils import calculate_eccentricity

# 1. Eccentricity per voxel = distance of the pRF centre from the centre of the screen
eccentricity = np.asarray(calculate_eccentricity(mu_x, mu_y))
# (equivalent: eccentricity = np.sqrt(mu_x**2 + mu_y**2))

# 2. Project only the selected voxels onto the surface (others are NaN)
ecc_surf = project_selected(eccentricity)

# 3. Plot
plot_surf_stat_map_helper(ecc_surf, vmin=0.0, vmax=4.0, title="Eccentricity of pRF centre (degrees)")
plt.show()

print(f"Eccentricity of selected voxels: min {eccentricity[is_selected].min():.2f}°, "
      f"median {np.median(eccentricity[is_selected]):.2f}°, max {eccentricity[is_selected].max():.2f}°")