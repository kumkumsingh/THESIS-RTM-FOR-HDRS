"""Practical 2 (cross-validation): load the training run and inspect the stimulus.

Run with:  python3 practical2.py
"""
import os

os.environ["KERAS_BACKEND"] = "tensorflow"   # must be set before importing prfmodel / keras

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prfmodel.examples import load_dataset
from prfmodel.plotting import animate_2d_prf_stimulus

pd.set_option("display.precision", 3)

# ------------------------------------------------------------------
# Load the training data (first run, first session, standard bar)
# ------------------------------------------------------------------
dataset_train = load_dataset("3t-chn-visual", split="ses-01_run-01", task="fixedbar")
stimulus_train = dataset_train.stimulus

# ------------------------------------------------------------------
# Exercise: inspect the design matrix and grid
# ------------------------------------------------------------------
print(stimulus_train)

design = np.asarray(stimulus_train.design)   # (time frames, y, x)
grid = np.asarray(stimulus_train.grid)       # (y, x, 2): grid[..., 0] = y, grid[..., 1] = x (degrees)

n_frames, n_y, n_x = design.shape
print(f"\nDesign matrix: {n_frames} time frames of {n_y} x {n_x} cells")
print(f"Frame duration (TR): {stimulus_train.frame_duration} s  ->  run length {n_frames * stimulus_train.frame_duration:.0f} s")
print(f"Grid: {grid.shape[0]} x {grid.shape[1]} cells, "
      f"x from {grid[..., 1].min():.2f} to {grid[..., 1].max():.2f} deg, "
      f"y from {grid[..., 0].min():.2f} to {grid[..., 0].max():.2f} deg")

# ------------------------------------------------------------------
# Track the bar: centre position and orientation in every frame
# ------------------------------------------------------------------
x_of_cell = grid[..., 1]
y_of_cell = grid[..., 0]

bar_on = design.reshape(n_frames, -1).max(axis=1) > 0
centre_x = np.full(n_frames, np.nan)
centre_y = np.full(n_frames, np.nan)
orientation = np.full(n_frames, "", dtype=object)

for t in np.flatnonzero(bar_on):
    cells = design[t] > 0
    xs, ys = x_of_cell[cells], y_of_cell[cells]
    centre_x[t], centre_y[t] = xs.mean(), ys.mean()
    # A horizontal bar is wide in x and narrow in y (and moves up/down); a vertical bar the opposite
    orientation[t] = "horizontal" if np.ptp(xs) > np.ptp(ys) else "vertical"

# ------------------------------------------------------------------
# Split into passes: a new pass starts after a blank period or when the orientation changes
# ------------------------------------------------------------------
passes = []
start = None
for t in range(n_frames):
    new_segment = bar_on[t] and (start is None or not bar_on[t - 1] or orientation[t] != orientation[t - 1])
    if new_segment:
        if start is not None:
            passes.append((start, t - 1))
        start = t
    elif not bar_on[t] and start is not None:
        passes.append((start, t - 1))
        start = None
if start is not None:
    passes.append((start, n_frames - 1))

rows = []
for i, (a, b) in enumerate(passes, start=1):
    ori = orientation[a]
    if ori == "horizontal":      # horizontal bar moves vertically
        delta = centre_y[b] - centre_y[a]
        direction = "bottom -> top" if delta > 0 else "top -> bottom"
    else:                        # vertical bar moves horizontally
        delta = centre_x[b] - centre_x[a]
        direction = "left -> right" if delta > 0 else "right -> left"
    rows.append({"pass": i, "frames": f"{a}-{b}", "n_frames": b - a + 1,
                 "bar": ori, "direction": direction})

passes_table = pd.DataFrame(rows).set_index("pass")
print(f"\nNumber of bar passes: {len(passes_table)}")
print(passes_table.to_string())
print(f"\nFrames without bar (blank): {(~bar_on).sum()}")

# ------------------------------------------------------------------
# Plot the bar position over time (each pass is one sweep)
# ------------------------------------------------------------------
frames = np.arange(n_frames)
fig, ax = plt.subplots(figsize=(12, 3.8), layout="constrained")
ax.plot(frames, centre_x, ".", label="bar centre x (deg)")
ax.plot(frames, centre_y, ".", label="bar centre y (deg)")
for a, b in passes:
    ax.axvspan(a, b, color="gray", alpha=0.1)
ax.set(xlabel="Time frame", ylabel="Position (deg)", title="Bar centre over time (grey = one pass)")
ax.legend(loc="upper right")
plt.show()

# ------------------------------------------------------------------
# A few frames from the middle of each pass
# ------------------------------------------------------------------
extent = (x_of_cell.min(), x_of_cell.max(), y_of_cell.min(), y_of_cell.max())
n_show = len(passes)
fig, axes = plt.subplots(1, n_show, figsize=(2.2 * n_show, 2.6), layout="constrained", squeeze=False)
for ax, (a, b), row in zip(axes[0], passes, rows):
    t = (a + b) // 2
    ax.imshow(design[t], extent=extent, origin="lower", cmap="Greys", vmin=0, vmax=1)
    ax.set_title(f"pass {row['pass']}\n{row['direction']}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

# ------------------------------------------------------------------
# Video of the stimulus (saved as a GIF, since HTML video needs a notebook)
# ------------------------------------------------------------------
ani = animate_2d_prf_stimulus(stimulus_train, interval=25)
ani.save("stimulus_train.gif", writer="pillow", fps=10)
print("\nSaved stimulus_train.gif: open it to watch the bar move")

# Look at how the frames looks like 
# Frames over time (real data)
show = [0, 2, 20, 39, 304]
fig, axes = plt.subplots(1, len(show), figsize=(12, 2.8), layout="constrained")
for ax, t in zip(axes, show):
    ax.imshow(design[t], cmap="Greys", vmin=0, vmax=1, origin="lower")
    ax.set_title(f"scan {t}")
    ax.set_xticks([]); ax.set_yticks([])
plt.show()

# The numbers inside one frame: a small 8 x 12 piece around the bar
t = 20
cols = np.flatnonzero(design[t].max(axis=0) > 0)                       # columns the bar covers
c0 = max(cols[0] - 4, 0) if len(cols) else 0
print(f"scan {t}, rows 50-57, columns {c0}-{c0 + 11}:")
print(np.round(design[t, 50:58, c0:c0 + 12], 2))

# Answer exercise 1:
#“The design matrix contains 305 frames (TR = 1.2 s) on a 108 × 108 grid covering −8° to 8° in x and y (about 0.15° per cell). After two blank frames, the bar makes 8 passes of about 37–38 frames (about 45 s) each, followed by three blank frames. The passes run in different directions: horizontal (left → right, right → left), vertical (top → bottom, bottom → top) and diagonal (e.g. top-left → bottom-right, top-right → bottom-left, bottom-right → top-left). Sweeping the bar through many directions ensures that every location in the visual field is crossed several times from different angles, which constrains both pRF position and size.”

# ------------------------------------------------------------------
# Exercise 2: inspect the BOLD time courses
# ------------------------------------------------------------------
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.models.prf import Gaussian2DPRFModel
from prfmodel.plotting import plot_response_heatmap

response_train = np.asarray(dataset_train.response)
print("\nResponse shape (vertices, time frames):", response_train.shape)

# --- What is the unit? Look at the typical level and spread of the values ---
vertex_mean = response_train.mean(axis=1)
vertex_sd = response_train.std(axis=1)
print(f"Mean over all values:          {response_train.mean():.3f}")
print(f"Median of per-vertex means:    {np.median(vertex_mean):.3f}")
print(f"Median of per-vertex SDs:      {np.median(vertex_sd):.3f}")
print(f"Min / max value:               {response_train.min():.2f} / {response_train.max():.2f}")
print(f"Vertices with constant signal: {(vertex_sd == 0).sum()}")

# --- Heatmap of many time courses (random subset, so it draws quickly) ---
rng = np.random.default_rng(0)
subset = rng.choice(len(response_train), size=min(20000, len(response_train)), replace=False)
plot_response_heatmap(
    response_train[np.sort(subset)],
    xlabel="Time frame (in TR)", ylabel="Vertex (random subset)", colorbar_label="BOLD response",
    figsize=(6, 6),
)
plt.show()

# --- Find vertices that follow the bar: correlate with simple pRF predictions ---
prf_model_train = Gaussian2DPRFModel(impulse_model=DerivativeTwoGammaImpulse(resolution=stimulus_train.frame_duration))

positions = [(x, y) for x in np.linspace(-6, 6, 7) for y in np.linspace(-6, 6, 7)]
candidates = pd.DataFrame({
    "mu_x": [p[0] for p in positions], "mu_y": [p[1] for p in positions], "sigma": 1.5,
    "weight_deriv": -0.5, "baseline": 0.0, "amplitude": 1.0,
})
predictions = np.asarray(prf_model_train(stimulus_train, candidates))   # (49, time frames)


def zscore_rows(a):
    a = a - a.mean(axis=1, keepdims=True)
    sd = a.std(axis=1, keepdims=True)
    sd[sd == 0] = 1
    return a / sd


z_pred = zscore_rows(predictions)
best_corr = np.empty(len(response_train))
for i in range(0, len(response_train), 20000):                          # in chunks to save memory
    z_data = zscore_rows(response_train[i:i + 20000])
    best_corr[i:i + 20000] = (z_data @ z_pred.T / z_pred.shape[1]).max(axis=1)

print(f"Vertices with best correlation > 0.5: {(best_corr > 0.5).sum()} ({(best_corr > 0.5).mean() * 100:.1f}%)")

# --- Plot a few well-responding vertices and one random vertex, with the bar passes marked ---
pass_starts = [2, 40, 78, 115, 152, 190, 227, 265]                      # from the bar-centre plot
good = np.argsort(best_corr)[-3:][::-1]
random_vertex = rng.integers(len(response_train))
frames = np.arange(response_train.shape[1])

fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, layout="constrained")
for ax, v in zip(axes, [*good, random_vertex]):
    ax.plot(frames, response_train[v], color="tab:blue", linewidth=1.2)
    for s in pass_starts:
        ax.axvline(s, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"Vertex {v}  (best correlation with a pRF prediction = {best_corr[v]:.2f})", loc="left", fontsize=10)
    ax.set_ylabel("BOLD")
axes[-1].set_xlabel("Time frame (TR = 1.2 s); dashed lines = start of each bar pass")
plt.show()

# The BOLD time courses are centred around zero, with fluctuations of several hundred units, so they are expressed in arbitrary (mean-centred) scanner units rather than percent signal change. Many vertices mainly show noise. However, vertices in visual cortex show clear responses to the bar sweeps: one response per pass, eight in total, appearing a few seconds after the bar crosses the vertex’s pRF and followed by a small undershoot. The timing of these responses differs between passes and between vertices, because the bar moves in different directions and the vertices have different pRF locations. Neighbouring vertices show almost identical time courses, since they sample the same underlying voxels.

# ------------------------------------------------------------------
# Exercise 3: standard deviation of each vertex on the inflated surface
# ------------------------------------------------------------------
from plotting_helpers import make_surface_plot

# The surface is the same for all splits of the dataset, so we take it from the training data
plot_surf_stat_map_helper = make_surface_plot(dataset_train.mesh)

response_sd = response_train.std(axis=1)                   # one SD per vertex
response_sd[response_sd == 0] = np.nan                     # vertices without signal -> shown as blank

print(f"SD per vertex: median {np.nanmedian(response_sd):.1f}, "
      f"95th percentile {np.nanpercentile(response_sd, 95):.1f}, max {np.nanmax(response_sd):.1f}")

# The data is in arbitrary units (hundreds), so set the colour limit from the data itself
vmax = np.nanpercentile(response_sd, 99)
plot_surf_stat_map_helper(response_sd, vmax=vmax, title="BOLD response standard deviation")
plt.show()

# Same map with a lower colour limit, so moderate variability (e.g. in V1) stands out
plot_surf_stat_map_helper(response_sd, vmax=np.nanpercentile(response_sd, 90),
                          title="BOLD response SD (lower colour limit)")
plt.show()

plot_surf_stat_map_helper(best_corr, vmax=0.8, title="Best correlation with a pRF prediction")
plt.show()
# “Variability is highest in the occipital lobe, particularly on the medial surface along the calcarine sulcus, which corresponds to the primary visual cortex (V1), and at the occipital pole. Elevated SD also appears in anterior medial regions near the medial wall, which likely reflects artefacts rather than visual responses. Overall, the SD map is noisy, so V1 is easier to identify on a map of stimulus-related signal, such as the correlation with pRF predictions or the model’s R².”

# ------------------------------------------------------------------
# Selecting vertices in visual cortex: ROI labels
# ------------------------------------------------------------------
roi_index = np.asarray(dataset_train.roi_index)      # one ROI label per vertex (0 = none, 1 = V1, 2 = V2, 3 = V3)
print("\nROI mapping:", dataset_train.roi_mapping)
for label in np.unique(roi_index):
    print(f"  label {label}: {(roi_index == label).sum()} vertices")

plot_surf_stat_map_helper(
    np.where(roi_index > 0, roi_index, np.nan),
    vmin=1, vmax=3, cmap="viridis",
    title="ROIs (1 = V1, 2 = V2, 3 = V3)",
)
plt.show()

# ------------------------------------------------------------------
# Exercise 4: do the most variable vertices lie inside the ROIs?
# ------------------------------------------------------------------
sd_all = response_train.std(axis=1)                  # raw SD per vertex (0 for constant vertices)
in_roi = roi_index > 0
varying = sd_all > 0

print(f"\nMedian SD inside ROIs:  {np.median(sd_all[in_roi & varying]):.1f}")
print(f"Median SD outside ROIs: {np.median(sd_all[~in_roi & varying]):.1f}")

for top in [1, 5, 10]:
    threshold = np.percentile(sd_all[varying], 100 - top)
    most_variable = varying & (sd_all >= threshold)
    print(f"Top {top:>2}% most variable vertices: {in_roi[most_variable].mean() * 100:.1f}% lie inside V1-V3 "
          f"(V1-V3 cover {in_roi.mean() * 100:.1f}% of the hemisphere)")

# SD map showing only vertices inside the ROIs, for a direct comparison with the full SD map
plot_surf_stat_map_helper(np.where(in_roi & varying, sd_all, np.nan), vmax=np.nanpercentile(sd_all[varying], 99),
                          title="BOLD response SD inside V1-V3 only")
plt.show()

# ------------------------------------------------------------------
# Practical: select valid vertices (in V1-V3 and not constant)
# ------------------------------------------------------------------
response_is_valid = in_roi & varying
response_train_valid = response_train[response_is_valid]


def fill_valid_vertices(values: np.ndarray) -> np.ndarray:
    """Fill values for the valid vertices into an array with one value per vertex, using NaN for all other vertices."""
    values_full = np.full((response_is_valid.shape[0],), fill_value=np.nan)
    values_full[response_is_valid] = values
    return values_full


print("Valid vertices (V1-V3, varying):", response_train_valid.shape)

# ------------------------------------------------------------------
# Exercise 5: vertices with a constant response
# ------------------------------------------------------------------
is_constant = sd_all == 0.0
print(f"\nConstant vertices in the whole hemisphere: {is_constant.sum()} "
      f"({is_constant.mean() * 100:.1f}% of {len(is_constant)})")
print(f"Constant vertices inside V1-V3:             {(is_constant & in_roi).sum()}")
for label in [1, 2, 3]:
    print(f"  in ROI {label}: {(is_constant & (roi_index == label)).sum()}")

# Where are they? Show constant vertices as 1 and everything else as blank
plot_surf_stat_map_helper(np.where(is_constant, 1.0, np.nan), vmin=0, vmax=1, cmap="autumn",
                          title="Vertices with a constant response")
plt.show()

# Many of the most variable vertices lie inside V1–V3: 33.4% of the top-1% most variable vertices are in these ROIs, although they cover only 7.8% of the hemisphere (about 4 times more than expected by chance), and the median SD is higher inside the ROIs (76.8) than outside (59.2). However, about two-thirds of the most variable vertices lie outside V1–V3. These likely include artefacts and noise, but also higher visual areas that respond to the bar and are not part of the ROIs. Variability is therefore enriched in, but not specific to, early visual cortex.”

# Fitting the model
# ------------------------------------------------------------------
# pRF model for this dataset (TR = 1.2 s)
# ------------------------------------------------------------------
from pathlib import Path

from prfmodel.fitters import GridFitter, grid_values_2d_prf
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.models.prf import Gaussian2DPRFModel
from prfmodel.plotting import plot_2d_prf_centers, plot_grid_parameter_distribution

tr = 1.2
impulse_model = DerivativeTwoGammaImpulse(resolution=tr)
prf_model = Gaussian2DPRFModel(impulse_model=impulse_model)

grid_param_ranges = grid_values_2d_prf(
    stimulus_train,
    num_mu=21,
    mu_extent=2.0,            # span twice the visual field covered by the stimulus
    num_sigma=10,
    sigma_range=(0.05, 4.0),  # log-spaced by default
) | {"weight_deriv": [-0.5], "baseline": [0.0], "amplitude": [1.0]}

print({k: np.round(v, 2) for k, v in grid_param_ranges.items()})

# ------------------------------------------------------------------
# Grid search (saved to disk, so later runs load it instead of refitting)
# ------------------------------------------------------------------
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)
grid_file = RESULTS / "practical2_grid_params.pkl"

if grid_file.exists():
    grid_params = pd.read_pickle(grid_file)
    print("Loaded grid parameters from", grid_file)
else:
    grid_fitter = GridFitter(model=prf_model, stimulus=stimulus_train, compile_step=True)
    grid_history, grid_params = grid_fitter.fit(
        data=response_train_valid,           # only the selected vertices (V1-V3)
        parameter_values=grid_param_ranges,
        batch_size=20,
    )
    grid_params.to_pickle(grid_file)
    print("Saved grid parameters to", grid_file)

print(grid_params.head())

# ------------------------------------------------------------------
# Exercise 7: centres on/outside the stimulus grid, sigma on the floor/ceiling
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
plot_2d_prf_centers(grid_params, grid_param_ranges, stimulus=stimulus_train, ax=axes[0])
axes[0].set(xlabel="mu_x (in degrees)", ylabel="mu_y (in degrees)", title="pRF centers")
plot_grid_parameter_distribution(grid_params, grid_param_ranges, "sigma", ax=axes[1])
axes[1].set(xlabel="sigma (in degrees)", title="pRF sizes")
plt.show()

# Count them
mu_x = grid_params["mu_x"].to_numpy()
mu_y = grid_params["mu_y"].to_numpy()
sigma = grid_params["sigma"].to_numpy()

x_min, x_max = grid[..., 1].min(), grid[..., 1].max()       # stimulus edges (about -8 and 8)
y_min, y_max = grid[..., 0].min(), grid[..., 0].max()
tol = 1e-3                                                   # tolerance for rounding

outside = (mu_x < x_min - tol) | (mu_x > x_max + tol) | (mu_y < y_min - tol) | (mu_y > y_max + tol)
on_edge = ~outside & (
    np.isclose(mu_x, x_min, atol=tol) | np.isclose(mu_x, x_max, atol=tol)
    | np.isclose(mu_y, y_min, atol=tol) | np.isclose(mu_y, y_max, atol=tol)
)
sigma_floor = np.isclose(sigma, np.min(grid_param_ranges["sigma"]))
sigma_ceiling = np.isclose(sigma, np.max(grid_param_ranges["sigma"]))

n = len(grid_params)
print(f"\nVertices fitted: {n}")
print(f"Centre exactly on the stimulus edge: {on_edge.sum():>6} ({on_edge.mean() * 100:.1f}%)")
print(f"Centre outside the stimulus:         {outside.sum():>6} ({outside.mean() * 100:.1f}%)")
print(f"Sigma at the floor (0.05 deg):       {sigma_floor.sum():>6} ({sigma_floor.mean() * 100:.1f}%)")
print(f"Sigma at the ceiling (4 deg):        {sigma_ceiling.sum():>6} ({sigma_ceiling.mean() * 100:.1f}%)")
print(f"Any of these problems:               {(on_edge | outside | sigma_floor | sigma_ceiling).sum():>6}")

# ------------------------------------------------------------------
# Least squares: fit baseline and amplitude for the grid pRFs
# ------------------------------------------------------------------
from prfmodel.fitters import LeastSquaresFitter

ls_file = RESULTS / "practical2_ls_params.pkl"

if ls_file.exists():
    ls_params = pd.read_pickle(ls_file)
    print("Loaded least-squares parameters from", ls_file)
else:
    ls_fitter = LeastSquaresFitter(model=prf_model, stimulus=stimulus_train)
    ls_history, ls_params = ls_fitter.fit(
        data=response_train_valid,
        parameters=grid_params,
        slope_name="amplitude",       # slope of the regression line   -> amplitude
        intercept_name="baseline",    # intercept of the regression line -> baseline
        batch_size=200,               # 200 vertices at a time
    )
    ls_params.to_pickle(ls_file)
    print("Saved least-squares parameters to", ls_file)

print(ls_params.head())
print(ls_params[["baseline", "amplitude"]].describe().round(2))
print(f"Vertices with a negative amplitude: {(ls_params['amplitude'] < 0).sum()}")

# Evaluating the model fit in-sample
# ------------------------------------------------------------------
# Evaluating the model fit in-sample (training data)
# ------------------------------------------------------------------
from prfmodel.plotting import plot_r_squared_hist
from prfmodel.utils import batched, calculate_r_squared

predict_batched = batched(prf_model)

# Predictions for the training stimulus with the training parameters
pred_response_train = np.asarray(predict_batched(stimulus_train, ls_params, batch_size=200))

r_squared_train = np.asarray(calculate_r_squared(response_train_valid, pred_response_train))  # one per vertex
print("\nR-squared shape:", r_squared_train.shape)

# ------------------------------------------------------------------
# Exercise 8
# ------------------------------------------------------------------
# 1. Distribution of R-squared
plot_r_squared_hist(r_squared_train, log=True)
plt.title("Training R-squared (in-sample)")
plt.show()

# 2. Fraction above 0.3, overall and per visual area
print(f"Median R²: {np.median(r_squared_train):.3f}   max: {r_squared_train.max():.3f}")
print(f"R² > 0.3: {(r_squared_train > 0.3).sum()} of {len(r_squared_train)} vertices "
      f"({(r_squared_train > 0.3).mean() * 100:.1f}%)")

roi_valid = roi_index[response_is_valid]           # ROI label of each selected vertex
for label, name in [(1, "V1"), (2, "V2"), (3, "V3")]:
    in_area = roi_valid == label
    print(f"  {name}: median R² {np.median(r_squared_train[in_area]):.3f}, "
          f"R² > 0.3 in {(r_squared_train[in_area] > 0.3).mean() * 100:.1f}% of its vertices")

# 3. R-squared on the inflated surface (data is already per vertex: no projection needed)
plot_surf_stat_map_helper(fill_valid_vertices(r_squared_train), vmin=0.0, vmax=0.8,
                          title="Training R-squared (V1-V3)")
plt.show()

# Same stimulus, new session
# ------------------------------------------------------------------
# Out-of-sample test 1: same stimulus, new session
# ------------------------------------------------------------------
dataset_test_ses = load_dataset("3t-chn-visual", split="ses-02_run-01", task="fixedbar")
stimulus_test_ses = dataset_test_ses.stimulus
response_test_ses = np.asarray(dataset_test_ses.response)
response_test_ses_valid = response_test_ses[response_is_valid]       # same vertices as for training

# ------------------------------------------------------------------
# Exercise 9: is the test stimulus identical to the training stimulus?
# ------------------------------------------------------------------
same_design = np.array_equal(np.asarray(stimulus_train.design), np.asarray(stimulus_test_ses.design))
same_grid = np.array_equal(np.asarray(stimulus_train.grid), np.asarray(stimulus_test_ses.grid))
same_tr = stimulus_train.frame_duration == stimulus_test_ses.frame_duration
print(f"\nDesign matrices identical: {same_design}")
print(f"Grids identical:           {same_grid}")
print(f"Same TR:                   {same_tr} ({stimulus_test_ses.frame_duration} s)")
print("Test response shape:", response_test_ses_valid.shape)

# Predictions for the test run, using the TRAINING parameters (ls_params, unchanged)
pred_response_test_ses = np.asarray(predict_batched(stimulus_test_ses, ls_params, batch_size=200))
print("Predictions identical to the training predictions:",
      np.allclose(pred_response_test_ses, pred_response_train))

# Out-of-sample R-squared vs in-sample R-squared
r_squared_test_ses = np.asarray(calculate_r_squared(response_test_ses_valid, pred_response_test_ses))
print(f"Median R²  training (in-sample): {np.median(r_squared_train):.3f}   "
      f"test, new session (out-of-sample): {np.median(r_squared_test_ses):.3f}")
print(f"R² > 0.3   training: {(r_squared_train > 0.3).mean() * 100:.1f}%   "
      f"test: {(r_squared_test_ses > 0.3).mean() * 100:.1f}%")
# ------------------------------------------------------------------
# Exercise 10: BOLD responses of the test run (new session) vs training run
# ------------------------------------------------------------------
# 1. Summary numbers (units and spread)
for name, data in [("training", response_train_valid), ("test (new session)", response_test_ses_valid)]:
    print(f"{name:<20} mean {data.mean():8.2f}   median SD {np.median(data.std(axis=1)):7.2f}   "
          f"min {data.min():9.1f}   max {data.max():9.1f}")

# 2. Heatmaps side by side (same V1-V3 vertices, same order, same colour scale)
lim = np.percentile(np.abs(response_train_valid), 99)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained", sharey=True)
for ax, data, title in [(axes[0], response_train_valid, "Training run (session 1)"),
                        (axes[1], response_test_ses_valid, "Test run (session 2)")]:
    im = ax.imshow(data, aspect="auto", cmap="inferno", vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set(title=title, xlabel="Time frame (TR = 1.2 s)")
axes[0].set_ylabel("Selected vertex (V1-V3)")
fig.colorbar(im, ax=axes, label="BOLD response")
plt.show()

# 3. The same well-fitting vertices in both runs, overlaid
best = np.argsort(r_squared_train)[-3:][::-1]
fig, axes = plt.subplots(len(best), 1, figsize=(12, 7), sharex=True, layout="constrained")
for ax, v in zip(axes, best):
    ax.plot(response_train_valid[v], color="tab:blue", linewidth=1.2, label="training (session 1)")
    ax.plot(response_test_ses_valid[v], color="tab:orange", linewidth=1.2, label="test (session 2)")
    ax.set_title(f"Selected vertex {v}: training R² = {r_squared_train[v]:.2f}", loc="left", fontsize=10)
axes[0].legend(loc="upper right")
axes[-1].set_xlabel("Time frame")
plt.show()

# 4. How similar are the two runs for each vertex? (same stimulus, so this measures reproducibility)
def row_corr(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return (a * b).sum(axis=1) / np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))

run_corr = row_corr(response_train_valid, response_test_ses_valid)
print(f"Correlation between training and test time courses: median {np.median(run_corr):.2f}, "
      f"> 0.5 in {(run_corr > 0.5).mean() * 100:.1f}% of vertices")

plot_surf_stat_map_helper(fill_valid_vertices(run_corr), vmin=-0.2, vmax=0.8,
                          title="Correlation between session 1 and session 2 (same stimulus)")
plt.show()
# ------------------------------------------------------------------
# Exercise 11: training vs test R-squared
# ------------------------------------------------------------------
from prfmodel.plotting import plot_observed_predicted, plot_r_squared_comparison

r2_train = np.asarray(r_squared_train)
r2_test = np.asarray(r_squared_test_ses)

print(f"\nMedian R²   training: {np.median(r2_train):.3f}   test (new session): {np.median(r2_test):.3f}")
print(f"R² > 0.3    training: {(r2_train > 0.3).mean() * 100:.1f}%   test: {(r2_test > 0.3).mean() * 100:.1f}%")
print(f"Test R² lower than training R² for {(r2_test < r2_train).mean() * 100:.1f}% of vertices")
print(f"Negative test R²: {(r2_test < 0).sum()} vertices ({(r2_test < 0).mean() * 100:.1f}%), "
      f"minimum {r2_test.min():.2f}")

# 1. Scatter: training R² vs test R² (points below the red line: worse on test data)
try:
    plot_r_squared_comparison(r2_train, r2_test)           # practical's helper (check help() if this fails)
except TypeError:
    fig, ax = plt.subplots(figsize=(5.5, 5.5), layout="constrained")
    ax.scatter(r2_train, r2_test, s=2, alpha=0.3)
    ax.plot([-1, 1], [-1, 1], color="red", linewidth=1)
    ax.set(xlabel="R² training (in-sample)", ylabel="R² test, new session (out-of-sample)",
           xlim=(-0.5, 1), ylim=(-0.5, 1))
plt.show()

# 2. Observed vs predicted for the best-fitting vertices (training and test data with the same prediction)
best = np.argsort(r2_train)[-3:][::-1]
fig, axes = plt.subplots(len(best), 1, figsize=(12, 8), sharex=True, layout="constrained")
for ax, v in zip(axes, best):
    ax.plot(response_train_valid[v], color="tab:blue", alpha=0.6, linewidth=1, label="observed, training")
    ax.plot(response_test_ses_valid[v], color="tab:orange", alpha=0.6, linewidth=1, label="observed, test")
    ax.plot(pred_response_train[v], color="black", linewidth=2, label="prediction (same for both)")
    ax.set_title(f"Vertex {v}: R² training {r2_train[v]:.2f}, test {r2_test[v]:.2f}", loc="left", fontsize=10)
axes[0].legend(loc="upper right")
axes[-1].set_xlabel("Time frame")
plt.show()

# (Optional) the practical's helper for one vertex; check help(plot_observed_predicted) for its arguments
# plot_observed_predicted(response_test_ses_valid[best[0]], pred_response_test_ses[best[0]])
# plt.show()

# 3. What do the vertices with negative test R² have in common?
negative = r2_test < 0
boundary = outside | on_edge | sigma_floor | sigma_ceiling            # from Exercise 7
amp = ls_params["amplitude"].to_numpy()
print(f"\nOf the {negative.sum()} vertices with negative test R²:")
print(f"  boundary estimates (centre outside/on edge, sigma at floor/ceiling): {(negative & boundary).mean() / negative.mean() * 100:.1f}%")
print(f"  training R² below 0.1:                                             {(negative & (r2_train < 0.1)).mean() / negative.mean() * 100:.1f}%")
print(f"  median amplitude: {np.median(amp[negative]):.1f}  (all vertices: {np.median(amp):.1f})")

# The worst vertex, to see what goes wrong
w = int(np.argmin(r2_test))
plt.figure(figsize=(12, 3.5))
plt.plot(response_test_ses_valid[w], color="tab:orange", label="observed, test")
plt.plot(pred_response_test_ses[w], color="black", linewidth=2, label="prediction")
plt.title(f"Worst vertex {w}: R² training {r2_train[w]:.2f}, test {r2_test[w]:.2f}")
plt.legend()
plt.show()

dataset_test_run = load_dataset("3t-chn-visual", split="ses-01_run-02", task="fixedbar")

stimulus_test_run = dataset_test_run.stimulus

response_test_run = dataset_test_run.response

# Select the same vertices as for the training data
response_test_run_valid = response_test_run[response_is_valid]

# ------------------------------------------------------------------
# Out-of-sample test 2: new stimulus, same session
# ------------------------------------------------------------------
dataset_test_run = load_dataset("3t-chn-visual", split="ses-01_run-02", task="fixedbar")
stimulus_test_run = dataset_test_run.stimulus
response_test_run = np.asarray(dataset_test_run.response)
response_test_run_valid = response_test_run[response_is_valid]       # same vertices as for training

# ------------------------------------------------------------------
# Exercise 13a: compare the two stimuli
# ------------------------------------------------------------------
design_train = np.asarray(stimulus_train.design)
design_run = np.asarray(stimulus_test_run.design)
print("\nTest stimulus:", stimulus_test_run)
print("Design matrices identical:", np.array_equal(design_train, design_run))

# Are the same bar positions used, just in a different order?
train_frames = {f.tobytes() for f in (design_train > 0)}
run_frames = {f.tobytes() for f in (design_run > 0)}
print(f"Distinct bar images: training {len(train_frames)}, test {len(run_frames)}, shared {len(train_frames & run_frames)}")


def bar_centre(design, grid):
    """Bar centre (x, y) in degrees for every frame (NaN when no bar is shown)."""
    on = design.reshape(len(design), -1).max(axis=1) > 0
    cx, cy = np.full(len(design), np.nan), np.full(len(design), np.nan)
    for t in np.flatnonzero(on):
        cells = design[t] > 0
        cx[t], cy[t] = grid[..., 1][cells].mean(), grid[..., 0][cells].mean()
    return cx, cy


fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, layout="constrained")
for ax, des, stim, title in [(axes[0], design_train, stimulus_train, "Training stimulus (sequence A)"),
                              (axes[1], design_run, stimulus_test_run, "Test stimulus (sequence B)")]:
    cx, cy = bar_centre(des, np.asarray(stim.grid))
    ax.plot(cx, ".", label="bar centre x (deg)")
    ax.plot(cy, ".", label="bar centre y (deg)")
    ax.set(title=title, ylabel="Position (deg)")
axes[0].legend(loc="upper right")
axes[-1].set_xlabel("Time frame")
plt.show()

# Video of the test stimulus (compare with stimulus_train.gif)
ani_run = animate_2d_prf_stimulus(stimulus_test_run, interval=25)
ani_run.save("stimulus_test_run.gif", writer="pillow", fps=10)
print("Saved stimulus_test_run.gif")

# ------------------------------------------------------------------
# Exercise 13b: out-of-sample prediction performance on the new stimulus
# ------------------------------------------------------------------
pred_response_test_run = np.asarray(predict_batched(stimulus_test_run, ls_params, batch_size=200))
r2_run = np.asarray(calculate_r_squared(response_test_run_valid, pred_response_test_run))

print("Predictions differ from the training predictions:",
      not np.allclose(pred_response_test_run, pred_response_train))

summary = pd.DataFrame({
    "median R²": [np.median(r2_train), np.median(r2_test), np.median(r2_run)],
    "% R² > 0.3": [(r2_train > 0.3).mean() * 100, (r2_test > 0.3).mean() * 100, (r2_run > 0.3).mean() * 100],
    "% R² < 0": [(r2_train < 0).mean() * 100, (r2_test < 0).mean() * 100, (r2_run < 0).mean() * 100],
}, index=["training (in-sample)", "test: same stimulus, new session", "test: new stimulus, same session"])
print(summary.round(3))

# Training vs new-stimulus R²
fig, ax = plt.subplots(figsize=(5.5, 5.5), layout="constrained")
ax.scatter(r2_train, r2_run, s=2, alpha=0.3)
ax.plot([-1, 1], [-1, 1], color="red", linewidth=1)
ax.set(xlabel="R² training", ylabel="R² test: new stimulus", xlim=(-0.5, 1), ylim=(-0.5, 1))
plt.show()

# Best vertices: the new prediction against the new data
best = np.argsort(r2_train)[-3:][::-1]
fig, axes = plt.subplots(len(best), 1, figsize=(12, 8), sharex=True, layout="constrained")
for ax, v in zip(axes, best):
    ax.plot(response_test_run_valid[v], color="tab:green", alpha=0.7, linewidth=1, label="observed, new stimulus")
    ax.plot(pred_response_test_run[v], color="black", linewidth=2, label="prediction for new stimulus")
    ax.set_title(f"Vertex {v}: R² training {r2_train[v]:.2f}, new stimulus {r2_run[v]:.2f}", loc="left", fontsize=10)
axes[0].legend(loc="upper right")
axes[-1].set_xlabel("Time frame")
plt.show()

# R² for the new stimulus on the surface
plot_surf_stat_map_helper(fill_valid_vertices(r2_run), vmin=0.0, vmax=0.8, title="Test R² (new stimulus)")
plt.show()

# “Training and new-stimulus R² values are strongly related and lie close to the identity line: vertices that fit the training data well also predict the responses to the new bar sequence well, even though the model was never fitted to this stimulus. This shows that the pRF model captures how each vertex responds to stimulation at different visual-field locations, rather than only fitting the training time course. Many vertices even reach a slightly higher R² for the new stimulus (maximum about 0.77), possibly because the new sequence drives them more effectively, noise differs between runs, and, unlike the session test, there is no alignment issue within one session. For the best-fitting vertices, test R² is slightly lower than training R², reflecting mild overfitting. Vertices with near-zero training R² remain near zero or slightly negative. A small group of vertices with moderate training R² (0.2–0.6) shows negative test R²; these likely have unreliable parameter estimates (pRF centres outside the stimulus, σ at the grid limits or extreme amplitudes) and do not generalise to the new stimulus.”

# ------------------------------------------------------------------
# Exercise 14: compare the two out-of-sample tests
# ------------------------------------------------------------------
# r2_train: training, r2_test: same stimulus / new session, r2_run: new stimulus / same session

comparison = pd.DataFrame({
    "median R²":   [np.median(r2_train), np.median(r2_test), np.median(r2_run)],
    "mean R²":     [r2_train.mean(), r2_test.mean(), r2_run.mean()],
    "% R² > 0.3":  [(r2_train > 0.3).mean() * 100, (r2_test > 0.3).mean() * 100, (r2_run > 0.3).mean() * 100],
    "% R² < 0":    [(r2_train < 0).mean() * 100, (r2_test < 0).mean() * 100, (r2_run < 0).mean() * 100],
}, index=["training (in-sample)",
          "test 1: same stimulus, new session",
          "test 2: new stimulus, same session"])
print("\n", comparison.round(3))

# Only the vertices that fit well in training (the ones where generalisation is meaningful)
good = r2_train > 0.3
print(f"\nVertices with training R² > 0.3 ({good.sum()}): median R² "
      f"test 1 = {np.median(r2_test[good]):.3f}, test 2 = {np.median(r2_run[good]):.3f}")

# Paired comparison: for how many vertices is each test better?
diff = r2_run - r2_test
print(f"Test 2 (new stimulus) better for {(diff > 0).mean() * 100:.1f}% of vertices, "
      f"test 1 (new session) better for {(diff < 0).mean() * 100:.1f}%  "
      f"(median difference {np.median(diff):+.3f})")

# Per visual area
for label, name in [(1, "V1"), (2, "V2"), (3, "V3")]:
    a = roi_valid == label
    print(f"  {name}: median R² test 1 = {np.median(r2_test[a]):.3f}, test 2 = {np.median(r2_run[a]):.3f}")

# Scatter: test 1 vs test 2 (points above the red line: better for the new stimulus)
fig, ax = plt.subplots(figsize=(5.5, 5.5), layout="constrained")
ax.scatter(r2_test, r2_run, s=2, alpha=0.3)
ax.plot([-1, 1], [-1, 1], color="red", linewidth=1)
ax.set(xlabel="R² test 1: same stimulus, new session",
       ylabel="R² test 2: new stimulus, same session", xlim=(-0.5, 1), ylim=(-0.5, 1))
plt.show()

# Distributions of all three
bins = np.linspace(-0.5, 1, 61)
plt.figure(figsize=(8, 4))
for r2, label in [(r2_train, "training"), (r2_test, "test 1: new session"), (r2_run, "test 2: new stimulus")]:
    plt.hist(r2, bins=bins, histtype="step", linewidth=2, label=label)
plt.yscale("log")
plt.xlabel("R²")
plt.ylabel("Number of vertices (log)")
plt.legend()
plt.show()
"The out-of-sample performance is slightly better for test 1 (same stimulus, new session): it yields a higher R² for 58.4% of the vertices, and its median R² is higher in V1 (0.076 vs 0.035), V2 (0.169 vs 0.143) and V3 (0.180 vs 0.155). The difference is small, however (median difference 0.006). Both test datasets contain independent noise, so noise affects both. Test 1 was recorded in a different session, so it could suffer from misalignment and session differences; the fact that it performs at least as well as test 2 indicates that the two sessions are well aligned. Since the stimulus is identical, test 1 only requires the model to reproduce the same predicted time course. Test 2 requires the model to predict responses to a new bar sequence, which works only to the extent that the fitted pRF parameters correctly describe each vertex’s spatial tuning. The slightly lower performance for the new stimulus therefore suggests that part of the training fit is specific to the training sequence, for example due to the coarse parameter grid, unreliable boundary estimates and simplifications of the model. Nevertheless, the small difference shows that the pRF model generalises well to a new stimulus. The low medians in V1 reflect the many V1 vertices that represent the periphery beyond the stimulated area."

# ------------------------------------------------------------------
# Fitting and interpreting the final model: average the two identical-stimulus runs
# ------------------------------------------------------------------
response_avg = np.stack([response_train_valid, response_test_ses_valid], axis=-1).mean(axis=-1)
print("\nAveraged response shape:", response_avg.shape)      # (17326, 305)

# Exercise 15: averaged vs. single-session responses

# ------------------------------------------------------------------
# Exercise 15: noise level of the averaged responses vs session 1 only
# ------------------------------------------------------------------
# Noise estimate: the two runs share the same signal, so their difference is (almost) pure noise.
noise_single = (response_train_valid - response_test_ses_valid).std(axis=1) / np.sqrt(2)   # noise SD of one run
noise_avg = noise_single / np.sqrt(2)                                                       # expected for the average

print(f"Median SD per vertex:   session 1 only {np.median(response_train_valid.std(axis=1)):.1f}, "
      f"averaged {np.median(response_avg.std(axis=1)):.1f}")
print(f"Median noise SD (est.): one run {np.median(noise_single):.1f}, average {np.median(noise_avg):.1f}")

# Smoothness: correlation between neighbouring time points (higher = less high-frequency noise)
def lag1(a):
    a = a - a.mean(axis=1, keepdims=True)
    return (a[:, 1:] * a[:, :-1]).sum(axis=1) / (a ** 2).sum(axis=1)

print(f"Median lag-1 autocorrelation: session 1 only {np.median(lag1(response_train_valid)):.2f}, "
      f"averaged {np.median(lag1(response_avg)):.2f}")

# Visual comparison for the best-fitting vertices
best = np.argsort(r2_train)[-3:][::-1]
fig, axes = plt.subplots(len(best), 1, figsize=(12, 7.5), sharex=True, layout="constrained")
for ax, v in zip(axes, best):
    ax.plot(response_train_valid[v], color="tab:blue", alpha=0.6, linewidth=1, label="session 1 only")
    ax.plot(response_avg[v], color="black", linewidth=1.8, label="average of both sessions")
    ax.set_title(f"Vertex {v}", loc="left", fontsize=10)
axes[0].legend(loc="upper right")
axes[-1].set_xlabel("Time frame")
plt.show()
# "Yes. The averaged responses are less noisy than the responses from session 1 alone. Because the two runs show the identical stimulus, their stimulus-driven signal is the same, while their noise is independent. Averaging keeps the signal but partly cancels the noise: for two runs with similar noise levels, the noise SD drops by about a factor √2 (about 30%). In the plots, the averaged time courses are smoother, with fewer random spikes between the response peaks, while the peaks to the bar sweeps remain. Accordingly, the SD per vertex [decreases from X to Y], and the lag-1 autocorrelation [increases], indicating less high-frequency noise."

# Exercise 16: why averaging across sessions is valid here

# "Averaging is valid because the two runs measure the same thing under the same conditions:

# The stimulus is identical (we checked with np.array_equal), so the expected response of each vertex is the same at every time point.
# The timing is identical: same number of frames (305) and the same TR (1.2 s), so time point t means the same moment in the bar sequence in both runs.
# The sessions are well aligned: each vertex refers to the same location on the cortical surface in both sessions. The good out-of-sample performance of test 1 (it even predicted slightly better than test 2) confirms this alignment.
# The responses are reproducible: the correlation between the two runs' time courses was high for responsive vertices.

# So the two runs are repeated measurements of the same signal with independent noise. Averaging them keeps the signal and reduces the noise, giving more precise parameter estimates."

# Exercise 17: why not average with the second run of session 1

# "The first runs of both sessions show the identical bar sequence (A), so at each time point the bar is in the same place and the vertices respond at the same moments. Averaging them combines two measurements of the same expected response.

# The second run of the first session shows a different bar sequence (B). At a given time point, the bar is somewhere else, so a vertex responds at different moments in run A and run B. Averaging them would mix responses to different stimuli: peaks from one run would be blurred, shifted or partly cancelled by the other, and the result would not correspond to any single stimulus. The pRF model needs one stimulus that describes the data, and no stimulus produces this averaged time course.

# Combining runs with different stimuli is still possible, but by fitting them jointly (each run with its own stimulus, e.g. by concatenating the runs and their design matrices), not by averaging time courses."

# Exercie 18
# ------------------------------------------------------------------
# Exercise 18: fit the model to the averaged responses (grid search + least squares)
# ------------------------------------------------------------------
avg_grid_file = RESULTS / "practical2_avg_grid_params.pkl"
avg_ls_file = RESULTS / "practical2_avg_ls_params.pkl"

if avg_ls_file.exists():
    ls_params_avg = pd.read_pickle(avg_ls_file)
    print("Loaded averaged-fit parameters from", avg_ls_file)
else:
    grid_fitter_avg = GridFitter(model=prf_model, stimulus=stimulus_train, compile_step=True)
    _, grid_params_avg = grid_fitter_avg.fit(data=response_avg, parameter_values=grid_param_ranges, batch_size=20)
    grid_params_avg.to_pickle(avg_grid_file)

    ls_fitter_avg = LeastSquaresFitter(model=prf_model, stimulus=stimulus_train)
    _, ls_params_avg = ls_fitter_avg.fit(
        data=response_avg, parameters=grid_params_avg,
        slope_name="amplitude", intercept_name="baseline", batch_size=200,
    )
    ls_params_avg.to_pickle(avg_ls_file)
    print("Saved averaged-fit parameters to", avg_ls_file)

# In-sample evaluation on the averaged responses (the stimulus is the same as in the training run)
pred_response_avg = np.asarray(predict_batched(stimulus_train, ls_params_avg, batch_size=200))
r2_avg = np.asarray(calculate_r_squared(response_avg, pred_response_avg))

print(f"Median in-sample R²:  run 1 only {np.median(r2_train):.3f}   averaged {np.median(r2_avg):.3f}")
print(f"R² > 0.3:             run 1 only {(r2_train > 0.3).mean() * 100:.1f}%   averaged {(r2_avg > 0.3).mean() * 100:.1f}%")
print(f"Higher R² with averaging for {(r2_avg > r2_train).mean() * 100:.1f}% of vertices")

fig, ax = plt.subplots(figsize=(5.5, 5.5), layout="constrained")
ax.scatter(r2_train, r2_avg, s=2, alpha=0.3)
ax.plot([0, 1], [0, 1], color="red", linewidth=1)
ax.set(xlabel="In-sample R² (run 1 of session 1)", ylabel="In-sample R² (averaged runs)", xlim=(0, 1), ylim=(0, 1))
plt.show()

# # "The in-sample R² of the model fitted to the averaged responses is higher than that of the model fitted to the first run only (median [X] vs [Y]; [P]% of vertices above 0.3 vs [Q]%), and most vertices lie above the identity line. Averaging the two runs reduces the noise while keeping the stimulus-driven signal, so a larger fraction of the variance in the data can be explained by the pRF model. Note that this R² is still in-sample and therefore optimistic; we accept this because cross-validation already showed how well the model generalises."

# # Exercise 19: threshold on in-sample R² and plot the parameters

# # ------------------------------------------------------------------
# # Exercise 19: parameters of vertices with in-sample R² above a threshold
# # ------------------------------------------------------------------
mu_x_avg = ls_params_avg["mu_x"].to_numpy()
mu_y_avg = ls_params_avg["mu_y"].to_numpy()
sigma_avg = ls_params_avg["sigma"].to_numpy()
ecc_avg = np.hypot(mu_x_avg, mu_y_avg)                    # eccentricity (deg)
angle_avg = np.arctan2(mu_y_avg, mu_x_avg)                # polar angle (radians)

tol = 1e-3
inside_avg = ((np.abs(mu_x_avg) <= grid[..., 1].max() + tol) & (np.abs(mu_y_avg) <= grid[..., 0].max() + tol))
below_ceiling_avg = ~np.isclose(sigma_avg, np.max(grid_param_ranges["sigma"]))


def plot_parameter_maps(selected, label):
    """Plot pRF size, eccentricity and polar angle for the selected vertices; all others are NaN."""
    print(f"{label}: {selected.sum()} vertices selected ({selected.mean() * 100:.1f}%)")
    for values, kwargs, name in [
        (sigma_avg, dict(vmin=0.0, vmax=4.0), "pRF size (sigma, deg)"),
        (ecc_avg, dict(vmin=0.0, vmax=8.0), "Eccentricity (deg)"),
        (angle_avg, dict(vmin=-np.pi, vmax=np.pi, cyclic=True), "Polar angle (rad)"),
    ]:
        plot_surf_stat_map_helper(fill_valid_vertices(np.where(selected, values, np.nan)), **kwargs,
                                  title=f"{name}: {label}")
        plt.show()


r2_threshold = 0.3
selected_insample = (r2_avg > r2_threshold) & inside_avg & below_ceiling_avg
plot_parameter_maps(selected_insample, f"in-sample R² > {r2_threshold}")

# # Compared with the first practical, the larger stimulus (±8°) covers more of the periphery, so the eccentricity map extends further anteriorly. Vertices representing locations beyond 8° are still excluded, because their centres fall outside the stimulus. The 3T data are noisier than the 7T data, so the maps are somewhat less smooth."

# # One caveat: the plotting helper averages vertices that fall in the same small cube. That's fine for σ and eccentricity, and for polar angle here too, because the left hemisphere's angles stay around 0 (the right visual field) rather than near the ±180° wrap-around.

# # Exercise 20: threshold on out-of-sample R² instead

# # ------------------------------------------------------------------
# # Exercise 20: select vertices by out-of-sample R² on ses-01_run-02
# # ------------------------------------------------------------------
pred_run_avgmodel = np.asarray(predict_batched(stimulus_test_run, ls_params_avg, batch_size=200))
r2_run_avgmodel = np.asarray(calculate_r_squared(response_test_run_valid, pred_run_avgmodel))

selected_outsample = (r2_run_avgmodel > r2_threshold) & inside_avg & below_ceiling_avg
plot_parameter_maps(selected_outsample, f"out-of-sample R² (run-02) > {r2_threshold}")

# How do the two selections differ?
both = selected_insample & selected_outsample
only_in = selected_insample & ~selected_outsample
only_out = selected_outsample & ~selected_insample
print(f"Selected by both: {both.sum()}, only by in-sample R²: {only_in.sum()}, only by out-of-sample R²: {only_out.sum()}")
print(f"Median sigma: in-sample selection {np.median(sigma_avg[selected_insample]):.2f}, "
      f"out-of-sample selection {np.median(sigma_avg[selected_outsample]):.2f}, "
      f"dropped vertices {np.median(sigma_avg[only_in]) if only_in.any() else np.nan:.2f}")
print(f"Median eccentricity of dropped vertices: {np.median(ecc_avg[only_in]) if only_in.any() else np.nan:.2f} deg")

# Where are the vertices that pass in-sample but fail out-of-sample?
plot_surf_stat_map_helper(fill_valid_vertices(np.where(only_in, 1.0, np.nan)), vmin=0, vmax=1, cmap="autumn",
                          title="Selected in-sample but not out-of-sample")
plt.show()

# # 