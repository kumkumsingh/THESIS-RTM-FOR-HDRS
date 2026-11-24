"""Practical 4: the canonical Gaussian CF model, from the interactive figures to the exercises.

Run from the folder that contains plotting_helpers.py. The V1 distance matrix is cached in results/.
Close each figure window to continue to the next one.
"""
import os
import pickle
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


def cached(name: str, compute):
    """Load a result from results/ if it exists, otherwise compute and save it."""
    path = RESULTS / f"practical4_{name}.pkl"
    if path.exists():
        print(f"Loading {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    result = compute()
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved {path}")
    return result


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
from prfmodel.examples import load_dataset

surface = load_dataset("hcp-999999-surface", surface_type="inflated")
dataset = load_dataset("7t-retbar-visual", hemisphere="both")

response_raw = np.asarray(dataset.response)
with np.errstate(invalid="ignore", divide="ignore"):
    response_psc = (response_raw / response_raw.mean(axis=1, keepdims=True) - 1.0) * 100.0


# ------------------------------------------------------------------
# Exercise 1: the stimulus design matrix
# ------------------------------------------------------------------
import scipy.io

design_raw = scipy.io.loadmat(dataset.files["design"])["stim"]
print("Raw design shape:", design_raw.shape, "dtype:", design_raw.dtype)
print("Response shape:", dataset.response.shape)

# Put the time axis first: it is the axis with as many entries as the response has time frames
num_frames = dataset.response.shape[1]
time_axis = [i for i, s in enumerate(design_raw.shape) if s == num_frames]
time_axis = time_axis[-1] if time_axis else int(np.argmax(design_raw.shape))
design = np.moveaxis(design_raw, time_axis, 0).astype("float32")
design = design / design.max()          # 1 = bar, 0 = background
print("Design (frames, y, x):", design.shape)

is_bar_visible = design.reshape(design.shape[0], -1).max(axis=1) > 0

# 1. Some single time frames
frames_to_show = np.linspace(0, design.shape[0] - 1, 8).astype(int)
fig, axes = plt.subplots(1, 8, figsize=(16, 2.6), layout="constrained")
for ax, t in zip(axes, frames_to_show):
    ax.imshow(design[t], cmap="Greys_r", vmin=0, vmax=1)
    ax.set_title(f"frame {t}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

# 2. Count the bar passes: follow the bar centre; a new pass starts after a blank or a jump
rows, cols = np.mgrid[0:design.shape[1], 0:design.shape[2]]
centre = np.full((design.shape[0], 2), np.nan)
for t in np.flatnonzero(is_bar_visible):
    w = design[t]
    centre[t] = [(w * cols).sum() / w.sum(), (w * rows).sum() / w.sum()]   # (x, y) in pixels

steps = np.linalg.norm(np.diff(centre, axis=0), axis=1)
new_pass = np.r_[True, ~(steps < 4 * np.nanmedian(steps))] & is_bar_visible

names = ["left to right", "bottom-left to top-right", "bottom to top", "bottom-right to top-left",
         "right to left", "top-right to bottom-left", "top to bottom", "top-left to bottom-right"]
passes = []
for s in np.flatnonzero(new_pass):
    e = s
    while e + 1 < design.shape[0] and is_bar_visible[e + 1] and not new_pass[e + 1]:
        e += 1
    if e - s < 3:
        continue                                   # ignore tiny fragments
    dx = centre[e, 0] - centre[s, 0]
    dy = -(centre[e, 1] - centre[s, 1])            # image rows go down; flip so up is positive
    angle = np.degrees(np.arctan2(dy, dx)) % 360
    passes.append({"frames": f"{s}-{e}", "num_frames": e - s + 1,
                   "direction": names[int(((angle + 22.5) % 360) // 45)]})

passes = pd.DataFrame(passes, index=pd.RangeIndex(1, len(passes) + 1, name="pass"))
print(f"\nNumber of bar passes: {len(passes)}")
print(passes.to_string())

# 3. When is the bar on screen?
fig, ax = plt.subplots(figsize=(10, 2), layout="constrained")
ax.step(np.arange(design.shape[0]), is_bar_visible, where="mid")
ax.set(xlabel="Time frame (TR = 1.5 s)", yticks=[0, 1], yticklabels=["Blank", "Bar"])
plt.show()

# ------------------------------------------------------------------
# Source region (V1) and distance matrix
# ------------------------------------------------------------------
label_v1 = 1
is_v1 = np.concatenate([surface.atlas["left"] == label_v1, surface.atlas["right"] == label_v1])
source_response = response_psc[is_v1]
print("Source response (V1 vertices, frames):", source_response.shape)
print("V1 vertices with NaN:", np.sum(~np.all(np.isfinite(source_response), axis=1)))

import nibabel as nib
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def calculate_distance_matrix(surface_path, source_indices: np.ndarray) -> np.ndarray:
    """Calculate the shortest distance along a surface mesh between every pair of source vertices."""
    coords, triangles = nib.load(surface_path).agg_data()
    i, j, k = triangles.T
    edges = np.concatenate([np.column_stack((i, j)), np.column_stack((j, k)), np.column_stack((k, i))])
    edge_lengths = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
    num_vertices = coords.shape[0]
    graph = csr_matrix((edge_lengths, (edges[:, 0], edges[:, 1])), shape=(num_vertices, num_vertices))
    # Only keep the edges between source vertices, so that the shortest paths stay inside the source region
    source_graph = graph[source_indices][:, source_indices]
    return dijkstra(source_graph, directed=False)


def make_distance_matrix(label: int) -> np.ndarray:
    """Distance matrix for one atlas region in both hemispheres (infinite between hemispheres)."""
    idx_left = np.where(surface.atlas["left"] == label)[0]
    idx_right = np.where(surface.atlas["right"] == label)[0]
    d_left = calculate_distance_matrix(surface.files["wm_lh"], idx_left)
    d_right = calculate_distance_matrix(surface.files["wm_rh"], idx_right)
    n_l, n_r = len(idx_left), len(idx_right)
    return np.block([
        [d_left, np.full((n_l, n_r), np.inf)],
        [np.full((n_r, n_l), np.inf), d_right],
    ])


distance_matrix = cached("distance_matrix_v1", lambda: make_distance_matrix(label_v1))
print("Distance matrix:", distance_matrix.shape)

from prfmodel.stimuli import CFStimulus

stimulus = CFStimulus(distance_matrix=distance_matrix, source_response=source_response)
print(stimulus)


# ------------------------------------------------------------------
# Exercise 2 (illustration): a non-BOLD source needs an HRF convolution
# ------------------------------------------------------------------
from scipy.stats import gamma

tr = 1.5
t_hrf = np.arange(0, 30, tr)                                   # 30 s of HRF, sampled every TR
hrf = gamma.pdf(t_hrf, 6) - 0.35 * gamma.pdf(t_hrf, 16)        # simple double-gamma HRF
hrf /= hrf.sum()

neural = np.zeros(120)
neural[[20, 50, 51, 52, 90]] = 1.0                             # brief bursts of activity
bold_pred = np.convolve(neural, hrf)[: len(neural)]

fig, ax = plt.subplots(figsize=(10, 3), layout="constrained")
ax.plot(neural, label="weighted source activity (neural, e.g. ECoG)")
ax.plot(bold_pred / bold_pred.max(), label="after HRF convolution (predicted BOLD, scaled)")
ax.set(xlabel="Time frame (TR)", ylabel="a.u.")
ax.legend()
plt.show()

# ------------------------------------------------------------------
# Submodels and full CF model
# ------------------------------------------------------------------
from prfmodel.models.cf import CFStimulusEncoder, GaussianCFModel, GaussianCFTuning

from prfmodel.scaling import BaselineAmplitude

cf_tuning = GaussianCFTuning()
cf_encoder = CFStimulusEncoder()
scaling_model = BaselineAmplitude()
cf_model = GaussianCFModel()

# Tuning profile for one example CF
tuning_params = pd.DataFrame({"center_index": [448], "sigma": [2.0]})
simulated_tuning_profile = cf_tuning(stimulus, tuning_params)
print("Tuning profile:", simulated_tuning_profile.shape)        # one value for every V1 vertex

# Stimulus encoding
simulated_encoded_profile = cf_encoder(stimulus, simulated_tuning_profile, tuning_params)
print("Encoded profile:", simulated_encoded_profile.shape)      # one value per time frame

# Scaling
scaling_params = pd.DataFrame({"baseline": [0.0], "amplitude": [1.0]})
scaled_response = scaling_model(simulated_encoded_profile, scaling_params)
print("Scaled response:", scaled_response.shape)

# Full model
start_params = pd.DataFrame({"center_index": [448], "sigma": [2.0], "baseline": [0.0], "amplitude": [1.0]})
simulated_response = cf_model(stimulus, start_params)
print("Full model prediction:", simulated_response.shape)

# ------------------------------------------------------------------
# Interactive figures (matplotlib sliders instead of ipywidgets)
# ------------------------------------------------------------------
SHOW_INTERACTIVE = True

from matplotlib.widgets import Slider, TextBox
from plotting_helpers import BLUE, GRAY, ORANGE

# Vertices without tuning are light gray, so that the outline of V1 stays visible
TUNING_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "tuning", ["#e6e6e6", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
)
SOURCE_CMAP = mpl.colors.LinearSegmentedColormap.from_list("source", ["#f2f2f2", "#000000"])

v1_indices_left = np.where(surface.atlas["left"] == label_v1)[0]
v1_indices_right = np.where(surface.atlas["right"] == label_v1)[0]
num_left = len(v1_indices_left)
distance_matrix_left = distance_matrix[:num_left, :num_left]
distance_matrix_right = distance_matrix[num_left:, num_left:]
time_frames = np.arange(source_response.shape[1])

# Starting value, minimum, maximum, and step size of the slider for each parameter
PARAM_SLIDERS = {
    "center_index": (448, 0, len(distance_matrix) - 1, 1),
    "sigma": (2.0, 0.5, 20.0, 0.5),
    "baseline": (0.0, -2.0, 2.0, 0.1),
    "amplitude": (1.0, 0.0, 3.0, 0.1),
    "time_frame": (0, 0, len(time_frames) - 1, 1),
}
START_VALUES = {name: value for name, (value, *_) in PARAM_SLIDERS.items()}


def embed_in_plane(distances: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Place vertices on a plane so that the distances between them approximate `distances` (classical MDS)."""
    distances = np.where(np.isfinite(distances), distances, np.nanmax(distances[np.isfinite(distances)]))
    num_vertices = distances.shape[0]
    centering = np.eye(num_vertices) - 1.0 / num_vertices
    gram = -0.5 * centering @ np.square(distances) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    plane_coords = eigenvectors[:, [-1, -2]] * np.sqrt(eigenvalues[[-1, -2]])
    for axis, brain_axis in enumerate([1, 2]):
        if np.corrcoef(plane_coords[:, axis], coords[:, brain_axis])[0, 1] < 0:
            plane_coords[:, axis] *= -1
    return plane_coords


v1_maps = {
    "Left V1": embed_in_plane(distance_matrix_left, surface.mesh.parts["left"].coordinates[v1_indices_left]),
    "Right V1": embed_in_plane(distance_matrix_right, surface.mesh.parts["right"].coordinates[v1_indices_right]),
}
v1_maps["Right V1"][:, 1] += v1_maps["Left V1"][:, 1].min() - v1_maps["Right V1"][:, 1].max() - 15.0  # 15 mm gap
v1_map = np.concatenate(list(v1_maps.values()))


def draw_v1_map(ax, **scatter_kwargs):
    """Draw every V1 vertex as a dot on the map of V1."""
    points = ax.scatter(v1_map[:, 0], v1_map[:, 1], s=6, linewidths=0, **scatter_kwargs)
    for name, hemi_map in v1_maps.items():
        ax.text(hemi_map[:, 0].min(), hemi_map[:, 1].max() + 3.0, name, color="#595959", fontsize=9)
    x_start, y_bar = v1_map[:, 0].min(), v1_map[:, 1].min() - 5.0
    ax.plot([x_start, x_start + 10.0], [y_bar, y_bar], color="#595959", linewidth=2)
    ax.text(x_start + 12.0, y_bar, "10 mm", color="#595959", fontsize=9, va="center")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_xlabel("Posterior → anterior")
    ax.set_ylabel("Ventral → dorsal")
    return points


def draw_cf_location(ax, center_index: int, sigma: float, fill: bool = True):
    """Draw the CF centre and size (circle with radius sigma); returns the artists so they can be removed."""
    center = v1_map[center_index]
    circle = ax.add_patch(mpl.patches.Circle(
        center, sigma, facecolor=BLUE if fill else "none", alpha=0.25 if fill else 1.0, edgecolor=BLUE, linewidth=2))
    (marker,) = ax.plot(*center, "o", color=BLUE, markersize=6, markeredgecolor="white", markeredgewidth=1)
    return [circle, marker]


def add_controls(fig, names, on_change):
    """Add one slider per parameter at the bottom of the figure (plus a text box to type a centre vertex)."""
    controls = {}
    for i, name in enumerate(reversed(names)):
        value, vmin, vmax, step = PARAM_SLIDERS[name]
        ax_s = fig.add_axes([0.15, 0.03 + 0.05 * i, 0.55, 0.03])
        controls[name] = Slider(ax_s, name, vmin, vmax, valinit=value, valstep=step)
        controls[name].on_changed(lambda _: on_change())
    if "center_index" in names:
        ax_box = fig.add_axes([0.82, 0.03 + 0.05 * (len(names) - 1 - names.index("center_index")), 0.08, 0.03])
        box = TextBox(ax_box, "vertex ", initial=str(PARAM_SLIDERS["center_index"][0]))

        def submit(text):
            try:
                controls["center_index"].set_val(int(np.clip(int(text), 0, len(distance_matrix) - 1)))
            except ValueError:
                pass

        box.on_submit(submit)
        controls["_box"] = box
    return controls


def tuning_values(center_index, sigma):
    params = pd.DataFrame({"center_index": [int(center_index)], "sigma": [float(sigma)]})
    return np.asarray(cf_tuning(stimulus, params)).reshape(-1)


def encoded_values(center_index, sigma):
    params = pd.DataFrame({"center_index": [int(center_index)], "sigma": [float(sigma)]})
    return np.asarray(cf_encoder(stimulus, cf_tuning(stimulus, params), params)).reshape(-1)


def full_prediction(center_index, sigma, baseline, amplitude):
    params = pd.DataFrame({"center_index": [int(center_index)], "sigma": [float(sigma)],
                           "baseline": [float(baseline)], "amplitude": [float(amplitude)]})
    return np.asarray(cf_model(stimulus, params)).reshape(-1)


def scaled_values(sigma, baseline, amplitude):
    """Encoded response (centre fixed at the start value) before and after scaling."""
    encoded = encoded_values(START_VALUES["center_index"], sigma)
    params = pd.DataFrame({"baseline": [float(baseline)], "amplitude": [float(amplitude)]})
    scaled = np.asarray(scaling_model(encoded[np.newaxis], params)).reshape(-1)
    return encoded, scaled


interactive_controls = []  # keep references, otherwise the sliders stop responding

if SHOW_INTERACTIVE:
    # ---- Figure 1: tuning profile (center_index, sigma) ----
    fig = plt.figure(figsize=(12, 6.5))
    ax_map = fig.add_axes([0.05, 0.25, 0.45, 0.68])
    ax_prof = fig.add_axes([0.58, 0.32, 0.38, 0.55])
    values = tuning_values(448, 2.0)
    points = draw_v1_map(ax_map, c=values, cmap=TUNING_CMAP, vmin=0, vmax=values.max())
    cf_artists = draw_cf_location(ax_map, 448, 2.0, fill=False)
    d = np.where(np.isfinite(distance_matrix[448]), distance_matrix[448], np.nan)
    prof = ax_prof.scatter(d, values, s=4, color=BLUE)
    ax_prof.set(xlabel="Distance to CF centre (mm)", ylabel="Tuning profile", xlim=(0, 40))
    ax_map.set_title("Tuning profile on the map of V1", loc="left")

    def update_tuning():
        c, s = int(ctrl1["center_index"].val), ctrl1["sigma"].val
        v = tuning_values(c, s)
        points.set_array(v)
        points.set_clim(0, v.max())
        for a in cf_artists:
            a.remove()
        cf_artists[:] = draw_cf_location(ax_map, c, s, fill=False)
        dist = np.where(np.isfinite(distance_matrix[c]), distance_matrix[c], np.nan)
        prof.set_offsets(np.c_[dist, v])
        ax_prof.set_ylim(0, v.max() * 1.05)
        fig.canvas.draw_idle()

    ctrl1 = add_controls(fig, ["center_index", "sigma"], update_tuning)
    interactive_controls.append(ctrl1)
    plt.show()

    # ---- Figure 2: stimulus encoding (center_index, sigma, time_frame), as in the notebook ----
    # Use the same colour range for all time frames (1st and 99th percentile of the source response)
    source_limits = np.nanpercentile(source_response, [1, 99])

    # Start at the time frame where the starting CF predicts the highest activity
    enc = encoded_values(START_VALUES["center_index"], START_VALUES["sigma"])
    start_frame = int(np.argmax(enc))
    PARAM_SLIDERS["time_frame"] = (start_frame, 0, len(time_frames) - 1, 1)

    fig = plt.figure(figsize=(13, 6.5))
    ax_map = fig.add_axes([0.04, 0.3, 0.36, 0.62])
    ax_cbar = fig.add_axes([0.41, 0.35, 0.012, 0.5])
    ax_ts = fig.add_axes([0.52, 0.38, 0.45, 0.5])

    # Vertices with a strong response are dark
    points2 = draw_v1_map(ax_map, c=source_response[:, start_frame], cmap=SOURCE_CMAP,
                          vmin=source_limits[0], vmax=source_limits[1])
    fig.colorbar(points2, cax=ax_cbar, label="BOLD response (PSC)")
    cf_artists2 = draw_cf_location(ax_map, START_VALUES["center_index"], START_VALUES["sigma"], fill=False)
    map_title = ax_map.set_title(f"Source region response and CF at time frame {start_frame}", loc="left")

    (enc_line,) = ax_ts.plot(time_frames, enc, color=BLUE, linewidth=2)
    frame_line = ax_ts.axvline(start_frame, color=GRAY, linestyle="--", linewidth=1)
    (enc_dot,) = ax_ts.plot([start_frame], [enc[start_frame]], "o", color=BLUE, markersize=9,
                            markeredgecolor="white", markeredgewidth=2)
    ax_ts.set_title("Predicted target neuron population activity", loc="left")
    ax_ts.set(xlabel="Time frame", ylabel="Predicted activity")

    def update_encoding():
        c, s, t = int(ctrl2["center_index"].val), ctrl2["sigma"].val, int(ctrl2["time_frame"].val)
        points2.set_array(source_response[:, t])
        for a in cf_artists2:
            a.remove()
        cf_artists2[:] = draw_cf_location(ax_map, c, s, fill=False)
        map_title.set_text(f"Source region response and CF at time frame {t}")
        e = encoded_values(c, s)
        enc_line.set_ydata(e)
        frame_line.set_xdata([t, t])
        enc_dot.set_data([t], [e[t]])
        ax_ts.relim()
        ax_ts.autoscale_view()
        fig.canvas.draw_idle()

    ctrl2 = add_controls(fig, ["center_index", "sigma", "time_frame"], update_encoding)
    interactive_controls.append(ctrl2)
    plt.show()

    # ---- Figure 3: scaling (baseline, amplitude, sigma), as in the notebook ----
    fig = plt.figure(figsize=(12, 6))
    ax_sc = fig.add_axes([0.08, 0.35, 0.88, 0.55])
    _, start_scaled = scaled_values(START_VALUES["sigma"], START_VALUES["baseline"], START_VALUES["amplitude"])
    enc0, sc0 = scaled_values(START_VALUES["sigma"], START_VALUES["baseline"], START_VALUES["amplitude"])
    ax_sc.plot(time_frames, start_scaled, color=GRAY, linestyle="--", linewidth=1.5, label="Starting parameters (scaled)")
    (before_line,) = ax_sc.plot(time_frames, enc0, color=BLUE, linewidth=2, label="Before scaling")
    (after_line,) = ax_sc.plot(time_frames, sc0, color=ORANGE, linewidth=2, label="After scaling")
    ax_sc.legend(loc="upper right")
    ax_sc.set_title("Encoded response before and after scaling", loc="left")
    ax_sc.set(xlabel="Time frame", ylabel="Predicted response")

    def update_scaling():
        e, sc = scaled_values(ctrl_sc["sigma"].val, ctrl_sc["baseline"].val, ctrl_sc["amplitude"].val)
        before_line.set_ydata(e)
        after_line.set_ydata(sc)
        ax_sc.relim()
        ax_sc.autoscale_view()
        fig.canvas.draw_idle()

    ctrl_sc = add_controls(fig, ["baseline", "amplitude", "sigma"], update_scaling)
    interactive_controls.append(ctrl_sc)
    plt.show()

    # ---- Figure 4: full model (all four parameters), as in the notebook ----
    def to_frame(**params: float) -> pd.DataFrame:
        """One-row parameter table, like plotting_helpers.to_frame."""
        return pd.DataFrame({name: [value] for name, value in params.items()})

    full_names = ["center_index", "sigma", "baseline", "amplitude"]
    start_response = np.asarray(cf_model(stimulus, to_frame(**{k: START_VALUES[k] for k in full_names}))).reshape(-1)

    fig = plt.figure(figsize=(12, 7))
    ax_map = fig.add_axes([0.04, 0.33, 0.34, 0.6])
    ax_ts = fig.add_axes([0.45, 0.42, 0.52, 0.48])
    draw_v1_map(ax_map, color="#d9d9d9")
    cf_artists3 = draw_cf_location(ax_map, START_VALUES["center_index"], START_VALUES["sigma"])
    ax_map.set_title("CF on V1", loc="left")

    ax_ts.plot(time_frames, start_response, color=GRAY, linestyle="--", linewidth=1.5, label="Starting parameters")
    (pred_line,) = ax_ts.plot(time_frames, start_response, color=BLUE, linewidth=2, label="Current parameters")
    ax_ts.legend(loc="upper right")
    ax_ts.set_title("Predicted BOLD response", loc="left")
    ax_ts.set(xlabel="Time frame", ylabel="Predicted BOLD response")

    def update_full():
        params = {name: ctrl3[name].val for name in full_names}
        params["center_index"] = int(params["center_index"])
        for a in cf_artists3:
            a.remove()
        cf_artists3[:] = draw_cf_location(ax_map, params["center_index"], params["sigma"])
        pred_line.set_ydata(np.asarray(cf_model(stimulus, to_frame(**params))).reshape(-1))
        ax_ts.relim()
        ax_ts.autoscale_view()
        fig.canvas.draw_idle()

    ctrl3 = add_controls(fig, full_names, update_full)
    interactive_controls.append(ctrl3)
    plt.show()


# ======================================================================
# Fitting the model to data, evaluation, interpretation and exercises
# ======================================================================
# Fitting every vertex of the brain (as in the notebook) can take a long time on a CPU.
# Set to False to fit only the vertices in V1-V4 (faster).
FIT_ALL_VERTICES = True

# ------------------------------------------------------------------
# Exercise 3: V1 response heatmap
# ------------------------------------------------------------------
from prfmodel.plotting import plot_response_heatmap

try:
    plot_response_heatmap(source_response)
except Exception as err:  # fall back to a plain image if the helper's arguments differ
    print("plot_response_heatmap failed:", err)
    plt.figure(figsize=(10, 4))
    plt.imshow(source_response, aspect="auto", cmap="RdBu_r", vmin=-5, vmax=5)
    plt.colorbar(label="PSC")
    plt.xlabel("Time frame (TR)")
    plt.ylabel("V1 vertex")
plt.show()

# Sorting the V1 vertices by the time of their peak response makes the travelling waves visible
order = np.argsort(np.nanargmax(np.nan_to_num(source_response), axis=1))
plt.figure(figsize=(10, 4))
plt.imshow(source_response[order], aspect="auto", cmap="RdBu_r", vmin=-5, vmax=5)
plt.colorbar(label="PSC")
plt.xlabel("Time frame (TR)")
plt.ylabel("V1 vertex (sorted by peak time)")
plt.title("V1 response, vertices sorted by the time of their peak")
plt.show()

# ------------------------------------------------------------------
# Surface plots: SD and visual areas
# ------------------------------------------------------------------
from plotting_helpers import make_surface_plot

plot_surf_stat_map_helper = make_surface_plot(surface.mesh)

response_sd = response_psc.std(axis=1)
plot_surf_stat_map_helper(response_sd, vmax=5.0, title="BOLD response standard deviation (in PSC)")
plt.show()

visual_area_labels = {"V1": 1, "V2": 4, "V3": 5, "V4": 6}
atlas_labels = np.concatenate([surface.atlas["left"], surface.atlas["right"]])

visual_area = np.full(atlas_labels.shape, "", dtype=object)
for name, label in visual_area_labels.items():
    visual_area[atlas_labels == label] = name

visual_area_number = np.full(atlas_labels.shape, np.nan)
for number, name in enumerate(visual_area_labels, start=1):
    visual_area_number[visual_area == name] = number

fig, _ = plot_surf_stat_map_helper(
    visual_area_number, vmin=0.5, vmax=4.5,
    cmap=mpl.colors.ListedColormap(["#2a78d6", "#eb6834", "#1baf7a", "#8c5bd6"]),
    title="Visual areas V1 to V4",
)
fig.axes[-1].set_yticks([1, 2, 3, 4], labels=list(visual_area_labels))
plt.show()

# ------------------------------------------------------------------
# Valid vertices
# ------------------------------------------------------------------
response_is_valid = np.all(np.isfinite(response_psc), axis=1)
if not FIT_ALL_VERTICES:
    response_is_valid &= visual_area != ""
response_valid = response_psc[response_is_valid]
print(f"Fraction of vertices fitted: {response_is_valid.mean():.3f} ({response_is_valid.sum()} vertices)")


def fill_valid_vertices(values: np.ndarray) -> np.ndarray:
    """Fill values for the valid vertices into an array with one value per vertex."""
    values_full = np.full((response_psc.shape[0],), fill_value=np.nan)
    values_full[response_is_valid] = values
    return values_full


# ------------------------------------------------------------------
# Grid search + least squares (V1 -> all vertices)
# ------------------------------------------------------------------
from prfmodel.fitters import GridFitter, LeastSquaresFitter
from prfmodel.plotting import (plot_observed_predicted, plot_parameter_by_roi, plot_r_squared_hist)
from prfmodel.utils import batched, calculate_r_squared

grid_param_ranges = {
    "center_index": np.arange(len(distance_matrix)),  # Every V1 vertex
    "sigma": np.geomspace(0.5, 20.0, 10),             # Log-spaced (in mm)
    "baseline": [0.0],
    "amplitude": [1.0],
}


def fit_cf(stim, data, grid_ranges):
    """Grid search for center_index and sigma, then least squares for baseline and amplitude."""
    _, grid_p = GridFitter(model=cf_model, stimulus=stim, compile_step=True).fit(
        data=data, parameter_values=grid_ranges, batch_size=20)
    _, ls_p = LeastSquaresFitter(model=cf_model, stimulus=stim).fit(
        data=data, parameters=grid_p, slope_name="amplitude", intercept_name="baseline", batch_size=200)
    return ls_p


tag = "all" if FIT_ALL_VERTICES else "v1to4"
ls_params = cached(f"cf_v1_{tag}", lambda: fit_cf(stimulus, response_valid, grid_param_ranges))
print(ls_params)

# ---- Figure: distribution of the grid search estimates (as in the notebook) ----
# Least squares does not change center_index and sigma, so ls_params holds the grid-search values for them
from prfmodel.plotting import plot_grid_parameter_distribution

fig, (ax_map, ax_sigma) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 1.3]},
                                       layout="constrained")
center_counts = np.bincount(ls_params["center_index"].astype(int), minlength=len(distance_matrix))
points = draw_v1_map(ax_map, c=center_counts, cmap=TUNING_CMAP, norm=mpl.colors.SymLogNorm(linthresh=1.0, vmin=0.0))
fig.colorbar(points, ax=ax_map, label="Number of vertices")
ax_map.set_title("CF centers", loc="left")
try:
    plot_grid_parameter_distribution(ls_params, grid_param_ranges, "sigma", ax=ax_sigma)
except Exception as err:  # fall back to a simple bar chart
    print("plot_grid_parameter_distribution failed:", err)
    counts = ls_params["sigma"].round(3).value_counts().sort_index()
    ax_sigma.bar(range(len(counts)), counts.to_numpy(),
                 color=["tab:red" if i in (0, len(counts) - 1) else "tab:blue" for i in range(len(counts))])
    ax_sigma.set_xticks(range(len(counts)), [f"{s:.1f}" for s in counts.index], rotation=45)
ax_sigma.set(xlabel="sigma (in mm)", title="CF sizes")
plt.show()

# Grid distribution of sigma (values at the grid floor/ceiling are suspicious)
fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
sig_counts = ls_params["sigma"].round(3).value_counts().sort_index()
ax.bar(range(len(sig_counts)), sig_counts.to_numpy(),
       color=["tab:red" if i in (0, len(sig_counts) - 1) else "tab:blue" for i in range(len(sig_counts))])
ax.set_xticks(range(len(sig_counts)), [f"{s:.1f}" for s in sig_counts.index], rotation=45)
ax.set(xlabel="CF size sigma (mm, grid values)", ylabel="Number of vertices")
plt.show()

predict_batched = batched(cf_model)
pred_response = np.asarray(predict_batched(stimulus, ls_params, batch_size=200))
r_squared = np.asarray(calculate_r_squared(response_valid, pred_response))

# ------------------------------------------------------------------
# Exercise 4: R-squared distribution
# ------------------------------------------------------------------
plot_r_squared_hist(r_squared)
plt.show()

is_v1_valid = is_v1[response_is_valid]
print(f"Median R² of V1 vertices:          {np.median(r_squared[is_v1_valid]):.3f}")
print(f"Fraction of R² > 0.99 that are V1:  {np.mean(is_v1_valid[r_squared > 0.99]):.3f}")
print(f"Median sigma of V1 vertices (mm):   {np.median(ls_params['sigma'][is_v1_valid]):.2f}")

plot_surf_stat_map_helper(fill_valid_vertices(r_squared), vmin=0.0, vmax=1.0, title="Variance explained (R-squared)")
plt.show()

# Median-R² vertex in V2, V3, V4
visual_area_valid = visual_area[response_is_valid]
fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True, layout="constrained")
for ax, area in zip(axes, ["V2", "V3", "V4"]):
    area_indices = np.flatnonzero(visual_area_valid == area)
    vertex_index = area_indices[np.argsort(r_squared[area_indices])[len(area_indices) // 2]]
    plot_observed_predicted(response_valid[vertex_index], pred_response[vertex_index], ax=ax)
    ax.set(xlabel="", ylabel="BOLD (PSC)",
           title=f"{area}: R² = {r_squared[vertex_index]:.2f}, "
                 f"sigma = {ls_params['sigma'].iloc[vertex_index]:.1f} mm")
axes[-1].set_xlabel("Time frame (in TR)")
plt.show()

# ------------------------------------------------------------------
# Interpreting the parameters: selection
# ------------------------------------------------------------------
center_index = ls_params["center_index"].to_numpy().astype(int)
sigma = ls_params["sigma"].to_numpy()

r_squared_threshold = 0.5
selection_criteria = pd.DataFrame({
    f"r_squared > {r_squared_threshold}": r_squared > r_squared_threshold,
    "outside V1": ~is_v1_valid,
    "sigma below grid ceiling": ~np.isclose(sigma, np.max(grid_param_ranges["sigma"])),
})
is_selected = selection_criteria.all(axis=1).to_numpy()


def fill_selected(values: np.ndarray) -> np.ndarray:
    """Fill values of the selected vertices into an array with one value per vertex; all others are NaN."""
    return fill_valid_vertices(np.where(is_selected, values, np.nan))


print(selection_criteria.assign(all_criteria=is_selected).mean())

plot_surf_stat_map_helper(fill_selected(sigma), vmin=0.0, vmax=10.0, title="CF size (sigma, in mm)")
plt.show()

plot_parameter_by_roi(
    ls_params[is_selected], "sigma", visual_area_valid[is_selected],
    order=["V2", "V3", "V4"], error="sem",
    xlabel="Visual area", ylabel="CF size (sigma, in mm)", figsize=(4, 4),
)
plt.show()

# ------------------------------------------------------------------
# Exercise 6: numbers behind the ROI comparison
# ------------------------------------------------------------------
sigma_by_area = pd.DataFrame({"area": visual_area_valid[is_selected], "sigma": sigma[is_selected]})
print(sigma_by_area.groupby("area")["sigma"].agg(["count", "mean", "median", "sem"]).reindex(["V2", "V3", "V4"]))

# ------------------------------------------------------------------
# Exercise 7: z-scored source response
# ------------------------------------------------------------------
with np.errstate(invalid="ignore", divide="ignore"):
    response_z = (response_raw - response_raw.mean(axis=1, keepdims=True)) / response_raw.std(axis=1, keepdims=True)

stimulus_z = CFStimulus(distance_matrix=distance_matrix, source_response=response_z[is_v1])

# To save time, only refit the selected target vertices (outside V1, good fit)
targets = response_valid[is_selected]
ls_params_z = cached(f"cf_v1_zscored_{tag}", lambda: fit_cf(stimulus_z, targets, grid_param_ranges))
pred_z = np.asarray(predict_batched(stimulus_z, ls_params_z, batch_size=200))
r_squared_z = np.asarray(calculate_r_squared(targets, pred_z))

same_centre = ls_params_z["center_index"].to_numpy().astype(int) == center_index[is_selected]
centre_shift = distance_matrix[center_index[is_selected], ls_params_z["center_index"].to_numpy().astype(int)]
print("\nZ-scored vs PSC source response (selected target vertices):")
print(f"  same CF centre vertex:          {100 * same_centre.mean():.1f}%")
print(f"  median shift of the CF centre:  {np.median(centre_shift):.2f} mm")
print(f"  sigma identical:                {100 * np.isclose(ls_params_z['sigma'], sigma[is_selected]).mean():.1f}%")
print(f"  median R² PSC / z-scored:       {np.median(r_squared[is_selected]):.3f} / {np.median(r_squared_z):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
axes[0].scatter(sigma[is_selected], ls_params_z["sigma"], s=4, alpha=0.3)
axes[0].plot([0.5, 20], [0.5, 20], "k--", lw=0.8)
axes[0].set(xscale="log", yscale="log", xlabel="sigma, PSC source (mm)", ylabel="sigma, z-scored source (mm)")
axes[1].scatter(r_squared[is_selected], r_squared_z, s=4, alpha=0.3)
axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
axes[1].set(xlabel="R², PSC source", ylabel="R², z-scored source")
plt.show()

# How different are the scales of the V1 vertices in the first place?
v1_sd = source_response.std(axis=1)
print(f"SD of V1 vertices (PSC): 5th-95th percentile = {np.percentile(v1_sd, 5):.2f} - {np.percentile(v1_sd, 95):.2f}")

# ------------------------------------------------------------------
# Exercise 9: V2 as source, V3 and V4 as targets
# ------------------------------------------------------------------
label_v2 = 4
is_v2 = atlas_labels == label_v2
distance_matrix_v2 = cached("distance_matrix_v2", lambda: make_distance_matrix(label_v2))
stimulus_v2 = CFStimulus(distance_matrix=distance_matrix_v2, source_response=response_psc[is_v2])

is_v3v4_valid = np.isin(visual_area_valid, ["V3", "V4"])
grid_param_ranges_v2 = grid_param_ranges | {"center_index": np.arange(len(distance_matrix_v2))}
ls_params_v2 = cached(f"cf_v2_to_v3v4_{tag}",
                      lambda: fit_cf(stimulus_v2, response_valid[is_v3v4_valid], grid_param_ranges_v2))
pred_v2 = np.asarray(predict_batched(stimulus_v2, ls_params_v2, batch_size=200))
r_squared_v2 = np.asarray(calculate_r_squared(response_valid[is_v3v4_valid], pred_v2))

sel_v2 = (r_squared_v2 > r_squared_threshold) & ~np.isclose(ls_params_v2["sigma"], grid_param_ranges["sigma"].max())
areas_v2 = visual_area_valid[is_v3v4_valid]
print("\nCF size with V2 as source:")
print(pd.DataFrame({"area": areas_v2[sel_v2], "sigma": ls_params_v2["sigma"].to_numpy()[sel_v2]})
      .groupby("area")["sigma"].agg(["count", "mean", "median", "sem"]))

plot_parameter_by_roi(
    ls_params_v2[sel_v2], "sigma", areas_v2[sel_v2], order=["V3", "V4"], error="sem",
    xlabel="Target area (source: V2)", ylabel="CF size (sigma, in mm of V2)", figsize=(4, 4),
)
plt.show()


# This is the encoded response: the Gaussian-weighted average of V1 activity inside the CF patch, at every time frame. It's the prediction before scaling (amplitude and baseline), for whatever target vertex would have this CF.
# Summary of understanding: the CF model is a linear model that predicts the target vertex's response from a weighted sum of the source region's responses. The weights are determined by a Gaussian tuning profile, which is parameterized by its centre (a vertex in the source region) and its size (sigma). The model also includes a baseline and an amplitude parameter to account for scaling differences.

# The shape matches well: a broad peak at frame 4. Only the scale differs, and the amplitude parameter fixes that (×1.6), just like in the pRF model.


# The centre tells you which part of V1 the target is "connected" to. That's why, in Exercise 10, the CF centre's pRF should match the target's own pRF.
# The size tells you how much of V1 it integrates. Higher areas (V3, V4) have broad responses (large pRFs), so they need larger patches. That's the result of Exercise 6: CF size grows from V2 to V3 to V4
# One refinement: the CF doesn't look for one V1 vertex whose response is close to V3. It looks for a patch of V1: a Gaussian-weighted average of neighbouring V1 vertices whose combined response is closest to the V3 response.