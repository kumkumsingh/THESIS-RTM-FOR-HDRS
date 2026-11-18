import os

import pandas as pd

# Set keras backend to 'tensorflow' (this is normally the default)
os.environ["KERAS_BACKEND"] = "tensorflow"
# Print parameter DataFrames with three decimals
pd.set_option("display.precision", 3)

import prfmodel 
import tensorflow as tf

print("prfmodel version:", prfmodel.__version__)
print("TensorFlow version:", tf.__version__)

from prfmodel.examples import load_dataset

# Downloads on first use and caches in a user data directory (see prfmodel.examples.get_data_dir)
dataset = load_dataset("7t-aot-visual", surface_type="inflated")

stimulus = dataset.stimulus

print('stimulus1', stimulus)

# import matplotlib as mpl
# import imageio_ffmpeg
# mpl.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

# from IPython.display import HTML
# from prfmodel.plotting import animate_2d_prf_stimulus

# ani = animate_2d_prf_stimulus(stimulus, interval=25)
# HTML(ani.to_html5_video())


import numpy as np
from prfmodel.stimuli import PRFStimulus

# Cells that the bar covers in at least one time frame lie on the screen
is_on_screen = stimulus.design.max(axis=0) > 0.0

# The screen is a rectangle, so we keep the rows (y) and columns (x) of the design matrix between its edges
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

print(stimulus)

# The grid stores y in grid[..., 0] and x in grid[..., 1]
(y_min, x_min), (y_max, x_max) = stimulus.grid.min(axis=(0, 1)), stimulus.grid.max(axis=(0, 1))
print(f"Screen: x from {x_min:.2f} to {x_max:.2f} degrees, y from {y_min:.2f} to {y_max:.2f} degrees")


# @title Helper functions for the interactive figures { display-mode: "form" }
import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from plotting_helpers import BLUE, GRAY, ORANGE, add_legend, make_slider, show_interactive, style_axes, to_frame

TUNING_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "tuning", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
)

# Visual field coordinates of the stimulus grid (dimensions are ordered as y, x)
x_coords = stimulus.grid[0, :, 1]
y_coords = stimulus.grid[:, 0, 0]
FIELD_EXTENT = (x_coords[0], x_coords[-1], y_coords[0], y_coords[-1])

time_frames = np.arange(stimulus.design.shape[0])

# Starting value, minimum, maximum, and step size of the slider for each parameter
PARAM_SLIDERS = {
    "mu_x": (0.0, -4.0, 4.0, 0.1),
    "mu_y": (0.0, -4.0, 4.0, 0.1),
    "sigma": (1.0, 0.1, 4.0, 0.1),
    "weight_deriv": (-0.5, -2.0, 2.0, 0.1),
    "baseline": (0.0, -100.0, 100.0, 5.0),
    "amplitude": (1.0, 0.0, 3.0, 0.1),
}
START_VALUES = {name: value for name, (value, *_) in PARAM_SLIDERS.items()}


def make_sliders(*names: str, **custom_sliders: tuple[float, float, float, float]) -> dict[str, widgets.FloatSlider]:
    """Create a slider for each parameter; `custom_sliders` replace the values from `PARAM_SLIDERS` for a parameter."""
    return {name: make_slider(name, *custom_sliders.get(name, PARAM_SLIDERS[name])) for name in names}


def draw_prf_location(ax: plt.Axes, mu_x: float, mu_y: float, sigma: float) -> None:
    """Draw the pRF center and size (a circle with radius `sigma`) in the visual field."""
    x_min, x_max, y_min, y_max = FIELD_EXTENT
    ax.add_patch(
        mpl.patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min, facecolor="#f2f2f2", edgecolor=GRAY, linewidth=1,
        )
    )
    ax.text(x_min, y_max + 0.3, "Stimulus area", color="#595959", fontsize=9)
    ax.add_patch(mpl.patches.Circle((mu_x, mu_y), sigma, facecolor=BLUE, alpha=0.25, edgecolor=BLUE, linewidth=2))
    ax.plot(mu_x, mu_y, "o", color=BLUE, markersize=6)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_aspect("equal")
    style_axes(ax, "x (degrees visual angle)", "y (degrees visual angle)")

from prfmodel.models.prf import Gaussian2DPRFTuning

prf_tuning = Gaussian2DPRFTuning()

tuning_params = pd.DataFrame(
    {
        "mu_x": [-2.1],
        "mu_y": [-1.4],
        "sigma": [0.7],
    },
)

# Predict the tuning profile
simulated_tuning_profile = prf_tuning(stimulus, tuning_params)

simulated_tuning_profile.shape  # One value for every cell of the stimulus grid

# @title Interactive figure: Tuning profile { display-mode: "form" }
def plot_tuning(**params: float) -> None:
    profile = prf_tuning(stimulus, to_frame(**params))[0]

    fig, ax = plt.subplots(figsize=(5.5, 4.5), layout="constrained")

    image = ax.imshow(profile, extent=FIELD_EXTENT, origin="lower", cmap=TUNING_CMAP)  # Positive y-coordinates point up
    fig.colorbar(image, ax=ax, label="Tuning")
    ax.set_title("Tuning profile")
    style_axes(ax, "x (degrees visual angle)", "y (degrees visual angle)", grid=False)

    plt.show()

show_interactive(plot_tuning, make_sliders("mu_x", "mu_y", "sigma"))

# def plot_tuning_side_by_side(param_sets, shared_scale=False):
#     """Plot several tuning profiles next to each other."""
#     profiles = [prf_tuning(stimulus, to_frame(**p))[0] for p in param_sets]
#     vmax = max(float(p.max()) for p in profiles) if shared_scale else None

#     fig, axes = plt.subplots(1, len(param_sets), figsize=(5.5 * len(param_sets), 4.5), layout="constrained")

#     for ax, profile, p in zip(axes, profiles, param_sets):
#         image = ax.imshow(profile, extent=FIELD_EXTENT, origin="lower", cmap=TUNING_CMAP, vmin=0, vmax=vmax)
#         fig.colorbar(image, ax=ax, label="Tuning")
#         ax.set_title(f"mu_x={p['mu_x']}, mu_y={p['mu_y']}, sigma={p['sigma']}")
#         style_axes(ax, "x (degrees visual angle)", "y (degrees visual angle)", grid=False)

#     plt.show()


# params = [
#     dict(mu_x=2.1, mu_y=1.4, sigma=2.0),
#     dict(mu_x=-2.1, mu_y=-1.4, sigma=0.7),
# ]

# plot_tuning_side_by_side(params)                     # each plot has its own colour scale
# plot_tuning_side_by_side(params, shared_scale=True)  # both plots on the same colour scale

# Stimulus encoding

from prfmodel.models.prf import PRFStimulusEncoder

prf_encoder = PRFStimulusEncoder()

# We first predict the tuning profile
simulated_tuning_profile = prf_tuning(stimulus, tuning_params)

# Then we stimulus-encode it
simulated_encoded_profile = prf_encoder(stimulus, simulated_tuning_profile, tuning_params)

simulated_encoded_profile.shape  # One value per time frame

# @title Interactive figure: Stimulus encoding { display-mode: "form" }
def predict_encoded(params: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Predict the tuning profile and the stimulus-encoded response."""
    profile = prf_tuning(stimulus, params)
    return profile[0], prf_encoder(stimulus, profile, params)[0]


def plot_encoding(time_frame: int, **params: float) -> None:
    profile, encoded = predict_encoded(to_frame(**params))

    fig, (ax_field, ax_response) = plt.subplots(
        1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 3]}, layout="constrained",
    )

    # Stimulated cells are black
    ax_field.imshow(stimulus.design[time_frame], extent=FIELD_EXTENT, origin="lower", cmap="Greys", vmin=0, vmax=1)
    if profile.max() > 0:
        ax_field.contour(
            x_coords, y_coords, profile, levels=profile.max() * np.array([0.25, 0.5, 0.75]), colors=BLUE, linewidths=1.5,
        )
    ax_field.set_title(f"Stimulus and tuning profile\nat time frame {time_frame}")
    style_axes(ax_field, "x (degrees visual angle)", "y (degrees visual angle)", grid=False)

    ax_response.plot(time_frames, encoded, color=BLUE, linewidth=2)
    ax_response.axvline(time_frame, color=GRAY, linestyle="--", linewidth=1)
    ax_response.plot(
        time_frame, encoded[time_frame], "o", color=BLUE, markersize=9, markeredgecolor="white", markeredgewidth=2,
    )
    ax_response.set_title("Predicted neuron population activity")
    style_axes(ax_response, "Time frame", "Predicted activity")

    plt.show()


# Start at the time frame where the stimulus overlaps most with the starting tuning profile
_, start_encoded = predict_encoded(to_frame(mu_x=0.0, mu_y=0.0, sigma=1.0))
time_frame_slider = widgets.IntSlider(
    value=int(np.argmax(start_encoded)),
    min=0,
    max=len(time_frames) - 1,
    description="time_frame",
    continuous_update=False,
)

show_interactive(plot_encoding, {"time_frame": time_frame_slider, **make_sliders("mu_x", "mu_y", "sigma")})

# Exercise 6:
# Why does the predicted neuron population activity have this particular shape? How does it relate to the experimental stimulus (compare it to the video and move the time_frame slider)?
# “The predicted neuron population activity is the overlap between the moving bar and the Gaussian pRF at each time frame. It is zero when the bar is far from the pRF or absent (blank periods), rises as the bar enters the pRF, peaks when the bar crosses the pRF centre, and falls as the bar leaves. Each of the eight peaks corresponds to one bar sweep crossing the centre of the visual field, where this pRF (centred at (0, 0) with σ = 1) is located. Because the pRF is circular and centred at fixation, all sweeps pass through its centre, so the peaks are symmetric and of similar height. Moving the time_frame slider confirms that peaks coincide with frames in which the bar passes through the blue contours.”

# Exercie 7:

param_sets = [
    dict(mu_x=0.0,  mu_y=0.0,  sigma=1.0),   # default, for reference
    dict(mu_x=2.1,  mu_y=1.4,  sigma=2.0),
    dict(mu_x=-2.1, mu_y=-1.4, sigma=0.7),
]

fig, axes = plt.subplots(len(param_sets), 2, figsize=(12, 3.2 * len(param_sets)),
                         gridspec_kw={"width_ratios": [1, 3]}, layout="constrained")

for (ax_field, ax_resp), p in zip(axes, param_sets):
    profile, encoded = predict_encoded(to_frame(**p))
    peak = int(np.argmax(encoded))

    ax_field.imshow(stimulus.design[peak], extent=FIELD_EXTENT, origin="lower", cmap="Greys", vmin=0, vmax=1)
    ax_field.contour(x_coords, y_coords, profile, levels=profile.max() * np.array([0.25, 0.5, 0.75]), colors=BLUE)
    ax_field.set_title(f"mu=({p['mu_x']}, {p['mu_y']}), sigma={p['sigma']}\nframe {peak}")

    ax_resp.plot(time_frames, encoded, color=BLUE, linewidth=2)
    ax_resp.axvline(peak, color=GRAY, linestyle="--")
    ax_resp.set_ylim(0, 130)                  # same y-scale for a fair comparison
    ax_resp.set_ylabel("Predicted activity")

axes[-1, 1].set_xlabel("Time frame")
plt.show()

# Exercise 7:
# Row 3: mu = (−2.1, −1.4), σ = 0.7 (small pRF, lower-left)
# Much taller peaks. A small pRF concentrates its sensitivity, and the bar covers almost all of it at once. The peaks are actually taller than 125; their tops are cut off because my code fixed the y-axis at 0–130. Remove ax_resp.set_ylim(0, 130) to see their full height.
# Very narrow peaks. The bar overlaps a small pRF for only a few frames.
# Fully on screen. All circles are inside the stimulus area, so the pRF is fully stimulated when the bar crosses it.
# Left panel (frame 222): the bar crosses at y ≈ −1.4, again matching the pRF’s height.

# Exercise 8:

param_sets = [
    dict(mu_x=0.0, mu_y=0.0, sigma=0.1),    # very small pRF
    dict(mu_x=0.0, mu_y=0.0, sigma=4.0),    # very large pRF
    dict(mu_x=3.9, mu_y=0.0, sigma=1.0),    # centre at the right edge of the screen
    dict(mu_x=4.0, mu_y=4.0, sigma=0.3),    # small pRF outside the screen (corner)
    dict(mu_x=4.0, mu_y=4.0, sigma=2.0),    # large pRF outside the screen
]

fig, axes = plt.subplots(len(param_sets), 2, figsize=(12, 2.8 * len(param_sets)),
                         gridspec_kw={"width_ratios": [1, 3]}, layout="constrained")

for (ax_field, ax_resp), p in zip(axes, param_sets):
    profile, encoded = predict_encoded(to_frame(**p))
    peak = int(np.argmax(encoded))

    ax_field.imshow(stimulus.design[peak], extent=FIELD_EXTENT, origin="lower", cmap="Greys", vmin=0, vmax=1)
    if profile.max() > 0:
        ax_field.contour(x_coords, y_coords, profile, levels=profile.max() * np.array([0.25, 0.5, 0.75]), colors=BLUE)
    ax_field.set_title(f"mu=({p['mu_x']}, {p['mu_y']}), sigma={p['sigma']}")

    ax_resp.plot(time_frames, encoded, color=BLUE, linewidth=2)
    ax_resp.set_title(f"max activity = {encoded.max():.2f}")

axes[-1, 1].set_xlabel("Time frame")
plt.show()


# Impulse response models for BOLD measurements

from prfmodel.impulse import DerivativeTwoGammaImpulse

impulse_model = DerivativeTwoGammaImpulse()

impulse_params = pd.DataFrame({"weight_deriv": [-0.5]})

simulated_impulse_response = impulse_model(impulse_params)

simulated_impulse_response.shape  # One value per time frame of the impulse response

# @title Interactive figure: Impulse response { display-mode: "form" }
no_deriv_impulse = impulse_model(to_frame(weight_deriv=0.0))[0]
impulse_frames = np.arange(len(no_deriv_impulse))

# Fix the y-axis to the range of the impulse responses for all values of the slider
_, deriv_min, deriv_max, deriv_step = PARAM_SLIDERS["weight_deriv"]
deriv_values = np.arange(deriv_min, deriv_max + deriv_step / 2, deriv_step)
all_impulses = np.asarray(impulse_model(pd.DataFrame({"weight_deriv": deriv_values})))
impulse_margin = 0.1 * np.ptp(all_impulses)
impulse_limits = (all_impulses.min() - impulse_margin, all_impulses.max() + impulse_margin)


def plot_impulse(weight_deriv: float) -> None:
    impulse = impulse_model(to_frame(weight_deriv=weight_deriv))[0]

    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")

    ax.axhline(0.0, color="#bdbdbd", linewidth=1)
    ax.plot(impulse_frames, no_deriv_impulse, color=GRAY, linestyle="--", linewidth=1.5, label="weight_deriv = 0")
    ax.plot(impulse_frames, impulse, color=BLUE, linewidth=2, label=f"weight_deriv = {weight_deriv:.1f}")
    ax.set_ylim(impulse_limits)
    add_legend(ax)
    ax.set_title("Impulse response", loc="left")
    style_axes(ax, "Time frame", "Predicted impulse response")

    plt.show()


show_interactive(plot_impulse, make_sliders("weight_deriv"))

# Different impulse responses for different values of weight_deriv
# fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
# ax.axhline(0, color="#bdbdbd", linewidth=1)

# for w in [-1.0, -0.5, 0.0, 0.5, 1.0]:
#     impulse = impulse_model(to_frame(weight_deriv=w))[0]
#     ax.plot(impulse_frames, impulse, linewidth=2 if w == 0 else 1.2, label=f"weight_deriv = {w}")

# ax.legend()
# ax.set_xlabel("Time frame")
# ax.set_ylabel("Predicted impulse response")
# plt.show()

#To predict the response that accounts for the shape of the BOLD measurement
from prfmodel.impulse import convolve_prf_impulse_response

convolution_params = pd.DataFrame(
    {
        "mu_x": [0.0],
        "mu_y": [0.0],
        "sigma": [1.0],
        "weight_deriv": [-0.5],
    },
)

# We first predict the tuning profile
simulated_tuning_profile = prf_tuning(stimulus, convolution_params)

# We stimulus-encode the tuning profile
simulated_encoded_profile = prf_encoder(stimulus, simulated_tuning_profile, convolution_params)

# We predict the impulse response
simulated_impulse_response = impulse_model(convolution_params)

# We convolve the encoded tuning profile with the impulse response
convolved_response = convolve_prf_impulse_response(simulated_encoded_profile, simulated_impulse_response)

convolved_response.shape

# @title Interactive figure: Convolution with the impulse response { display-mode: "form" }
def predict_convolved(params: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Predict the neuron population activity and its convolution with the impulse response."""
    _, encoded = predict_encoded(params)
    convolved = convolve_prf_impulse_response(encoded[np.newaxis], impulse_model(params))[0]
    return encoded, np.asarray(convolved)


def plot_convolution(**params: float) -> None:
    encoded, convolved = predict_convolved(to_frame(**params))

    fig, ax = plt.subplots(figsize=(12, 3.8), layout="constrained")

    ax.plot(time_frames, encoded, color=BLUE, linewidth=2, label="Neuron population activity")
    ax.plot(time_frames, convolved, color=ORANGE, linewidth=2, label="Convolved with impulse response")
    add_legend(ax)
    ax.set_title("Predicted response before and after convolution", loc="left")
    style_axes(ax, "Time frame", "Predicted response")

    plt.show()


show_interactive(plot_convolution, make_sliders("mu_x", "mu_y", "sigma", "weight_deriv"))

# Exercise 9

# Move the weight_deriv slider to change the impulse model parameter. How does the convolved response change and why?

weight_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

base = dict(mu_x=0.0, mu_y=0.0, sigma=1.0)

# Neural activity does not depend on weight_deriv, so compute it once
base_params = to_frame(**base, weight_deriv=0.0)
tuning = prf_tuning(stimulus, base_params)
neural = prf_encoder(stimulus, tuning, base_params)
neural_curve = np.asarray(neural)[0]

colors = plt.cm.coolwarm(np.linspace(0, 1, len(weight_values)))   # blue = negative, red = positive

fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")

for ax in (ax_full, ax_zoom):
    ax.plot(time_frames, neural_curve, color="black", linestyle="--", linewidth=1.5, label="Neural activity")
    ax.axhline(0, color="#bdbdbd", linewidth=1)

print("weight_deriv   first BOLD peak (frame)")
for w, c in zip(weight_values, colors):
    params = to_frame(**base, weight_deriv=w)
    impulse = impulse_model(params)
    bold = np.asarray(convolve_prf_impulse_response(neural, impulse))[0]

    for ax in (ax_full, ax_zoom):
        ax.plot(time_frames, bold, color=c, linewidth=2 if w == 0 else 1.3, label=f"weight_deriv = {w}")

    first_peak = int(np.argmax(bold[:60]))           # first peak lies before frame 60
    print(f"{w:>12}   {first_peak}")

ax_full.set_title("Predicted BOLD for different weight_deriv values (full run)", loc="left")
ax_zoom.set_title("Zoom on the first pair of peaks", loc="left")
ax_zoom.set_xlim(25, 100)
ax_zoom.set_xlabel("Time frame")
for ax in (ax_full, ax_zoom):
    ax.set_ylabel("Predicted response")
ax_full.legend(ncol=4, fontsize=8)
plt.show()      # first peak lies before

# Exercise 9 answer:
# The first peak of the BOLD response occurs later for more positive values of weight_deriv
# weight_deriv fine-tunes when each voxel’s BOLD response happens, so that differences in blood-flow timing aren’t mistaken for differences in pRF position.

# Exercise 10
# No, a pRF model for ECoG would not need convolution with a haemodynamic impulse response. The HIRF accounts for the slow, delayed blood-oxygenation response measured by fMRI, which peaks about 5–6 s after neural activity and lasts about 20 s. ECoG records the electrical activity of neural populations directly, with millisecond resolution, so the predicted neuron population activity can be compared with the measured signal more directly. At high temporal resolution, it may still be useful to account for short neural response dynamics (latency, transient responses, adaptation) with a much faster neural impulse response, but not with the BOLD HIRF.”

# Scaling models

from prfmodel.scaling import BaselineAmplitude

scaling_model = BaselineAmplitude()

scaling_params = pd.DataFrame({"baseline": [0.0], "amplitude": [1.0]})

scaled_response = scaling_model(convolved_response, scaling_params)

scaled_response.shape

# @title Interactive figure: Scaling { display-mode: "form" }
def predict_scaled(**params: float) -> tuple[np.ndarray, np.ndarray]:
    """Predict the convolved response before and after scaling."""
    params = to_frame(**{**START_VALUES, **params})
    _, convolved = predict_convolved(params)
    return convolved, scaling_model(convolved[np.newaxis], params)[0]


_, start_scaled = predict_scaled()


def plot_scaling(**params: float) -> None:
    convolved, scaled = predict_scaled(**params)

    fig, ax = plt.subplots(figsize=(12, 3.8), layout="constrained")

    ax.plot(time_frames, start_scaled, color=GRAY, linestyle="--", linewidth=1.5, label="Starting parameters (scaled)")
    ax.plot(time_frames, convolved, color=BLUE, linewidth=2, label="Before scaling")
    ax.plot(time_frames, scaled, color=ORANGE, linewidth=2, label="After scaling")
    add_legend(ax)
    ax.set_title("Convolved response before and after scaling", loc="left")
    style_axes(ax, "Time frame", "Predicted response")

    plt.show()


# show_interactive(plot_scaling, make_sliders("baseline", "amplitude", "sigma"))

# plot_scaling(baseline=0.0,   amplitude=1.0)   # starting point (all lines overlap)
# plot_scaling(baseline=30.0,  amplitude=1.0)   # only baseline changed
# plot_scaling(baseline=-30.0, amplitude=1.0)
# plot_scaling(baseline=0.0,   amplitude=2.0)   # only amplitude changed
# plot_scaling(baseline=0.0,   amplitude=0.3)
# plot_scaling(baseline=30.0,  amplitude=0.5)

# orange(t) = baseline + amplitude × blue(t)
# amplitude matches how strongly this voxel’s signal changes;
# baseline matches the resting level of this voxel’s signal.
# After scaling, the prediction and the data are in the same units, so the comparison is fair. 
# scaling lets the model match each voxel’s signal strength and resting level, so that mu_x, mu_y and σ only have to explain the shape and timing of the response. That’s what makes the pRF estimates comparable across voxels. 

# exercise 12

plot_scaling(sigma=1.0, amplitude=1.0)   # reference
plot_scaling(sigma=2.0, amplitude=1.0)   # bigger pRF, same amplitude → lower peaks
plot_scaling(sigma=2.0, amplitude=2.5)   # bigger pRF, more amplitude → height restored?
plot_scaling(sigma=0.5, amplitude=1.0)   # smaller pRF → taller peaks
plot_scaling(sigma=0.5, amplitude=0.5)   # smaller pRF, less amplitude → height reduced again

# The full canonical model
# Complete pRF model in one step
from prfmodel.models.prf import Gaussian2DPRFModel

prf_model = Gaussian2DPRFModel()

start_params = pd.DataFrame(
    {
        "mu_x": [0.0],
        "mu_y": [0.0],
        "sigma": [1.0],
        "weight_deriv": [-0.5],
        "baseline": [0.0],
        "amplitude": [1.0],
    },
)

# Make prediction with pRF model
simulated_response = prf_model(stimulus, start_params)

simulated_response.shape  # One predicted timecourse with one value per time frame

# @title Interactive figure: Full model { display-mode: "form" }
start_response = prf_model(stimulus, to_frame(**START_VALUES))[0]


def plot_prediction(**params: float) -> None:
    response = prf_model(stimulus, to_frame(**params))[0]

    fig, (ax_field, ax_response) = plt.subplots(
        1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 3]}, layout="constrained",
    )

    draw_prf_location(ax_field, params["mu_x"], params["mu_y"], params["sigma"])
    ax_field.set_title("pRF in the visual field")

    ax_response.plot(time_frames, start_response, color=GRAY, linestyle="--", linewidth=1.5, label="Starting parameters")
    ax_response.plot(time_frames, response, color=BLUE, linewidth=2, label="Current parameters")
    add_legend(ax_response)
    ax_response.set_title("Predicted BOLD response", loc="left")
    style_axes(ax_response, "Time frame", "Predicted BOLD response")

    plt.show()


show_interactive(plot_prediction, make_sliders("mu_x", "mu_y", "sigma", "weight_deriv", "baseline", "amplitude"))

# Extra bcos slider is not running here 
from matplotlib.widgets import Slider

slider_names = ["mu_x", "mu_y", "sigma", "weight_deriv", "baseline", "amplitude"]

fig = plt.figure(figsize=(13, 7))
ax_field = fig.add_axes([0.05, 0.42, 0.25, 0.50])
ax_resp = fig.add_axes([0.37, 0.42, 0.60, 0.50])

# One slider per parameter, using the same ranges as the practical
sliders = {}
for i, name in enumerate(slider_names):
    start, lo, hi, step = PARAM_SLIDERS[name]
    ax_slider = fig.add_axes([0.15, 0.30 - i * 0.045, 0.70, 0.03])
    sliders[name] = Slider(ax_slider, name, lo, hi, valinit=start, valstep=step)

start_response = np.asarray(prf_model(stimulus, to_frame(**START_VALUES)))[0]


def update(_=None):
    params = {name: s.val for name, s in sliders.items()}
    response = np.asarray(prf_model(stimulus, to_frame(**params)))[0]

    ax_field.clear()
    draw_prf_location(ax_field, params["mu_x"], params["mu_y"], params["sigma"])
    ax_field.set_title("pRF in the visual field")

    ax_resp.clear()
    ax_resp.plot(time_frames, start_response, color=GRAY, linestyle="--", linewidth=1.5, label="Starting parameters")
    ax_resp.plot(time_frames, response, color=BLUE, linewidth=2, label="Current parameters")
    ax_resp.set_title("Predicted BOLD response", loc="left")
    ax_resp.set_xlabel("Time frame")
    ax_resp.legend(loc="upper right")
    fig.canvas.draw_idle()


for s in sliders.values():
    s.on_changed(update)

update()
plt.show()

# Fitting the model to data

response_raw = dataset.response
response_raw.shape 
# Exercise 13
response_raw = np.asarray(dataset.response)
print("Shape:", response_raw.shape)
print("Overall mean:", response_raw.mean())

# Voxels with the largest fluctuations (likely responsive) and one random voxel
voxel_sd = response_raw.std(axis=1)
top_voxels = np.argsort(voxel_sd)[-3:]                     # 3 most variable voxels
random_voxel = np.random.default_rng(0).integers(len(response_raw))
voxels = [60686, *top_voxels, random_voxel]

fig, axes = plt.subplots(len(voxels), 1, figsize=(12, 2.2 * len(voxels)), sharex=True, layout="constrained")
for ax, v in zip(axes, voxels):
    ax.plot(time_frames, response_raw[v], color=BLUE, linewidth=1.5)
    ax.axhline(100, color=GRAY, linestyle="--", linewidth=1)
    ax.set_title(f"Voxel {v}  (SD = {voxel_sd[v]:.2f})", loc="left", fontsize=10)
    ax.set_ylabel("BOLD")
axes[-1].set_xlabel("Time frame (0.9 s each)")
plt.show()

response_no_baseline = response_raw - 100.0

from prfmodel.plotting import plot_response_heatmap

# Uses matplotlib because plotly cannot handle this many voxels
plot_response_heatmap(
    response_no_baseline,
    vmin=-2,
    vmax=5,
    xlabel="Time frame (in TR)",
    ylabel="Voxel index",
    colorbar_label="BOLD response",
    figsize=(6, 6),
);

plt.show()

plot_response_heatmap(response_no_baseline[60000:70000], vmin=-2, vmax=5,
                      xlabel="Time frame (in TR)", ylabel="Voxel index", colorbar_label="BOLD response", figsize=(6, 6))
plt.show()

# Exercise 14 (patterns): “Most voxels show noise around 0 without clear structure. Groups of voxels show increased responses (orange/yellow) clustered in time windows that match the periods when the bar was presented (approximately frames 30–145 and 190–310), with lower activity during the start, end and the middle period without stimulation (around frames 150–190). Some voxels show constant extreme values across the whole run, which are likely artefacts.”

# Exercise 15 (why voxels differ): “Voxels differ because only some lie in visual cortex and respond to the bar. Among visual voxels, the timing of responses differs with their pRF location, since the bar reaches different parts of the visual field at different times, and the strength and width differ with pRF size. In addition, non-neural factors such as distance to blood vessels and to the receiver coil, partial grey-matter coverage, noise and artefacts change the amplitude and reliability of each voxel’s signal.”

from plotting_helpers import make_surface_projection

project_to_surface = make_surface_projection(dataset)

from plotting_helpers import make_surface_plot

plot_surf_stat_map_helper = make_surface_plot(dataset.mesh)

# Calculate standard deviation of each timecourse
response_sd = response_no_baseline.std(axis=1)

plot_surf_stat_map_helper(project_to_surface(response_sd), vmax=1.5, title="BOLD response SD (vmax = 1.5)")
plt.show()

# Exercise 17: average response on the surface

response_mean = response_no_baseline.mean(axis=1)     # one mean per voxel
plot_surf_stat_map_helper(project_to_surface(response_mean), vmax=1.0, title="BOLD response mean")
plt.show()

# bar_on = stimulus.design.reshape(len(time_frames), -1).any(axis=1)     # True when the bar is visible
# contrast = response_no_baseline[:, bar_on].mean(axis=1) - response_no_baseline[:, ~bar_on].mean(axis=1)
# plot_surf_stat_map_helper(project_to_surface(contrast), vmax=1.0, title="Bar on minus bar off")
# plt.show()

# That’s the main lesson of this exercise: high SD can come from stimulus-driven activity or from artefacts, and the SD alone can’t tell them apart.