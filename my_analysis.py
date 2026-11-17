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

# print(stimulus)

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
        "mu_x": [0.0],
        "mu_y": [0.0],
        "sigma": [1.0],
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