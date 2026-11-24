# # import os

# # import numpy as np
# # import pandas as pd

# # # Set keras backend to 'tensorflow' (this is normally the default)
# # os.environ["KERAS_BACKEND"] = "tensorflow"
# # # Print parameter DataFrames with three decimals
# # pd.set_option("display.precision", 3)

# # from prfmodel.examples import load_dataset

# # # Downloads on first use and caches in a user data directory (see prfmodel.examples.get_data_dir)
# # dataset = load_dataset("7t-aot-visual", surface_type="inflated")

# # stimulus = dataset.stimulus

# # from prfmodel.stimuli import PRFStimulus

# # # Cells that the bar covers in at least one time frame lie on the screen
# # is_on_screen = stimulus.design.max(axis=0) > 0.0

# # # The screen is a rectangle, so we keep the rows (y) and columns (x) of the design matrix between its edges
# # rows_on_screen = np.flatnonzero(is_on_screen.any(axis=1))
# # cols_on_screen = np.flatnonzero(is_on_screen.any(axis=0))
# # rows = slice(rows_on_screen[0], rows_on_screen[-1] + 1)
# # cols = slice(cols_on_screen[0], cols_on_screen[-1] + 1)

# # stimulus = PRFStimulus(
# #     design=stimulus.design[:, rows, cols],
# #     grid=stimulus.grid[rows, cols],
# #     frame_duration=stimulus.frame_duration,
# #     dimension_labels=stimulus.dimension_labels,
# # )

# # print(stimulus)

# # # The grid stores y in grid[..., 0] and x in grid[..., 1]
# # (y_min, x_min), (y_max, x_max) = stimulus.grid.min(axis=(0, 1)), stimulus.grid.max(axis=(0, 1))
# # print(f"Screen: x from {x_min:.2f} to {x_max:.2f} degrees, y from {y_min:.2f} to {y_max:.2f} degrees")

# # response_no_baseline = dataset.response - 100.0

# # # Voxels with a response that varies over time
# # response_is_valid = response_no_baseline.std(axis=1) > 0.0

# # response_valid = response_no_baseline[response_is_valid]


# # def fill_valid_voxels(values: np.ndarray) -> np.ndarray:
# #     """Fill values for the valid voxels into an array with one value per voxel in the mask."""
# #     values_full = np.full((response_no_baseline.shape[0],), fill_value=np.nan)
# #     values_full[response_is_valid] = values
# #     return values_full


# # response_valid.shape

# # # std(axis=1) computes each voxel's standard deviation over its 340 time points.
# # # SD = 0 means the time course is a flat line (the same value at every scan).
# # # Only voxels with SD > 0 are kept.

# # from plotting_helpers import make_surface_plot, make_surface_projection

# # project_to_surface = make_surface_projection(dataset)
# # plot_surf_stat_map_helper = make_surface_plot(dataset.mesh)

# # # Splitting the run into two halves
# # import matplotlib.pyplot as plt

# # num_frames = stimulus.design.shape[0]

# # # A frame shows the bar if any pixel of the design matrix is non-zero
# # is_bar_visible = stimulus.design.reshape(num_frames, -1).max(axis=1) > 0

# # split = 168  # Middle of the blank period between frames 151 and 185

# # fig, ax = plt.subplots(figsize=(10, 2), layout="constrained")
# # ax.step(np.arange(num_frames), is_bar_visible, where="mid")
# # ax.axvline(split, color="black", linestyle="--", label="Split")
# # ax.set(xlabel="Time frame", yticks=[0, 1], yticklabels=["Blank", "Bar"])
# # ax.legend(loc="upper right");

# # # ------------------------------------------------------------------
# # # Exercise: video of the stimulus + check the bar sweeps in the table
# # # ------------------------------------------------------------------
# # from prfmodel.plotting import animate_2d_prf_stimulus

# # # 1. Video (saved as a GIF, since HTML video needs a notebook)
# # ani = animate_2d_prf_stimulus(stimulus, interval=25)
# # ani.save("practical3_stimulus.gif", writer="pillow", fps=10)
# # print("Saved practical3_stimulus.gif")

# # # 2. Find the sweeps: consecutive frames in which the bar is visible
# # design = np.asarray(stimulus.design)          # (frames, y, x)
# # grid = np.asarray(stimulus.grid)              # (y, x, 2): [..., 0] = y, [..., 1] = x
# # num_frames = design.shape[0]
# # is_bar_visible = design.reshape(num_frames, -1).max(axis=1) > 0

# # starts = np.flatnonzero(is_bar_visible & ~np.r_[False, is_bar_visible[:-1]])
# # ends = np.flatnonzero(is_bar_visible & ~np.r_[is_bar_visible[1:], False])

# # rows = []
# # for s, e in zip(starts, ends):
# #     cells_s, cells_e = design[s] > 0, design[e] > 0
# #     xs, ys = grid[..., 1][design[(s + e) // 2] > 0], grid[..., 0][design[(s + e) // 2] > 0]
# #     bar = "horizontal" if np.ptp(xs) > np.ptp(ys) else "vertical"
# #     if bar == "horizontal":            # a horizontal bar moves up or down
# #         moved = grid[..., 0][cells_e].mean() - grid[..., 0][cells_s].mean()
# #         direction = "bottom to top" if moved > 0 else "top to bottom"
# #     else:                              # a vertical bar moves left or right
# #         moved = grid[..., 1][cells_e].mean() - grid[..., 1][cells_s].mean()
# #         direction = "left to right" if moved > 0 else "right to left"
# #     rows.append({"frames": f"{s}-{e}", "half": "first" if e < 168 else "second",
# #                  "bar": bar, "direction": direction})

# # sweeps = pd.DataFrame(rows, index=pd.RangeIndex(1, len(rows) + 1, name="sweep"))
# # print(f"\nNumber of sweeps found: {len(sweeps)}")
# # print(sweeps.to_string())

# # # 3. One frame from the middle of each sweep, for a visual check
# # extent = (grid[..., 1].min(), grid[..., 1].max(), grid[..., 0].min(), grid[..., 0].max())
# # fig, axes = plt.subplots(2, 4, figsize=(12, 5), layout="constrained")
# # for ax, (s, e), (_, row) in zip(axes.ravel(), zip(starts, ends), sweeps.iterrows()):
# #     ax.imshow(design[(s + e) // 2], extent=extent, origin="lower", cmap="Greys", vmin=0, vmax=1)
# #     ax.set_title(f"{row['frames']}: {row['bar']}\n{row['direction']}", fontsize=9)
# #     ax.set_xticks([])
# #     ax.set_yticks([])
# # plt.show()

# # first_half = slice(None, split)  # Frames 0 to 167
# # second_half = slice(split, None)  # Frames 168 to 339

# # stimulus_first = PRFStimulus(
# #     design=stimulus.design[first_half],
# #     grid=stimulus.grid,
# #     frame_duration=stimulus.frame_duration,
# #     dimension_labels=["y", "x"],
# # )
# # stimulus_second = PRFStimulus(
# #     design=stimulus.design[second_half],
# #     grid=stimulus.grid,
# #     frame_duration=stimulus.frame_duration,
# #     dimension_labels=["y", "x"],
# # )

# # stimulus_first.design.shape, stimulus_second.design.shape

# # # Selecting voxels with a high noise ceiling

# # # ------------------------------------------------------------------
# # # Noise ceiling from the repeated bar sweeps in both halves
# # # ------------------------------------------------------------------
# # from prfmodel.plotting import plot_r_squared_hist

# # sweeps = pd.DataFrame(
# #     {
# #         "onset_first": [30, 53, 101, 124],
# #         "onset_second": [287, 258, 216, 186],
# #         "num_frames": [20, 27, 20, 27],
# #     },
# #     index=[
# #         "Horizontal bar, top to bottom",
# #         "Vertical bar, left to right",
# #         "Horizontal bar, bottom to top",
# #         "Vertical bar, right to left",
# #     ],
# # )

# # lag = 5  # delay of the BOLD response in frames (5 x 0.9 s = 4.5 s)


# # def cut_sweeps(onsets: pd.Series) -> np.ndarray:
# #     """Concatenate the responses of all valid voxels to the bar sweeps, shifted by the delay of the BOLD response."""
# #     return np.concatenate(
# #         [response_valid[:, onset + lag:onset + lag + n] for onset, n in zip(onsets, sweeps["num_frames"])],
# #         axis=1,
# #     )


# # sweep_response_first = cut_sweeps(sweeps["onset_first"])
# # sweep_response_second = cut_sweeps(sweeps["onset_second"])
# # print("Sweep responses:", sweep_response_first.shape)        # (valid voxels, 94)


# # def correlate_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
# #     """Pearson correlation between each row of x and the same row of y."""
# #     x = x - x.mean(axis=1, keepdims=True)
# #     y = y - y.mean(axis=1, keepdims=True)
# #     return (x * y).sum(axis=1) / np.sqrt((x**2).sum(axis=1) * (y**2).sum(axis=1))


# # with np.errstate(invalid="ignore"):          # constant voxels give NaN instead of a warning
# #     noise_ceiling = correlate_rows(sweep_response_first, sweep_response_second)
# # print("Noise ceiling:", noise_ceiling.shape)

# # plot_r_squared_hist(noise_ceiling[np.isfinite(noise_ceiling)], value_range=(-1.0, 1.0), log=True,
# #                     xlabel="Noise ceiling")
# # plt.show()

# # plot_surf_stat_map_helper(project_to_surface(fill_valid_voxels(noise_ceiling)), vmin=0.0, vmax=1.0,
# #                           title="Noise ceiling")
# # plt.show()

# # # Exercise 5: how many voxels for different thresholds?
# # with np.errstate(invalid="ignore"):            # NaNs compare as False
# #     for threshold in [0.5, 0.7, 0.9]:
# #         n = np.sum(noise_ceiling > threshold)
# #         print(f"Noise ceiling > {threshold}: {n} voxels "
# #               f"({100 * n / len(noise_ceiling):.1f}% of valid voxels)")

# # # Where are they on the brain?
# # for threshold in [0.5, 0.9]:
# #     with np.errstate(invalid="ignore"):
# #         selected = noise_ceiling > threshold
# #     values = np.where(selected, noise_ceiling, np.nan)
# #     plot_surf_stat_map_helper(
# #         project_to_surface(fill_valid_voxels(values)),
# #         vmin=0.0, vmax=1.0,
# #         title=f"Voxels with noise ceiling > {threshold}",
# #     )
# #     plt.show()

# #     # Candidate models

# #     from keras import ops
# #     from keras.optimizers import Adam
# #     from prfmodel.fitters import GridFitter, LeastSquaresFitter, SGDFitter, grid_values_2d_prf
# #     from prfmodel.fitters.adapter import Adapter, ParameterTransform
# #     from prfmodel.impulse import DerivativeTwoGammaImpulse

# #     # Same SGD settings for all models
# #     num_steps = 500
# #     learning_rate = 0.01

# #     # Define repetition time (TR)
# #     tr = 0.9

# #     # Create custom impulse model that is shared by all models
# #     impulse_model = DerivativeTwoGammaImpulse(resolution=tr)

# #     # @title Helper functions for the interactive figures { display-mode: "form" }
# #     from plotting_helpers import BLUE, GRAY, add_legend, make_slider, show_interactive, style_axes, to_frame
# #     from prfmodel.models.prf import Gaussian2DPRFModel

# #     # All figures show a pRF in the center of the visual field
# #     CENTER_PARAMS = {"mu_x": 0.0, "mu_y": 0.0, "weight_deriv": -0.5, "baseline": 0.0}
# #     GAUSSIAN_PARAMS = {"sigma": 1.0, "amplitude": 1.0}

# #     # Stimulus with one vertical bar per frame; the bar moves from left to right over frames
# #     bar_width = 1.7  # Width of the bar in the experiment (degrees visual angle)
# #     bar_positions = np.linspace(-4.0, 4.0, 81)
# #     bar_stimulus = PRFStimulus(
# #         design=(np.abs(stimulus.grid[..., 1] - bar_positions[:, None, None]) <= bar_width / 2).astype("float32"),
# #         grid=stimulus.grid,
# #         frame_duration=stimulus.frame_duration,
# #         dimension_labels=["y", "x"],
# #     )

# #     time_frames_first = np.arange(stimulus_first.design.shape[0])


# #     def scale_to_max(values: np.ndarray) -> np.ndarray:
# #         """Scale values so that their largest absolute value is 1."""
# #         values = np.asarray(values)
# #         max_value = np.abs(values).max()
# #         return values / max_value if max_value > 0 else values


# #     def plot_model_comparison(model_class: type, model_name: str, **params: float) -> None:
# #         """Plot the bar response and the predicted BOLD response of a model next to those of the Gaussian model."""
# #         fig, (ax_bar, ax_response) = plt.subplots(
# #             1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 3]}, layout="constrained",
# #         )

# #         for current_class, current_name, current_params, style in [
# #             (Gaussian2DPRFModel, "Gaussian", GAUSSIAN_PARAMS, {"color": GRAY, "linestyle": "--", "linewidth": 1.5}),
# #             (model_class, model_name, params, {"color": BLUE, "linewidth": 2}),
# #         ]:
# #             param_frame = to_frame(**CENTER_PARAMS, **current_params)
# #             # Without an impulse response, every frame of the bar stimulus gives the response to the bar at one position
# #             bar_response = current_class(impulse_model=None)(bar_stimulus, param_frame)[0]
# #             bold_response = current_class(impulse_model=impulse_model)(stimulus_first, param_frame)[0]
# #             ax_bar.plot(bar_positions, scale_to_max(bar_response), label=current_name, **style)
# #             ax_response.plot(time_frames_first, scale_to_max(bold_response), label=current_name, **style)

# #         ax_bar.set_title("Response to the bar", loc="left")
# #         style_axes(ax_bar, "Bar position (degrees visual angle)", "Scaled response")

# #         add_legend(ax_response)
# #         ax_response.set_title("Predicted BOLD response", loc="left")
# #         style_axes(ax_response, "Time frame", "Scaled BOLD response")

# #         plt.show()


# #     from prfmodel.models.prf import (Gaussian2DCSSPRFModel, DoG2DPRFModel,
# #                                 DivNormGaussian2DPRFModel)

# #     # CSS: compare different compression exponents
# #     for n in [1.0, 0.5, 0.2]:
# #         plot_model_comparison(Gaussian2DCSSPRFModel, f"CSS (n={n})", **GAUSSIAN_PARAMS, n=n)

# #    # DoG: add a suppressive surround
# #     plot_model_comparison(DoG2DPRFModel, "DoG",
# #                       sigma_center=1.0, amplitude_center=1.0,
# #                       sigma_surround=2.0, amplitude_surround=0.3)

# #     # DN: divisive normalization
# #     plot_model_comparison(DivNormGaussian2DPRFModel, "DN",
# #                           sigma_activation=1.0, amplitude_activation=1.0,
# #                           sigma_normalization=3.0, amplitude_normalization=1.0,
# #                           baseline_activation=1.0, baseline_normalization=1.0)

# #     # Fitting the models to the first half

# #     # Guassian model
# #     from prfmodel.models.prf import Gaussian2DPRFModel

# #     gaussian_model = Gaussian2DPRFModel(impulse_model=impulse_model)

# #     grid_param_ranges = grid_values_2d_prf(
# #         stimulus_first,
# #         num_mu=11,
# #         mu_extent=1.0,  # Only span the visual field covered by the stimulus
# #         num_sigma=10,
# #         sigma_range=(0.05, 4.0),  # Log-spaced by default
# #     ) | {  # Add constant parameters
# #         "weight_deriv": [-0.5],
# #         "baseline": [0.0],
# #         "amplitude": [1.0],
# #     }

# #     grid_fitter = GridFitter(
# #         model=gaussian_model,
# #         stimulus=stimulus_first,
# #         compile_step=True,  # Setting 'compile_step=True' speeds up the fitting
# #     )

# #     grid_history, grid_params = grid_fitter.fit(
# #         data=response_first,
# #         parameter_values=grid_param_ranges,
# #         batch_size=20,
# #     )

# #     ls_fitter = LeastSquaresFitter(
# #         model=gaussian_model,
# #         stimulus=stimulus_first,
# #     )

# #     ls_history, ls_params = ls_fitter.fit(
# #         data=response_first,
# #         parameters=grid_params,
# #         slope_name="amplitude",
# #         intercept_name="baseline",
# #         batch_size=200,
# #     )
# # gaussian_sgd_fitter = SGDFitter(
# #     model=gaussian_model,
# #     stimulus=stimulus_first,
# #     adapter=Adapter([ParameterTransform(["sigma"], transform_fun=ops.log, inverse_fun=ops.exp)]),
# #     optimizer=Adam(learning_rate=learning_rate),
# #     compile_step=True,
# # )

# # gaussian_history, gaussian_params = gaussian_sgd_fitter.fit(
# #     data=response_first,
# #     init_parameters=ls_params,
# #     fixed_parameters=["weight_deriv"],
# #     num_steps=num_steps,
# # )

# # gaussian_params

# # # CSS MODEL

# # # @title Interactive figure: CSS model { display-mode: "form" }
# # from prfmodel.models.prf import Gaussian2DCSSPRFModel


# # def plot_css(n: float) -> None:
# #     plot_model_comparison(Gaussian2DCSSPRFModel, "CSS", sigma=1.0, n=n, amplitude=1.0)


# # show_interactive(plot_css, {"n": make_slider("n", 0.5, 0.05, 1.5, 0.05)})

# # from prfmodel.models.prf import Gaussian2DCSSPRFModel, init_css_from_gaussian

# # css_model = Gaussian2DCSSPRFModel(impulse_model=impulse_model)

# # # Start from the Gaussian estimates with n = 0.5
# # css_init_params = init_css_from_gaussian(gaussian_params)

# # _, css_init_params = LeastSquaresFitter(model=css_model, stimulus=stimulus_first).fit(
# #     data=response_first,
# #     parameters=css_init_params,
# #     slope_name="amplitude",
# #     intercept_name="baseline",
# #     batch_size=200,
# # )

# # css_sgd_fitter = SGDFitter(
# #     model=css_model,
# #     stimulus=stimulus_first,
# #     adapter=Adapter([ParameterTransform(["sigma", "n"], transform_fun=ops.log, inverse_fun=ops.exp)]),
# #     optimizer=Adam(learning_rate=learning_rate),
# #     compile_step=True,
# # )

# # css_history, css_params = css_sgd_fitter.fit(
# #     data=response_first,
# #     init_parameters=css_init_params,
# #     fixed_parameters=["weight_deriv"],
# #     num_steps=num_steps,
# # )

# # css_params

# # # DOG model

# # # @title Interactive figure: DoG model { display-mode: "form" }
# # from prfmodel.models.prf import DoG2DPRFModel


# # def plot_dog(sigma_surround: float, amplitude_surround: float) -> None:
# #     plot_model_comparison(
# #         DoG2DPRFModel,
# #         "DoG",
# #         sigma_center=1.0,
# #         sigma_surround=sigma_surround,
# #         amplitude_center=1.0,
# #         amplitude_surround=amplitude_surround,
# #     )


# # show_interactive(
# #     plot_dog,
# #     {
# #         "sigma_surround": make_slider("sigma_surround", 2.0, 1.0, 6.0, 0.1),
# #         "amplitude_surround": make_slider("amplitude_surround", 0.5, 0.0, 1.0, 0.05),
# #     },
# # )

# # from prfmodel.models.prf import DoG2DPRFModel, init_dog_from_gaussian

# # dog_model = DoG2DPRFModel(impulse_model=impulse_model)

# # # Start from the Gaussian estimates with a weak surround that is three times as large as the center
# # dog_init_params = init_dog_from_gaussian(gaussian_params, sigma_ratio=3.0)

# # _, dog_ls_params = LeastSquaresFitter(model=dog_model, stimulus=stimulus_first).fit(
# #     data=response_first,
# #     parameters=dog_init_params,
# #     slope_name=["amplitude_center", "amplitude_surround"],
# #     intercept_name="baseline",
# #     batch_size=200,
# # )

# # # Only use the least-squares estimates for voxels with a suppressive surround
# # is_suppressive = dog_ls_params["amplitude_surround"] > 0.0
# # dog_init_params.loc[is_suppressive] = dog_ls_params.loc[is_suppressive]

# # is_suppressive.mean()

# # dog_sgd_fitter = SGDFitter(
# #     model=dog_model,
# #     stimulus=stimulus_first,
# #     adapter=Adapter([
# #         ParameterTransform(
# #             ["sigma_center", "sigma_surround", "amplitude_surround"], transform_fun=ops.log, inverse_fun=ops.exp,
# #         ),
# #     ]),
# #     optimizer=Adam(learning_rate=learning_rate),
# #     compile_step=True,
# # )

# # dog_history, dog_params = dog_sgd_fitter.fit(
# #     data=response_first,
# #     init_parameters=dog_init_params,
# #     fixed_parameters=["weight_deriv"],
# #     num_steps=num_steps,
# # )

# # dog_params

# # # DN model
# # # @title Interactive figure: DN model { display-mode: "form" }
# # from prfmodel.models.prf import DivNormGaussian2DPRFModel


# # def plot_dn(sigma_normalization: float, baseline_activation: float, baseline_normalization: float) -> None:
# #     plot_model_comparison(
# #         DivNormGaussian2DPRFModel,
# #         "DN",
# #         sigma_activation=1.0,
# #         sigma_normalization=sigma_normalization,
# #         amplitude_activation=1.0,
# #         amplitude_normalization=1.0,
# #         baseline_activation=baseline_activation,
# #         baseline_normalization=baseline_normalization,
# #     )


# # show_interactive(
# #     plot_dn,
# #     {
# #         "sigma_normalization": make_slider("sigma_normalization", 2.0, 1.0, 6.0, 0.1),
# #         "baseline_activation": make_slider("baseline_activation", 1.0, 0.01, 100.0, 0.1, log=True),
# #         "baseline_normalization": make_slider("baseline_normalization", 1.0, 0.01, 100.0, 0.1, log=True),
# #     },
# # )

# import os
# import pickle
# from pathlib import Path

# import numpy as np
# import pandas as pd

# # Set keras backend to 'tensorflow' (this is normally the default)
# os.environ["KERAS_BACKEND"] = "tensorflow"
# # Print parameter DataFrames with three decimals
# pd.set_option("display.precision", 3)

# import matplotlib.pyplot as plt
# from prfmodel.examples import load_dataset
# from prfmodel.stimuli import PRFStimulus

# RESULTS = Path("results")
# RESULTS.mkdir(exist_ok=True)

# # ------------------------------------------------------------------
# # Data
# # ------------------------------------------------------------------
# # Downloads on first use and caches in a user data directory (see prfmodel.examples.get_data_dir)
# dataset = load_dataset("7t-aot-visual", surface_type="inflated")

# stimulus = dataset.stimulus

# # Cells that the bar covers in at least one time frame lie on the screen
# is_on_screen = stimulus.design.max(axis=0) > 0.0

# # The screen is a rectangle, so we keep the rows (y) and columns (x) of the design matrix between its edges
# rows_on_screen = np.flatnonzero(is_on_screen.any(axis=1))
# cols_on_screen = np.flatnonzero(is_on_screen.any(axis=0))
# rows = slice(rows_on_screen[0], rows_on_screen[-1] + 1)
# cols = slice(cols_on_screen[0], cols_on_screen[-1] + 1)

# stimulus = PRFStimulus(
#     design=stimulus.design[:, rows, cols],
#     grid=stimulus.grid[rows, cols],
#     frame_duration=stimulus.frame_duration,
#     dimension_labels=stimulus.dimension_labels,
# )

# print(stimulus)

# # The grid stores y in grid[..., 0] and x in grid[..., 1]
# (y_min, x_min), (y_max, x_max) = stimulus.grid.min(axis=(0, 1)), stimulus.grid.max(axis=(0, 1))
# print(f"Screen: x from {x_min:.2f} to {x_max:.2f} degrees, y from {y_min:.2f} to {y_max:.2f} degrees")

# response_no_baseline = dataset.response - 100.0

# # Voxels with a response that varies over time (SD = 0 means a flat line)
# response_is_valid = response_no_baseline.std(axis=1) > 0.0
# response_valid = response_no_baseline[response_is_valid]
# print("Valid voxels:", response_valid.shape)


# def fill_valid_voxels(values: np.ndarray) -> np.ndarray:
#     """Fill values for the valid voxels into an array with one value per voxel in the mask."""
#     values_full = np.full((response_no_baseline.shape[0],), fill_value=np.nan)
#     values_full[response_is_valid] = values
#     return values_full


# from plotting_helpers import make_surface_plot, make_surface_projection

# project_to_surface = make_surface_projection(dataset)
# plot_surf_stat_map_helper = make_surface_plot(dataset.mesh)

# # ------------------------------------------------------------------
# # Splitting the run into two halves
# # ------------------------------------------------------------------
# num_frames = stimulus.design.shape[0]

# # A frame shows the bar if any pixel of the design matrix is non-zero
# is_bar_visible = stimulus.design.reshape(num_frames, -1).max(axis=1) > 0

# split = 168  # Middle of the blank period between frames 151 and 185

# fig, ax = plt.subplots(figsize=(10, 2), layout="constrained")
# ax.step(np.arange(num_frames), is_bar_visible, where="mid")
# ax.axvline(split, color="black", linestyle="--", label="Split")
# ax.set(xlabel="Time frame", yticks=[0, 1], yticklabels=["Blank", "Bar"])
# ax.legend(loc="upper right")
# plt.show()

# # ------------------------------------------------------------------
# # Exercise: video of the stimulus + check the bar sweeps in the table
# # ------------------------------------------------------------------
# from prfmodel.plotting import animate_2d_prf_stimulus

# # 1. Video (saved as a GIF, since HTML video needs a notebook)
# if not Path("practical3_stimulus.gif").exists():
#     ani = animate_2d_prf_stimulus(stimulus, interval=25)
#     ani.save("practical3_stimulus.gif", writer="pillow", fps=10)
#     print("Saved practical3_stimulus.gif")

# # 2. Find the sweeps: consecutive frames in which the bar is visible
# design = np.asarray(stimulus.design)          # (frames, y, x)
# grid = np.asarray(stimulus.grid)              # (y, x, 2): [..., 0] = y, [..., 1] = x

# starts = np.flatnonzero(is_bar_visible & ~np.r_[False, is_bar_visible[:-1]])
# ends = np.flatnonzero(is_bar_visible & ~np.r_[is_bar_visible[1:], False])

# sweep_rows = []
# for s, e in zip(starts, ends):
#     cells_s, cells_e = design[s] > 0, design[e] > 0
#     xs, ys = grid[..., 1][design[(s + e) // 2] > 0], grid[..., 0][design[(s + e) // 2] > 0]
#     bar = "horizontal" if np.ptp(xs) > np.ptp(ys) else "vertical"
#     if bar == "horizontal":            # a horizontal bar moves up or down
#         moved = grid[..., 0][cells_e].mean() - grid[..., 0][cells_s].mean()
#         direction = "bottom to top" if moved > 0 else "top to bottom"
#     else:                              # a vertical bar moves left or right
#         moved = grid[..., 1][cells_e].mean() - grid[..., 1][cells_s].mean()
#         direction = "left to right" if moved > 0 else "right to left"
#     sweep_rows.append({"frames": f"{s}-{e}", "half": "first" if e < split else "second",
#                        "bar": bar, "direction": direction})

# sweep_table = pd.DataFrame(sweep_rows, index=pd.RangeIndex(1, len(sweep_rows) + 1, name="sweep"))
# print(f"\nNumber of sweeps found: {len(sweep_table)}")
# print(sweep_table.to_string())

# # 3. One frame from the middle of each sweep, for a visual check
# extent = (grid[..., 1].min(), grid[..., 1].max(), grid[..., 0].min(), grid[..., 0].max())
# fig, axes = plt.subplots(2, 4, figsize=(12, 5), layout="constrained")
# for ax, (s, e), (_, row) in zip(axes.ravel(), zip(starts, ends), sweep_table.iterrows()):
#     ax.imshow(design[(s + e) // 2], extent=extent, origin="lower", cmap="Greys", vmin=0, vmax=1)
#     ax.set_title(f"{row['frames']}: {row['bar']}\n{row['direction']}", fontsize=9)
#     ax.set_xticks([])
#     ax.set_yticks([])
# plt.show()

# first_half = slice(None, split)   # Frames 0 to 167
# second_half = slice(split, None)  # Frames 168 to the end

# stimulus_first = PRFStimulus(
#     design=stimulus.design[first_half],
#     grid=stimulus.grid,
#     frame_duration=stimulus.frame_duration,
#     dimension_labels=["y", "x"],
# )
# stimulus_second = PRFStimulus(
#     design=stimulus.design[second_half],
#     grid=stimulus.grid,
#     frame_duration=stimulus.frame_duration,
#     dimension_labels=["y", "x"],
# )
# print("Stimulus halves:", stimulus_first.design.shape, stimulus_second.design.shape)

# # ------------------------------------------------------------------
# # Noise ceiling from the repeated bar sweeps in both halves
# # ------------------------------------------------------------------
# from prfmodel.plotting import plot_r_squared_hist

# sweeps = pd.DataFrame(
#     {
#         "onset_first": [30, 53, 101, 124],
#         "onset_second": [287, 258, 216, 186],
#         "num_frames": [20, 27, 20, 27],
#     },
#     index=[
#         "Horizontal bar, top to bottom",
#         "Vertical bar, left to right",
#         "Horizontal bar, bottom to top",
#         "Vertical bar, right to left",
#     ],
# )

# lag = 5  # delay of the BOLD response in frames (5 x 0.9 s = 4.5 s)


# def cut_sweeps(onsets: pd.Series) -> np.ndarray:
#     """Concatenate the responses of all valid voxels to the bar sweeps, shifted by the delay of the BOLD response."""
#     return np.concatenate(
#         [response_valid[:, onset + lag:onset + lag + n] for onset, n in zip(onsets, sweeps["num_frames"])],
#         axis=1,
#     )


# sweep_response_first = cut_sweeps(sweeps["onset_first"])
# sweep_response_second = cut_sweeps(sweeps["onset_second"])
# print("Sweep responses:", sweep_response_first.shape)        # (valid voxels, 94)


# def correlate_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
#     """Pearson correlation between each row of x and the same row of y."""
#     x = x - x.mean(axis=1, keepdims=True)
#     y = y - y.mean(axis=1, keepdims=True)
#     return (x * y).sum(axis=1) / np.sqrt((x**2).sum(axis=1) * (y**2).sum(axis=1))


# with np.errstate(invalid="ignore"):          # constant voxels give NaN instead of a warning
#     noise_ceiling = correlate_rows(sweep_response_first, sweep_response_second)
# print("Noise ceiling:", noise_ceiling.shape)

# plot_r_squared_hist(noise_ceiling[np.isfinite(noise_ceiling)], value_range=(-1.0, 1.0), log=True,
#                     xlabel="Noise ceiling")
# plt.show()

# plot_surf_stat_map_helper(project_to_surface(fill_valid_voxels(noise_ceiling)), vmin=0.0, vmax=1.0,
#                           title="Noise ceiling")
# plt.show()

# # ------------------------------------------------------------------
# # Selecting voxels with a high noise ceiling
# # ------------------------------------------------------------------
# noise_ceiling_threshold = 0.7

# # NaN values are never above the threshold
# with np.errstate(invalid="ignore"):
#     is_selected = noise_ceiling > noise_ceiling_threshold

# response_first = response_valid[is_selected][:, first_half]
# response_second = response_valid[is_selected][:, second_half]

# noise_ceiling_selected = noise_ceiling[is_selected]


# def fill_selected_voxels(values: np.ndarray) -> np.ndarray:
#     """Fill values for the selected voxels into an array with one value per voxel in the mask, NaN for all others."""
#     values_valid = np.full((response_valid.shape[0],), fill_value=np.nan)
#     values_valid[is_selected] = values
#     return fill_valid_voxels(values_valid)


# print("Selected responses:", response_first.shape, response_second.shape)

# # ------------------------------------------------------------------
# # Exercise 5: how many voxels for different thresholds?
# # ------------------------------------------------------------------
# with np.errstate(invalid="ignore"):            # NaNs compare as False
#     for threshold in [0.5, 0.7, 0.9]:
#         n_vox = np.sum(noise_ceiling > threshold)
#         print(f"Noise ceiling > {threshold}: {n_vox} voxels "
#               f"({100 * n_vox / len(noise_ceiling):.1f}% of valid voxels)")

# # Where are they on the brain?
# for threshold in [0.5, 0.9]:
#     with np.errstate(invalid="ignore"):
#         selected = noise_ceiling > threshold
#     values = np.where(selected, noise_ceiling, np.nan)
#     plot_surf_stat_map_helper(
#         project_to_surface(fill_valid_voxels(values)),
#         vmin=0.0, vmax=1.0,
#         title=f"Voxels with noise ceiling > {threshold}",
#     )
#     plt.show()

# # ------------------------------------------------------------------
# # Candidate models
# # ------------------------------------------------------------------
# from keras import ops
# from keras.optimizers import Adam
# from prfmodel.fitters import GridFitter, LeastSquaresFitter, SGDFitter, grid_values_2d_prf
# from prfmodel.fitters.adapter import Adapter, ParameterTransform
# from prfmodel.impulse import DerivativeTwoGammaImpulse
# from prfmodel.models.prf import (
#     DivNormGaussian2DPRFModel,
#     DoG2DPRFModel,
#     Gaussian2DCSSPRFModel,
#     Gaussian2DPRFModel,
#     init_css_from_gaussian,
#     init_dog_from_gaussian,
# )

# # Same SGD settings for all models
# num_steps = 500
# learning_rate = 0.01

# # Define repetition time (TR)
# tr = 0.9

# # Create custom impulse model that is shared by all models
# impulse_model = DerivativeTwoGammaImpulse(resolution=tr)

# # Helper functions for the comparison figures
# from plotting_helpers import BLUE, GRAY, add_legend, style_axes, to_frame

# # All figures show a pRF in the center of the visual field
# CENTER_PARAMS = {"mu_x": 0.0, "mu_y": 0.0, "weight_deriv": -0.5, "baseline": 0.0}
# GAUSSIAN_PARAMS = {"sigma": 1.0, "amplitude": 1.0}

# # Stimulus with one vertical bar per frame; the bar moves from left to right over frames
# bar_width = 1.7  # Width of the bar in the experiment (degrees visual angle)
# bar_positions = np.linspace(-4.0, 4.0, 81)
# bar_stimulus = PRFStimulus(
#     design=(np.abs(stimulus.grid[..., 1] - bar_positions[:, None, None]) <= bar_width / 2).astype("float32"),
#     grid=stimulus.grid,
#     frame_duration=stimulus.frame_duration,
#     dimension_labels=["y", "x"],
# )

# time_frames_first = np.arange(stimulus_first.design.shape[0])


# def scale_to_max(values: np.ndarray) -> np.ndarray:
#     """Scale values so that their largest absolute value is 1."""
#     values = np.asarray(values)
#     max_value = np.abs(values).max()
#     return values / max_value if max_value > 0 else values


# def plot_model_comparison(model_class: type, model_name: str, **params: float) -> None:
#     """Plot the bar response and the predicted BOLD response of a model next to those of the Gaussian model."""
#     fig, (ax_bar, ax_response) = plt.subplots(
#         1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 3]}, layout="constrained",
#     )

#     for current_class, current_name, current_params, style in [
#         (Gaussian2DPRFModel, "Gaussian", GAUSSIAN_PARAMS, {"color": GRAY, "linestyle": "--", "linewidth": 1.5}),
#         (model_class, model_name, params, {"color": BLUE, "linewidth": 2}),
#     ]:
#         param_frame = to_frame(**CENTER_PARAMS, **current_params)
#         # Without an impulse response, every frame of the bar stimulus gives the response to the bar at one position
#         bar_response = current_class(impulse_model=None)(bar_stimulus, param_frame)[0]
#         bold_response = current_class(impulse_model=impulse_model)(stimulus_first, param_frame)[0]
#         ax_bar.plot(bar_positions, scale_to_max(bar_response), label=current_name, **style)
#         ax_response.plot(time_frames_first, scale_to_max(bold_response), label=current_name, **style)

#     ax_bar.set_title("Response to the bar", loc="left")
#     style_axes(ax_bar, "Bar position (degrees visual angle)", "Scaled response")

#     add_legend(ax_response)
#     ax_response.set_title(f"Predicted BOLD response ({model_name})", loc="left")
#     style_axes(ax_response, "Time frame", "Scaled BOLD response")

#     plt.show()


# def cached_fit(name: str, fit_function) -> pd.DataFrame:
#     """Load fitted parameters from results/ if they exist, otherwise fit and save them.

#     Delete the .pkl files if you change the noise-ceiling threshold or the fitting settings.
#     """
#     path = RESULTS / f"practical3_{name}_params.pkl"
#     if path.exists():
#         print(f"Loading {path}")
#         with open(path, "rb") as f:
#             return pickle.load(f)
#     params = fit_function()
#     with open(path, "wb") as f:
#         pickle.dump(params, f)
#     print(f"Saved {path}")
#     return params


# # ------------------------------------------------------------------
# # Fitting the models to the first half: Gaussian
# # ------------------------------------------------------------------
# gaussian_model = Gaussian2DPRFModel(impulse_model=impulse_model)


# def fit_gaussian() -> pd.DataFrame:
#     grid_param_ranges = grid_values_2d_prf(
#         stimulus_first,
#         num_mu=11,
#         mu_extent=1.0,  # Only span the visual field covered by the stimulus
#         num_sigma=10,
#         sigma_range=(0.05, 4.0),  # Log-spaced by default
#     ) | {  # Add constant parameters
#         "weight_deriv": [-0.5],
#         "baseline": [0.0],
#         "amplitude": [1.0],
#     }

#     grid_fitter = GridFitter(
#         model=gaussian_model,
#         stimulus=stimulus_first,
#         compile_step=True,  # Setting 'compile_step=True' speeds up the fitting
#     )
#     _, grid_params = grid_fitter.fit(
#         data=response_first,
#         parameter_values=grid_param_ranges,
#         batch_size=20,
#     )

#     ls_fitter = LeastSquaresFitter(model=gaussian_model, stimulus=stimulus_first)
#     _, ls_params = ls_fitter.fit(
#         data=response_first,
#         parameters=grid_params,
#         slope_name="amplitude",
#         intercept_name="baseline",
#         batch_size=200,
#     )

#     gaussian_sgd_fitter = SGDFitter(
#         model=gaussian_model,
#         stimulus=stimulus_first,
#         adapter=Adapter([ParameterTransform(["sigma"], transform_fun=ops.log, inverse_fun=ops.exp)]),
#         optimizer=Adam(learning_rate=learning_rate),
#         compile_step=True,
#     )
#     _, params = gaussian_sgd_fitter.fit(
#         data=response_first,
#         init_parameters=ls_params,
#         fixed_parameters=["weight_deriv"],
#         num_steps=num_steps,
#     )
#     return params


# gaussian_params = cached_fit("gaussian", fit_gaussian)
# print("\nGaussian parameters:\n", gaussian_params)

# # ------------------------------------------------------------------
# # CSS model
# # ------------------------------------------------------------------
# # Figure (replaces the interactive slider): the effect of the exponent n
# for n in [1.0, 0.5, 0.2]:
#     plot_model_comparison(Gaussian2DCSSPRFModel, f"CSS (n={n})", sigma=1.0, n=n, amplitude=1.0)

# css_model = Gaussian2DCSSPRFModel(impulse_model=impulse_model)


# def fit_css() -> pd.DataFrame:
#     # Start from the Gaussian estimates with n = 0.5
#     css_init_params = init_css_from_gaussian(gaussian_params)

#     _, css_init_params = LeastSquaresFitter(model=css_model, stimulus=stimulus_first).fit(
#         data=response_first,
#         parameters=css_init_params,
#         slope_name="amplitude",
#         intercept_name="baseline",
#         batch_size=200,
#     )

#     css_sgd_fitter = SGDFitter(
#         model=css_model,
#         stimulus=stimulus_first,
#         adapter=Adapter([ParameterTransform(["sigma", "n"], transform_fun=ops.log, inverse_fun=ops.exp)]),
#         optimizer=Adam(learning_rate=learning_rate),
#         compile_step=True,
#     )
#     _, params = css_sgd_fitter.fit(
#         data=response_first,
#         init_parameters=css_init_params,
#         fixed_parameters=["weight_deriv"],
#         num_steps=num_steps,
#     )
#     return params


# css_params = cached_fit("css", fit_css)
# print("\nCSS parameters:\n", css_params)

# # ------------------------------------------------------------------
# # DoG model
# # ------------------------------------------------------------------
# # Figure (replaces the interactive sliders): surround strength
# for amplitude_surround in [0.0, 0.3, 0.6]:
#     plot_model_comparison(
#         DoG2DPRFModel,
#         f"DoG (amplitude_surround={amplitude_surround})",
#         sigma_center=1.0,
#         sigma_surround=2.0,
#         amplitude_center=1.0,
#         amplitude_surround=amplitude_surround,
#     )

# dog_model = DoG2DPRFModel(impulse_model=impulse_model)


# def fit_dog() -> pd.DataFrame:
#     # Start from the Gaussian estimates with a weak surround that is three times as large as the center
#     dog_init_params = init_dog_from_gaussian(gaussian_params, sigma_ratio=3.0)

#     _, dog_ls_params = LeastSquaresFitter(model=dog_model, stimulus=stimulus_first).fit(
#         data=response_first,
#         parameters=dog_init_params,
#         slope_name=["amplitude_center", "amplitude_surround"],
#         intercept_name="baseline",
#         batch_size=200,
#     )

#     # Only use the least-squares estimates for voxels with a suppressive surround
#     is_suppressive = dog_ls_params["amplitude_surround"] > 0.0
#     dog_init_params.loc[is_suppressive] = dog_ls_params.loc[is_suppressive]
#     print(f"Fraction of voxels with a suppressive surround after least squares: {is_suppressive.mean():.3f}")

#     dog_sgd_fitter = SGDFitter(
#         model=dog_model,
#         stimulus=stimulus_first,
#         adapter=Adapter([
#             ParameterTransform(
#                 ["sigma_center", "sigma_surround", "amplitude_surround"], transform_fun=ops.log, inverse_fun=ops.exp,
#             ),
#         ]),
#         optimizer=Adam(learning_rate=learning_rate),
#         compile_step=True,
#     )
#     _, params = dog_sgd_fitter.fit(
#         data=response_first,
#         init_parameters=dog_init_params,
#         fixed_parameters=["weight_deriv"],
#         num_steps=num_steps,
#     )
#     return params


# dog_params = cached_fit("dog", fit_dog)
# print("\nDoG parameters:\n", dog_params)

# # ------------------------------------------------------------------
# # DN model
# # ------------------------------------------------------------------
# # Figures (replace the interactive sliders): normalization size and the two baselines
# for sigma_normalization, baseline_activation, baseline_normalization in [
#     (2.0, 1.0, 1.0),
#     (2.0, 0.01, 1.0),    # small baseline_activation
#     (2.0, 1.0, 0.01),    # small baseline_normalization
#     (6.0, 1.0, 1.0),     # large normalization pool
# ]:
#     plot_model_comparison(
#         DivNormGaussian2DPRFModel,
#         f"DN (sigma_N={sigma_normalization}, b={baseline_activation}, d={baseline_normalization})",
#         sigma_activation=1.0,
#         sigma_normalization=sigma_normalization,
#         amplitude_activation=1.0,
#         amplitude_normalization=1.0,
#         baseline_activation=baseline_activation,
#         baseline_normalization=baseline_normalization,
#     )


# # ------------------------------------------------------------------
# # Exercise 7: parameter names and the distributions of n and the DoG size ratio
# # ------------------------------------------------------------------
# dn_model = DivNormGaussian2DPRFModel(impulse_model=impulse_model)  # only needed for its names here

# for name, model in [("Gaussian", gaussian_model), ("CSS", css_model),
#                     ("DoG", dog_model), ("DN", dn_model)]:
#     names = list(model.parameter_names)
#     print(f"{name:8s} ({len(names)} names): {names}")

# # --- CSS exponent n ---
# n_values = css_params["n"].to_numpy()
# print(f"\nCSS n: median {np.median(n_values):.2f}, "
#       f"{100 * np.mean(n_values < 1):.1f}% of voxels have n < 1 (compression)")

# # --- DoG surround/center size ratio ---
# size_ratio = (dog_params["sigma_surround"] / dog_params["sigma_center"]).to_numpy()
# strength_ratio = (dog_params["amplitude_surround"] / dog_params["amplitude_center"]).to_numpy()
# print(f"DoG sigma_surround / sigma_center: median {np.median(size_ratio):.2f}")
# print(f"DoG amplitude_surround / amplitude_center: median {np.median(strength_ratio):.3f}")

# fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), layout="constrained")

# axes[0].hist(n_values, bins=40, color=BLUE)
# axes[0].axvline(1.0, color="black", ls="--", label="n = 1 (Gaussian)")
# axes[0].set(xlabel="CSS exponent n", ylabel="Number of voxels", title="Compression")
# axes[0].legend()

# axes[1].hist(size_ratio, bins=np.logspace(0, np.log10(size_ratio.max()), 40), color=BLUE)
# axes[1].set_xscale("log")
# axes[1].axvline(3.0, color="black", ls="--", label="starting value (3)")
# axes[1].set(xlabel="sigma_surround / sigma_center", title="Surround size")
# axes[1].legend()

# axes[2].hist(strength_ratio, bins=40, color=BLUE)
# axes[2].set(xlabel="amplitude_surround / amplitude_center", title="Surround strength")
# plt.show()

# # Where on the brain? (selected voxels only)
# plot_surf_stat_map_helper(project_to_surface(fill_selected_voxels(n_values)),
#                           vmin=0.0, vmax=1.0, title="CSS exponent n")
# plt.show()
# plot_surf_stat_map_helper(project_to_surface(fill_selected_voxels(np.log10(size_ratio))),
#                           title="log10(sigma_surround / sigma_center)")
# plt.show()

# print(r_squared_train.agg(["mean", "median"]))
# # ... the plot_r_squared_comparison loop ...
# plt.show()

# # ------------------------------------------------------------------
# # Exercise 8: in-sample comparison against the Gaussian model
# # ------------------------------------------------------------------
# print("In-sample R² (training set, first half):")
# print(r_squared_train.agg(["mean", "median"]))

# diff_train = r_squared_train[model_names[1:]].sub(r_squared_train["Gaussian"], axis=0)

# summary = pd.DataFrame({
#     "better than Gaussian (%)": 100 * (diff_train > 0).mean(),
#     "clearly worse, ΔR² < -0.01 (%)": 100 * (diff_train < -0.01).mean(),
#     "median ΔR²": diff_train.median(),
# })
# print("\nCompared with the Gaussian model (per voxel):")
# print(summary)

# # Which model has the highest training R² per voxel?
# best_train = r_squared_train.idxmax(axis=1)
# print("\nBest model per voxel on the training set (%):")
# print((100 * best_train.value_counts(normalize=True)).round(1))

# # Voxels where a complex model fits clearly worse: inspect a few
# for name in model_names[1:]:
#     worse = np.flatnonzero(diff_train[name] < -0.01)
#     print(f"{name}: {len(worse)} voxels clearly worse than Gaussian", worse[:10])


# # ------------------------------------------------------------------
# # Exercise 9: out-of-sample comparison
# # ------------------------------------------------------------------
# comparison = pd.DataFrame({
#     "train mean": r_squared_train.mean(),
#     "test mean": r_squared_test.mean(),
#     "train median": r_squared_train.median(),
#     "test median": r_squared_test.median(),
# })
# comparison["drop (train − test)"] = comparison["train mean"] - comparison["test mean"]
# comparison["rank train"] = comparison["train mean"].rank(ascending=False).astype(int)
# comparison["rank test"] = comparison["test mean"].rank(ascending=False).astype(int)
# print(comparison)

# # Advantage over the Gaussian model: training vs test set
# diff_test = r_squared_test[model_names[1:]].sub(r_squared_test["Gaussian"], axis=0)
# advantage = pd.DataFrame({
#     "mean ΔR² train": diff_train.mean(),
#     "mean ΔR² test": diff_test.mean(),
#     "% voxels better (train)": 100 * (diff_train > 0).mean(),
#     "% voxels better (test)": 100 * (diff_test > 0).mean(),
# })
# print("\nAdvantage over the Gaussian model:")
# print(advantage)

# # Is the test-set difference consistent across voxels? (paired test per model)
# from scipy.stats import wilcoxon
# for name in model_names[1:]:
#     stat, p = wilcoxon(r_squared_test[name], r_squared_test["Gaussian"])
#     print(f"{name} vs Gaussian (test): median ΔR² = {diff_test[name].median():+.4f}, p = {p:.2g}")

# # Best model per voxel on the test set
# best_test = r_squared_test.idxmax(axis=1)
# print("\nBest model per voxel (%):")
# print(pd.DataFrame({
#     "train": 100 * best_train.value_counts(normalize=True),
#     "test": 100 * best_test.value_counts(normalize=True),
# }).round(1))

# # Bar plot: mean ΔR² vs Gaussian, train and test side by side
# x = np.arange(len(model_names) - 1)
# fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
# ax.bar(x - 0.2, advantage["mean ΔR² train"], width=0.4, label="training set", color=GRAY)
# ax.bar(x + 0.2, advantage["mean ΔR² test"], width=0.4, label="test set", color=BLUE)
# ax.axhline(0, color="black", lw=0.8)
# ax.set_xticks(x, model_names[1:])
# ax.set_ylabel("Mean R² − Gaussian R²")
# ax.legend()
# plt.show()

# # ------------------------------------------------------------------
# # Exercise 10: models vs noise ceiling
# # ------------------------------------------------------------------
# # 1. How close do the models get?
# ratio = r_squared_test.div(noise_ceiling_selected, axis=0)  # 1 = as good as the noise allows
# print("Test R² as a fraction of the noise ceiling:")
# print(ratio.agg(["mean", "median"]))

# # 2. Voxels predicted better than the noise ceiling
# above = r_squared_test.gt(noise_ceiling_selected, axis=0)
# print("\nVoxels with test R² > noise ceiling (%):")
# print((100 * above.mean()).round(1))

# # 3. Voxels with a high noise ceiling that no model predicts well
# best_r2_test = r_squared_test.max(axis=1)
# is_bad = (noise_ceiling_selected > 0.85) & (best_r2_test < 0.3)
# bad_voxels = np.flatnonzero(is_bad)
# print(f"\n{len(bad_voxels)} voxels with noise ceiling > 0.85 but best test R² < 0.3")

# # Their estimated Gaussian pRFs: are they at the edge of / outside the stimulus? Very large?
# print(gaussian_params.iloc[bad_voxels[:10]])

# # Where are they on the brain?
# plot_surf_stat_map_helper(
#     project_to_surface(fill_selected_voxels(np.where(is_bad, 1.0, np.nan))),
#     vmin=0.0, vmax=1.0, title="High noise ceiling, poorly predicted",
# )
# plt.show()

# # 4. Plot their responses with the predictions of the best model
# frames_first = np.arange(response_first.shape[1])
# frames_second = np.arange(response_second.shape[1]) + split
# visible_first = is_bar_visible[first_half]
# visible_second = is_bar_visible[second_half]

# for v in bad_voxels[:6]:
#     best = r_squared_test.iloc[v].idxmax()
#     fig, ax = plt.subplots(figsize=(13, 3), layout="constrained")
#     # grey background = bar on screen
#     ax.fill_between(np.arange(num_frames), 0, 1, where=is_bar_visible,
#                     color="0.9", transform=ax.get_xaxis_transform(), label="bar visible")
#     ax.plot(frames_first, response_first[v], color="black", lw=1, label="measured")
#     ax.plot(frames_second, response_second[v], color="black", lw=1)
#     ax.plot(frames_first, pred_response_first[best][v], color=GRAY, lw=1.5,
#             label=f"{best} (fit, 1st half)")
#     ax.plot(frames_second, pred_response_second[best][v], color=BLUE, lw=1.5,
#             label=f"{best} (prediction, 2nd half)")
#     ax.axvline(split, color="black", ls="--", lw=0.8)
#     ax.set(xlabel="Time frame", ylabel="BOLD (% signal change)",
#            title=f"Voxel {v}: noise ceiling {noise_ceiling_selected[v]:.2f}, "
#                  f"best test R² {best_r2_test.iloc[v]:.2f} ({best})")
#     ax.legend(loc="upper right", fontsize=8)
#     plt.show()

# # ------------------------------------------------------------------
# # Exercise 11: voxels where CSS, DoG or DN clearly beat the Gaussian (test set)
# # ------------------------------------------------------------------
# from prfmodel.plotting import plot_observed_predicted

# delta_threshold = 0.1  # "much higher": at least 0.1 more test R² than the Gaussian

# for name in model_names[1:]:
#     better = diff_test[name] > delta_threshold
#     print(f"{name}: {better.sum()} voxels with test R² > Gaussian + {delta_threshold}")

# # For each complex model, take the 2 voxels with the largest advantage
# examples = {name: diff_test[name].nlargest(2).index.to_numpy() for name in model_names[1:]}
# print(examples)

# for name, voxels in examples.items():
#     for v in voxels:
#         predictions = {m: pred_response_second[m][v] for m in model_names}
#         plot_observed_predicted(response_second[v], predictions)
#         plt.title(f"Voxel {v}: {name} ΔR² = {diff_test[name].iloc[v]:+.2f} "
#                   f"(Gaussian {r_squared_test['Gaussian'].iloc[v]:.2f}, "
#                   f"{name} {r_squared_test[name].iloc[v]:.2f})", fontsize=9)
#         plt.show()

#         # Parameters that tell us *why* the model is better
#         if name == "CSS":
#             print(f"  voxel {v}: n = {css_params['n'].iloc[v]:.2f}")
#         elif name == "DoG":
#             p = dog_params.iloc[v]
#             print(f"  voxel {v}: surround/center size = {p['sigma_surround'] / p['sigma_center']:.1f}, "
#                   f"strength = {p['amplitude_surround'] / p['amplitude_center']:.3f}")

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# Set keras backend to 'tensorflow' (this is normally the default)
os.environ["KERAS_BACKEND"] = "tensorflow"
# Print parameter DataFrames with three decimals
pd.set_option("display.precision", 3)

import matplotlib.pyplot as plt
from prfmodel.examples import load_dataset
from prfmodel.stimuli import PRFStimulus

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
# Downloads on first use and caches in a user data directory (see prfmodel.examples.get_data_dir)
dataset = load_dataset("7t-aot-visual", surface_type="inflated")

stimulus = dataset.stimulus

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

response_no_baseline = dataset.response - 100.0

# Voxels with a response that varies over time (SD = 0 means a flat line)
response_is_valid = response_no_baseline.std(axis=1) > 0.0
response_valid = response_no_baseline[response_is_valid]
print("Valid voxels:", response_valid.shape)


def fill_valid_voxels(values: np.ndarray) -> np.ndarray:
    """Fill values for the valid voxels into an array with one value per voxel in the mask."""
    values_full = np.full((response_no_baseline.shape[0],), fill_value=np.nan)
    values_full[response_is_valid] = values
    return values_full


from plotting_helpers import make_surface_plot, make_surface_projection

project_to_surface = make_surface_projection(dataset)
plot_surf_stat_map_helper = make_surface_plot(dataset.mesh)

# ------------------------------------------------------------------
# Splitting the run into two halves
# ------------------------------------------------------------------
num_frames = stimulus.design.shape[0]

# A frame shows the bar if any pixel of the design matrix is non-zero
is_bar_visible = stimulus.design.reshape(num_frames, -1).max(axis=1) > 0

split = 168  # Middle of the blank period between frames 151 and 185

fig, ax = plt.subplots(figsize=(10, 2), layout="constrained")
ax.step(np.arange(num_frames), is_bar_visible, where="mid")
ax.axvline(split, color="black", linestyle="--", label="Split")
ax.set(xlabel="Time frame", yticks=[0, 1], yticklabels=["Blank", "Bar"])
ax.legend(loc="upper right")
plt.show()

# ------------------------------------------------------------------
# Exercise: video of the stimulus + check the bar sweeps in the table
# ------------------------------------------------------------------
from prfmodel.plotting import animate_2d_prf_stimulus

# 1. Video (saved as a GIF, since HTML video needs a notebook)
if not Path("practical3_stimulus.gif").exists():
    ani = animate_2d_prf_stimulus(stimulus, interval=25)
    ani.save("practical3_stimulus.gif", writer="pillow", fps=10)
    print("Saved practical3_stimulus.gif")

# 2. Find the sweeps: consecutive frames in which the bar is visible
design = np.asarray(stimulus.design)          # (frames, y, x)
grid = np.asarray(stimulus.grid)              # (y, x, 2): [..., 0] = y, [..., 1] = x

starts = np.flatnonzero(is_bar_visible & ~np.r_[False, is_bar_visible[:-1]])
ends = np.flatnonzero(is_bar_visible & ~np.r_[is_bar_visible[1:], False])

sweep_rows = []
for s, e in zip(starts, ends):
    cells_s, cells_e = design[s] > 0, design[e] > 0
    xs, ys = grid[..., 1][design[(s + e) // 2] > 0], grid[..., 0][design[(s + e) // 2] > 0]
    bar = "horizontal" if np.ptp(xs) > np.ptp(ys) else "vertical"
    if bar == "horizontal":            # a horizontal bar moves up or down
        moved = grid[..., 0][cells_e].mean() - grid[..., 0][cells_s].mean()
        direction = "bottom to top" if moved > 0 else "top to bottom"
    else:                              # a vertical bar moves left or right
        moved = grid[..., 1][cells_e].mean() - grid[..., 1][cells_s].mean()
        direction = "left to right" if moved > 0 else "right to left"
    sweep_rows.append({"frames": f"{s}-{e}", "half": "first" if e < split else "second",
                       "bar": bar, "direction": direction})

sweep_table = pd.DataFrame(sweep_rows, index=pd.RangeIndex(1, len(sweep_rows) + 1, name="sweep"))
print(f"\nNumber of sweeps found: {len(sweep_table)}")
print(sweep_table.to_string())

# 3. One frame from the middle of each sweep, for a visual check
extent = (grid[..., 1].min(), grid[..., 1].max(), grid[..., 0].min(), grid[..., 0].max())
fig, axes = plt.subplots(2, 4, figsize=(12, 5), layout="constrained")
for ax, (s, e), (_, row) in zip(axes.ravel(), zip(starts, ends), sweep_table.iterrows()):
    ax.imshow(design[(s + e) // 2], extent=extent, origin="lower", cmap="Greys", vmin=0, vmax=1)
    ax.set_title(f"{row['frames']}: {row['bar']}\n{row['direction']}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

first_half = slice(None, split)   # Frames 0 to 167
second_half = slice(split, None)  # Frames 168 to the end

stimulus_first = PRFStimulus(
    design=stimulus.design[first_half],
    grid=stimulus.grid,
    frame_duration=stimulus.frame_duration,
    dimension_labels=["y", "x"],
)
stimulus_second = PRFStimulus(
    design=stimulus.design[second_half],
    grid=stimulus.grid,
    frame_duration=stimulus.frame_duration,
    dimension_labels=["y", "x"],
)
print("Stimulus halves:", stimulus_first.design.shape, stimulus_second.design.shape)

# ------------------------------------------------------------------
# Noise ceiling from the repeated bar sweeps in both halves
# ------------------------------------------------------------------
from prfmodel.plotting import plot_r_squared_hist

sweeps = pd.DataFrame(
    {
        "onset_first": [30, 53, 101, 124],
        "onset_second": [287, 258, 216, 186],
        "num_frames": [20, 27, 20, 27],
    },
    index=[
        "Horizontal bar, top to bottom",
        "Vertical bar, left to right",
        "Horizontal bar, bottom to top",
        "Vertical bar, right to left",
    ],
)

lag = 5  # delay of the BOLD response in frames (5 x 0.9 s = 4.5 s)


def cut_sweeps(onsets: pd.Series) -> np.ndarray:
    """Concatenate the responses of all valid voxels to the bar sweeps, shifted by the delay of the BOLD response."""
    return np.concatenate(
        [response_valid[:, onset + lag:onset + lag + n] for onset, n in zip(onsets, sweeps["num_frames"])],
        axis=1,
    )


sweep_response_first = cut_sweeps(sweeps["onset_first"])
sweep_response_second = cut_sweeps(sweeps["onset_second"])
print("Sweep responses:", sweep_response_first.shape)        # (valid voxels, 94)


def correlate_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation between each row of x and the same row of y."""
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    return (x * y).sum(axis=1) / np.sqrt((x**2).sum(axis=1) * (y**2).sum(axis=1))


with np.errstate(invalid="ignore"):          # constant voxels give NaN instead of a warning
    noise_ceiling = correlate_rows(sweep_response_first, sweep_response_second)
print("Noise ceiling:", noise_ceiling.shape)

plot_r_squared_hist(noise_ceiling[np.isfinite(noise_ceiling)], value_range=(-1.0, 1.0), log=True,
                    xlabel="Noise ceiling")
plt.show()

plot_surf_stat_map_helper(project_to_surface(fill_valid_voxels(noise_ceiling)), vmin=0.0, vmax=1.0,
                          title="Noise ceiling")
plt.show()

# ------------------------------------------------------------------
# Selecting voxels with a high noise ceiling
# ------------------------------------------------------------------
noise_ceiling_threshold = 0.7

# NaN values are never above the threshold
with np.errstate(invalid="ignore"):
    is_selected = noise_ceiling > noise_ceiling_threshold

response_first = response_valid[is_selected][:, first_half]
response_second = response_valid[is_selected][:, second_half]

noise_ceiling_selected = noise_ceiling[is_selected]


def fill_selected_voxels(values: np.ndarray) -> np.ndarray:
    """Fill values for the selected voxels into an array with one value per voxel in the mask, NaN for all others."""
    values_valid = np.full((response_valid.shape[0],), fill_value=np.nan)
    values_valid[is_selected] = values
    return fill_valid_voxels(values_valid)


print("Selected responses:", response_first.shape, response_second.shape)

# ------------------------------------------------------------------
# Exercise 5: how many voxels for different thresholds?
# ------------------------------------------------------------------
with np.errstate(invalid="ignore"):            # NaNs compare as False
    for threshold in [0.5, 0.7, 0.9]:
        n_vox = np.sum(noise_ceiling > threshold)
        print(f"Noise ceiling > {threshold}: {n_vox} voxels "
              f"({100 * n_vox / len(noise_ceiling):.1f}% of valid voxels)")

# Where are they on the brain?
for threshold in [0.5, 0.9]:
    with np.errstate(invalid="ignore"):
        selected = noise_ceiling > threshold
    values = np.where(selected, noise_ceiling, np.nan)
    plot_surf_stat_map_helper(
        project_to_surface(fill_valid_voxels(values)),
        vmin=0.0, vmax=1.0,
        title=f"Voxels with noise ceiling > {threshold}",
    )
    plt.show()

# ------------------------------------------------------------------
# Candidate models
# ------------------------------------------------------------------
from keras import ops
from keras.optimizers import Adam
from prfmodel.fitters import GridFitter, LeastSquaresFitter, SGDFitter, grid_values_2d_prf
from prfmodel.fitters.adapter import Adapter, ParameterTransform
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.models.prf import (
    DivNormGaussian2DPRFModel,
    DoG2DPRFModel,
    Gaussian2DCSSPRFModel,
    Gaussian2DPRFModel,
    init_css_from_gaussian,
    init_dog_from_gaussian,
)

# Same SGD settings for all models
num_steps = 500
learning_rate = 0.01

# Define repetition time (TR)
tr = 0.9

# Create custom impulse model that is shared by all models
impulse_model = DerivativeTwoGammaImpulse(resolution=tr)

# Helper functions for the comparison figures
from plotting_helpers import BLUE, GRAY, add_legend, style_axes, to_frame

# All figures show a pRF in the center of the visual field
CENTER_PARAMS = {"mu_x": 0.0, "mu_y": 0.0, "weight_deriv": -0.5, "baseline": 0.0}
GAUSSIAN_PARAMS = {"sigma": 1.0, "amplitude": 1.0}

# Stimulus with one vertical bar per frame; the bar moves from left to right over frames
bar_width = 1.7  # Width of the bar in the experiment (degrees visual angle)
bar_positions = np.linspace(-4.0, 4.0, 81)
bar_stimulus = PRFStimulus(
    design=(np.abs(stimulus.grid[..., 1] - bar_positions[:, None, None]) <= bar_width / 2).astype("float32"),
    grid=stimulus.grid,
    frame_duration=stimulus.frame_duration,
    dimension_labels=["y", "x"],
)

time_frames_first = np.arange(stimulus_first.design.shape[0])


def scale_to_max(values: np.ndarray) -> np.ndarray:
    """Scale values so that their largest absolute value is 1."""
    values = np.asarray(values)
    max_value = np.abs(values).max()
    return values / max_value if max_value > 0 else values


def plot_model_comparison(model_class: type, model_name: str, **params: float) -> None:
    """Plot the bar response and the predicted BOLD response of a model next to those of the Gaussian model."""
    fig, (ax_bar, ax_response) = plt.subplots(
        1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 3]}, layout="constrained",
    )

    for current_class, current_name, current_params, style in [
        (Gaussian2DPRFModel, "Gaussian", GAUSSIAN_PARAMS, {"color": GRAY, "linestyle": "--", "linewidth": 1.5}),
        (model_class, model_name, params, {"color": BLUE, "linewidth": 2}),
    ]:
        param_frame = to_frame(**CENTER_PARAMS, **current_params)
        # Without an impulse response, every frame of the bar stimulus gives the response to the bar at one position
        bar_response = current_class(impulse_model=None)(bar_stimulus, param_frame)[0]
        bold_response = current_class(impulse_model=impulse_model)(stimulus_first, param_frame)[0]
        ax_bar.plot(bar_positions, scale_to_max(bar_response), label=current_name, **style)
        ax_response.plot(time_frames_first, scale_to_max(bold_response), label=current_name, **style)

    ax_bar.set_title("Response to the bar", loc="left")
    style_axes(ax_bar, "Bar position (degrees visual angle)", "Scaled response")

    add_legend(ax_response)
    ax_response.set_title(f"Predicted BOLD response ({model_name})", loc="left")
    style_axes(ax_response, "Time frame", "Scaled BOLD response")

    plt.show()


def cached_fit(name: str, fit_function) -> pd.DataFrame:
    """Load fitted parameters from results/ if they exist, otherwise fit and save them.

    Delete the .pkl files if you change the noise-ceiling threshold or the fitting settings.
    """
    path = RESULTS / f"practical3_{name}_params.pkl"
    if path.exists():
        print(f"Loading {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    params = fit_function()
    with open(path, "wb") as f:
        pickle.dump(params, f)
    print(f"Saved {path}")
    return params


# ------------------------------------------------------------------
# Fitting the models to the first half: Gaussian
# ------------------------------------------------------------------
gaussian_model = Gaussian2DPRFModel(impulse_model=impulse_model)


def fit_gaussian() -> pd.DataFrame:
    grid_param_ranges = grid_values_2d_prf(
        stimulus_first,
        num_mu=11,
        mu_extent=1.0,  # Only span the visual field covered by the stimulus
        num_sigma=10,
        sigma_range=(0.05, 4.0),  # Log-spaced by default
    ) | {  # Add constant parameters
        "weight_deriv": [-0.5],
        "baseline": [0.0],
        "amplitude": [1.0],
    }

    grid_fitter = GridFitter(
        model=gaussian_model,
        stimulus=stimulus_first,
        compile_step=True,  # Setting 'compile_step=True' speeds up the fitting
    )
    _, grid_params = grid_fitter.fit(
        data=response_first,
        parameter_values=grid_param_ranges,
        batch_size=20,
    )

    ls_fitter = LeastSquaresFitter(model=gaussian_model, stimulus=stimulus_first)
    _, ls_params = ls_fitter.fit(
        data=response_first,
        parameters=grid_params,
        slope_name="amplitude",
        intercept_name="baseline",
        batch_size=200,
    )

    gaussian_sgd_fitter = SGDFitter(
        model=gaussian_model,
        stimulus=stimulus_first,
        adapter=Adapter([ParameterTransform(["sigma"], transform_fun=ops.log, inverse_fun=ops.exp)]),
        optimizer=Adam(learning_rate=learning_rate),
        compile_step=True,
    )
    _, params = gaussian_sgd_fitter.fit(
        data=response_first,
        init_parameters=ls_params,
        fixed_parameters=["weight_deriv"],
        num_steps=num_steps,
    )
    return params


gaussian_params = cached_fit("gaussian", fit_gaussian)
print("\nGaussian parameters:\n", gaussian_params)

# ------------------------------------------------------------------
# CSS model
# ------------------------------------------------------------------
# Figure (replaces the interactive slider): the effect of the exponent n
for n in [1.0, 0.5, 0.2]:
    plot_model_comparison(Gaussian2DCSSPRFModel, f"CSS (n={n})", sigma=1.0, n=n, amplitude=1.0)

css_model = Gaussian2DCSSPRFModel(impulse_model=impulse_model)


def fit_css() -> pd.DataFrame:
    # Start from the Gaussian estimates with n = 0.5
    css_init_params = init_css_from_gaussian(gaussian_params)

    _, css_init_params = LeastSquaresFitter(model=css_model, stimulus=stimulus_first).fit(
        data=response_first,
        parameters=css_init_params,
        slope_name="amplitude",
        intercept_name="baseline",
        batch_size=200,
    )

    css_sgd_fitter = SGDFitter(
        model=css_model,
        stimulus=stimulus_first,
        adapter=Adapter([ParameterTransform(["sigma", "n"], transform_fun=ops.log, inverse_fun=ops.exp)]),
        optimizer=Adam(learning_rate=learning_rate),
        compile_step=True,
    )
    _, params = css_sgd_fitter.fit(
        data=response_first,
        init_parameters=css_init_params,
        fixed_parameters=["weight_deriv"],
        num_steps=num_steps,
    )
    return params


css_params = cached_fit("css", fit_css)
print("\nCSS parameters:\n", css_params)

# ------------------------------------------------------------------
# DoG model
# ------------------------------------------------------------------
# Figure (replaces the interactive sliders): surround strength
for amplitude_surround in [0.0, 0.3, 0.6]:
    plot_model_comparison(
        DoG2DPRFModel,
        f"DoG (amplitude_surround={amplitude_surround})",
        sigma_center=1.0,
        sigma_surround=2.0,
        amplitude_center=1.0,
        amplitude_surround=amplitude_surround,
    )

dog_model = DoG2DPRFModel(impulse_model=impulse_model)


def fit_dog() -> pd.DataFrame:
    # Start from the Gaussian estimates with a weak surround that is three times as large as the center
    dog_init_params = init_dog_from_gaussian(gaussian_params, sigma_ratio=3.0)

    _, dog_ls_params = LeastSquaresFitter(model=dog_model, stimulus=stimulus_first).fit(
        data=response_first,
        parameters=dog_init_params,
        slope_name=["amplitude_center", "amplitude_surround"],
        intercept_name="baseline",
        batch_size=200,
    )

    # Only use the least-squares estimates for voxels with a suppressive surround
    is_suppressive = dog_ls_params["amplitude_surround"] > 0.0
    dog_init_params.loc[is_suppressive] = dog_ls_params.loc[is_suppressive]
    print(f"Fraction of voxels with a suppressive surround after least squares: {is_suppressive.mean():.3f}")

    dog_sgd_fitter = SGDFitter(
        model=dog_model,
        stimulus=stimulus_first,
        adapter=Adapter([
            ParameterTransform(
                ["sigma_center", "sigma_surround", "amplitude_surround"], transform_fun=ops.log, inverse_fun=ops.exp,
            ),
        ]),
        optimizer=Adam(learning_rate=learning_rate),
        compile_step=True,
    )
    _, params = dog_sgd_fitter.fit(
        data=response_first,
        init_parameters=dog_init_params,
        fixed_parameters=["weight_deriv"],
        num_steps=num_steps,
    )
    return params


dog_params = cached_fit("dog", fit_dog)
print("\nDoG parameters:\n", dog_params)

# ------------------------------------------------------------------
# DN model
# ------------------------------------------------------------------
# Figures (replace the interactive sliders): normalization size and the two baselines
for sigma_normalization, baseline_activation, baseline_normalization in [
    (2.0, 1.0, 1.0),
    (2.0, 0.01, 1.0),    # small baseline_activation
    (2.0, 1.0, 0.01),    # small baseline_normalization
    (6.0, 1.0, 1.0),     # large normalization pool
]:
    plot_model_comparison(
        DivNormGaussian2DPRFModel,
        f"DN (sigma_N={sigma_normalization}, b={baseline_activation}, d={baseline_normalization})",
        sigma_activation=1.0,
        sigma_normalization=sigma_normalization,
        amplitude_activation=1.0,
        amplitude_normalization=1.0,
        baseline_activation=baseline_activation,
        baseline_normalization=baseline_normalization,
    )

dn_model = DivNormGaussian2DPRFModel(impulse_model=impulse_model)


def fit_dn() -> pd.DataFrame:
    """Fit the DN model, starting from the Gaussian estimates.

    NOTE: if your notebook has its own DN fitting cell, use that one instead of this function.
    Starting values: activation Gaussian = Gaussian pRF, normalization Gaussian 3x larger,
    c (amplitude_normalization) fixed to 1, d (baseline_normalization) = 1, b (baseline_activation) tiny.
    """
    g = gaussian_params
    dn_init_params = pd.DataFrame({
        "mu_x": g["mu_x"],
        "mu_y": g["mu_y"],
        "sigma_activation": g["sigma"],
        "sigma_normalization": 3.0 * g["sigma"],
        "amplitude_activation": g["amplitude"],
        "amplitude_normalization": 1.0,
        "baseline_activation": 1e-3,
        "baseline_normalization": 1.0,
        "baseline": g["baseline"],
        "weight_deriv": g["weight_deriv"],
    })

    ls_fitter = LeastSquaresFitter(model=dn_model, stimulus=stimulus_first)

    # With a tiny b, the response is (almost) linear in a: estimate a and the BOLD baseline
    _, dn_ls_a = ls_fitter.fit(
        data=response_first,
        parameters=dn_init_params,
        slope_name="amplitude_activation",
        intercept_name="baseline",
        batch_size=200,
    )

    # The response is linear in both a and b: estimate them together
    _, dn_ls_ab = ls_fitter.fit(
        data=response_first,
        parameters=dn_init_params,
        slope_name=["amplitude_activation", "baseline_activation"],
        intercept_name="baseline",
        batch_size=200,
    )

    # b must be positive (optimized in log space): only use the joint estimates where a and b > 0
    is_valid = (dn_ls_ab["amplitude_activation"] > 0.0) & (dn_ls_ab["baseline_activation"] > 0.0)
    dn_init_params = dn_ls_a.copy()
    dn_init_params.loc[is_valid] = dn_ls_ab.loc[is_valid]
    print(f"Fraction of voxels using the joint a/b least-squares estimates: {is_valid.mean():.3f}")

    dn_sgd_fitter = SGDFitter(
        model=dn_model,
        stimulus=stimulus_first,
        adapter=Adapter([
            ParameterTransform(
                ["sigma_activation", "sigma_normalization", "baseline_activation", "baseline_normalization"],
                transform_fun=ops.log, inverse_fun=ops.exp,
            ),
        ]),
        optimizer=Adam(learning_rate=learning_rate),
        compile_step=True,
    )
    _, params = dn_sgd_fitter.fit(
        data=response_first,
        init_parameters=dn_init_params,
        fixed_parameters=["weight_deriv", "amplitude_normalization"],  # c is fixed to 1
        num_steps=num_steps,
    )
    return params


dn_params = cached_fit("dn", fit_dn)
print("\nDN parameters:\n", dn_params)

# ------------------------------------------------------------------
# Exercise 7: parameter names and the distributions of n and the DoG size ratio
# ------------------------------------------------------------------
for name, model in [("Gaussian", gaussian_model), ("CSS", css_model),
                    ("DoG", dog_model), ("DN", dn_model)]:
    try:
        names = list(model.parameter_names)
    except AttributeError:
        names = list({"Gaussian": gaussian_params, "CSS": css_params,
                      "DoG": dog_params, "DN": dn_params}[name].columns)
    print(f"{name:8s} ({len(names)} names): {names}")

n_values = css_params["n"].to_numpy()
print(f"\nCSS n: median {np.median(n_values):.2f}, "
      f"{100 * np.mean(n_values < 1):.1f}% of voxels have n < 1 (compression)")

size_ratio = (dog_params["sigma_surround"] / dog_params["sigma_center"]).to_numpy()
strength_ratio = (dog_params["amplitude_surround"] / dog_params["amplitude_center"]).to_numpy()
print(f"DoG sigma_surround / sigma_center: median {np.median(size_ratio):.2f}")
print(f"DoG amplitude_surround / amplitude_center: median {np.median(strength_ratio):.3f}")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), layout="constrained")
axes[0].hist(n_values, bins=40, color=BLUE)
axes[0].axvline(1.0, color="black", ls="--", label="n = 1 (Gaussian)")
axes[0].set(xlabel="CSS exponent n", ylabel="Number of voxels", title="Compression")
axes[0].legend()
axes[1].hist(size_ratio, bins=np.logspace(0, np.log10(size_ratio.max()), 40), color=BLUE)
axes[1].set_xscale("log")
axes[1].axvline(3.0, color="black", ls="--", label="starting value (3)")
axes[1].set(xlabel="sigma_surround / sigma_center", title="Surround size")
axes[1].legend()
axes[2].hist(strength_ratio, bins=40, color=BLUE)
axes[2].set(xlabel="amplitude_surround / amplitude_center", title="Surround strength")
plt.show()

plot_surf_stat_map_helper(project_to_surface(fill_selected_voxels(n_values)),
                          vmin=0.0, vmax=1.0, title="CSS exponent n")
plt.show()
plot_surf_stat_map_helper(project_to_surface(fill_selected_voxels(np.log10(size_ratio))),
                          title="log10(sigma_surround / sigma_center)")
plt.show()

# ------------------------------------------------------------------
# Comparing the in-sample fit (notebook cell)
# ------------------------------------------------------------------
from prfmodel.plotting import plot_observed_predicted, plot_r_squared_comparison
from prfmodel.utils import batched, calculate_r_squared

model_names = ["Gaussian", "CSS", "DoG", "DN"]  # From the simplest to the most complex model

models = dict(zip(model_names, [gaussian_model, css_model, dog_model, dn_model]))
model_params = dict(zip(model_names, [gaussian_params, css_params, dog_params, dn_params]))

# Make predictions for batches of voxels to save memory
predict_batched = {name: batched(model) for name, model in models.items()}

pred_response_first = {
    name: np.asarray(predict_batched[name](stimulus_first, model_params[name], batch_size=200))
    for name in model_names
}

# One column per model and one row per selected voxel
r_squared_train = pd.DataFrame(
    {name: calculate_r_squared(response_first, pred_response_first[name]) for name in model_names},
)
print("\nIn-sample R²:")
print(r_squared_train.agg(["mean", "median"]))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
for ax, name in zip(axes, model_names[1:]):
    plot_r_squared_comparison(
        r_squared_train["Gaussian"],
        r_squared_train[name],
        value_range=(0.0, 1.0),
        xlabel="R-squared Gaussian (training set)",
        ylabel=f"R-squared {name} (training set)",
        ax=ax,
    )
plt.show()

# ------------------------------------------------------------------
# Exercise 8: in-sample comparison against the Gaussian model
# ------------------------------------------------------------------
diff_train = r_squared_train[model_names[1:]].sub(r_squared_train["Gaussian"], axis=0)

summary = pd.DataFrame({
    "better than Gaussian (%)": 100 * (diff_train > 0).mean(),
    "clearly worse, dR2 < -0.01 (%)": 100 * (diff_train < -0.01).mean(),
    "median dR2": diff_train.median(),
})
print("\nCompared with the Gaussian model (per voxel, training set):")
print(summary)

best_train = r_squared_train.idxmax(axis=1)
print("\nBest model per voxel on the training set (%):")
print((100 * best_train.value_counts(normalize=True)).round(1))

for name in model_names[1:]:
    worse = np.flatnonzero(diff_train[name] < -0.01)
    print(f"{name}: {len(worse)} voxels clearly worse than Gaussian", worse[:10])

# ------------------------------------------------------------------
# Comparing the out-of-sample fit (notebook cell)
# ------------------------------------------------------------------
pred_response_second = {
    name: np.asarray(predict_batched[name](stimulus_second, model_params[name], batch_size=200))
    for name in model_names
}

r_squared_test = pd.DataFrame(
    {name: calculate_r_squared(response_second, pred_response_second[name]) for name in model_names},
)
print(pd.DataFrame({"training set": r_squared_train.mean(), "test set": r_squared_test.mean()}))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
for ax, name in zip(axes, model_names[1:]):
    plot_r_squared_comparison(
        r_squared_test["Gaussian"],
        r_squared_test[name],
        value_range=(0.0, 1.0),
        xlabel="R-squared Gaussian (test set)",
        ylabel=f"R-squared {name} (test set)",
        ax=ax,
    )
plt.show()

# ------------------------------------------------------------------
# Exercise 9: out-of-sample comparison
# ------------------------------------------------------------------
comparison = pd.DataFrame({
    "train mean": r_squared_train.mean(),
    "test mean": r_squared_test.mean(),
    "train median": r_squared_train.median(),
    "test median": r_squared_test.median(),
})
comparison["drop (train - test)"] = comparison["train mean"] - comparison["test mean"]
comparison["rank train"] = comparison["train mean"].rank(ascending=False).astype(int)
comparison["rank test"] = comparison["test mean"].rank(ascending=False).astype(int)
print(comparison)

diff_test = r_squared_test[model_names[1:]].sub(r_squared_test["Gaussian"], axis=0)
advantage = pd.DataFrame({
    "mean dR2 train": diff_train.mean(),
    "mean dR2 test": diff_test.mean(),
    "% voxels better (train)": 100 * (diff_train > 0).mean(),
    "% voxels better (test)": 100 * (diff_test > 0).mean(),
})
print("\nAdvantage over the Gaussian model:")
print(advantage)

try:
    from scipy.stats import wilcoxon
    for name in model_names[1:]:
        _, p = wilcoxon(r_squared_test[name], r_squared_test["Gaussian"])
        print(f"{name} vs Gaussian (test): median dR2 = {diff_test[name].median():+.4f}, p = {p:.2g}")
except ImportError:
    print("scipy not installed: skipping the Wilcoxon test")

best_test = r_squared_test.idxmax(axis=1)
print("\nBest model per voxel (%):")
print(pd.DataFrame({
    "train": 100 * best_train.value_counts(normalize=True),
    "test": 100 * best_test.value_counts(normalize=True),
}).round(1))

x = np.arange(len(model_names) - 1)
fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
ax.bar(x - 0.2, advantage["mean dR2 train"], width=0.4, label="training set", color=GRAY)
ax.bar(x + 0.2, advantage["mean dR2 test"], width=0.4, label="test set", color=BLUE)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x, model_names[1:])
ax.set_ylabel("Mean R² − Gaussian R²")
ax.legend()
plt.show()

# ------------------------------------------------------------------
# Test R² vs noise ceiling (notebook cell)
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), layout="constrained")
for ax, name in zip(axes, model_names):
    plot_r_squared_comparison(
        noise_ceiling_selected,
        r_squared_test[name],
        value_range=(0.0, 1.0),
        xlabel="Noise ceiling",
        ylabel=f"R-squared {name} (test set)",
        ax=ax,
    )
plt.show()

# ------------------------------------------------------------------
# Exercise 10: models vs noise ceiling
# ------------------------------------------------------------------
ratio = r_squared_test.div(noise_ceiling_selected, axis=0)  # 1 = as good as the noise allows
print("Test R² as a fraction of the noise ceiling:")
print(ratio.agg(["mean", "median"]))

above = r_squared_test.gt(noise_ceiling_selected, axis=0)
print("\nVoxels with test R² > noise ceiling (%):")
print((100 * above.mean()).round(1))

best_r2_test = r_squared_test.max(axis=1)
is_bad = (noise_ceiling_selected > 0.85) & (best_r2_test.to_numpy() < 0.3)
bad_voxels = np.flatnonzero(is_bad)
print(f"\n{len(bad_voxels)} voxels with noise ceiling > 0.85 but best test R² < 0.3")
print(gaussian_params.iloc[bad_voxels[:10]])

if len(bad_voxels) > 0:
    plot_surf_stat_map_helper(
        project_to_surface(fill_selected_voxels(np.where(is_bad, 1.0, np.nan))),
        vmin=0.0, vmax=1.0, title="High noise ceiling, poorly predicted",
    )
    plt.show()

frames_first = np.arange(response_first.shape[1])
frames_second = np.arange(response_second.shape[1]) + split

for v in bad_voxels[:6]:
    best = r_squared_test.iloc[v].idxmax()
    fig, ax = plt.subplots(figsize=(13, 3), layout="constrained")
    ax.fill_between(np.arange(num_frames), 0, 1, where=is_bar_visible,
                    color="0.9", transform=ax.get_xaxis_transform(), label="bar visible")
    ax.plot(frames_first, response_first[v], color="black", lw=1, label="measured")
    ax.plot(frames_second, response_second[v], color="black", lw=1)
    ax.plot(frames_first, pred_response_first[best][v], color=GRAY, lw=1.5, label=f"{best} (fit, 1st half)")
    ax.plot(frames_second, pred_response_second[best][v], color=BLUE, lw=1.5,
            label=f"{best} (prediction, 2nd half)")
    ax.axvline(split, color="black", ls="--", lw=0.8)
    ax.set(xlabel="Time frame", ylabel="BOLD (% signal change)",
           title=f"Voxel {v}: noise ceiling {noise_ceiling_selected[v]:.2f}, "
                 f"best test R² {best_r2_test.iloc[v]:.2f} ({best})")
    ax.legend(loc="upper right", fontsize=8)
    plt.show()

# ------------------------------------------------------------------
# Exercise 11: voxels where CSS, DoG or DN clearly beat the Gaussian (test set)
# ------------------------------------------------------------------
delta_threshold = 0.1  # "much higher": at least 0.1 more test R² than the Gaussian

for name in model_names[1:]:
    print(f"{name}: {(diff_test[name] > delta_threshold).sum()} voxels "
          f"with test R² > Gaussian + {delta_threshold}")

examples = {name: diff_test[name].nlargest(2).index.to_numpy() for name in model_names[1:]}
print(examples)

frames = np.arange(response_second.shape[1]) + split
colors = {"Gaussian": GRAY, "CSS": "tab:orange", "DoG": "tab:green", "DN": BLUE}
for name, voxels in examples.items():
    for v in voxels:
        fig, ax = plt.subplots(figsize=(13, 3), layout="constrained")
        ax.plot(frames, response_second[v], color="black", lw=1, label="observed")
        for m in model_names:
            ax.plot(frames, pred_response_second[m][v], color=colors[m], lw=1.5, label=m)
        ax.axhline(0, color="black", lw=0.5)
        ax.set(xlabel="Time frame", ylabel="BOLD",
               title=f"Voxel {v}: {name} dR2 = {diff_test[name].iloc[v]:+.2f} "
                     f"(Gaussian {r_squared_test['Gaussian'].iloc[v]:.2f}, "
                     f"{name} {r_squared_test[name].iloc[v]:.2f})")
        ax.legend(fontsize=8)
        plt.show()

        if name == "CSS":
            print(f"  voxel {v}: n = {css_params['n'].iloc[v]:.2f}")
        elif name == "DoG":
            p = dog_params.iloc[v]
            print(f"  voxel {v}: surround/center size = {p['sigma_surround'] / p['sigma_center']:.1f}, "
                  f"strength = {p['amplitude_surround'] / p['amplitude_center']:.3f}")

# The cost
# More parameters, so more risk of overfitting. You saw this: DN's training wins dropped from 79% to 57% of voxels on the test set.
# Harder fitting: local minima, sequential initialization, and non-identifiability (that's why c is fixed in DN). Parameters can also trade off, like σ and n.
# More computation (the DN fit took about 2.5 minutes on your CPU).
# Compression is the main thing the Gaussian model misses. Almost all selected voxels have n < 1, and CSS gets most of the improvement with just one extra parameter, with no loss on the test set.
# DN is the best model on average, but its lead over CSS is tiny, and part of its training advantage is overfitting.
# Surround suppression (DoG) matters only for a minority of voxels; for most, the surround adds nothing.
# All models are close together and reach about 80% of the noise ceiling. The remaining gap comes from things none of these models capture: per-voxel HRF timing, non-retinotopic responses, and the noise-ceiling estimate itself.
# No single model is best everywhere: DN wins 57% of voxels, CSS 28%, DoG 8% and Gaussian 7%. That's the motivation for the next section, selecting a model per voxel.

# Selecting a model per voxel
def select_models(r_squared: pd.DataFrame, min_gain: float) -> np.ndarray:
    """Select the most complex model that beats all simpler models by more than min_gain for each voxel.

    The columns of r_squared must be ordered from the simplest to the most complex model.
    """
    scores = r_squared.to_numpy()
    selected = np.zeros(scores.shape[0], dtype=int)  # Start with the simplest model for all voxels
    for i in range(1, scores.shape[1]):
        beats_simpler = np.all(scores[:, [i]] > scores[:, :i] + min_gain, axis=1)
        selected[beats_simpler] = i
    return r_squared.columns.to_numpy()[selected]


min_gain = 0.01

best_model = select_models(r_squared_test[model_names], min_gain=min_gain)

# Fraction of selected voxels for which each model is the best model
pd.Series(best_model).value_counts(normalize=True).reindex(model_names, fill_value=0.0)

is_best = (best_model == name).astype(float)
plot_surf_stat_map_helper(
    project_to_surface(fill_selected_voxels(is_best)),
    vmin=0.0,
    vmax=1.0,
    title=f"Fraction of voxels for which {name} is selected",
)

for name in model_names:
    is_best = (best_model == name).astype(float)
    plot_surf_stat_map_helper(
        project_to_surface(fill_selected_voxels(is_best)),
        vmin=0.0, vmax=1.0,
        title=f"Fraction of voxels for which {name} is selected",
    )
    plt.show()

# ------------------------------------------------------------------
# Exercise 12
# ------------------------------------------------------------------
plot_surf_stat_map_helper(project_to_surface(fill_selected_voxels(noise_ceiling_selected)),
                          vmin=0.0, vmax=1.0, title="Noise ceiling (selected voxels)")
plt.show()

# Numbers behind the maps: properties of the voxels per selected model
eccentricity = np.sqrt(gaussian_params["mu_x"] ** 2 + gaussian_params["mu_y"] ** 2).to_numpy()
per_model = pd.DataFrame({
    "selected model": best_model,
    "noise ceiling": noise_ceiling_selected,
    "eccentricity (Gaussian)": eccentricity,
    "sigma (Gaussian)": gaussian_params["sigma"].to_numpy(),
}).groupby("selected model").median().reindex(model_names)
per_model["number of voxels"] = pd.Series(best_model).value_counts().reindex(model_names, fill_value=0)
print(per_model)

# ------------------------------------------------------------------
# Exercise 13
# ------------------------------------------------------------------
fractions = pd.DataFrame({
    f"min_gain = {g}": pd.Series(select_models(r_squared_test[model_names], min_gain=g))
    .value_counts(normalize=True).reindex(model_names, fill_value=0.0)
    for g in [0.0, 0.01, 0.05]
})
print((100 * fractions).round(1))

fractions.T.plot(kind="bar", stacked=True, figsize=(6, 4), rot=0)
plt.ylabel("Fraction of voxels")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# Exercise 14
# ------------------------------------------------------------------
# diff_test (from Exercise 9) = test R² of each model − test R² of the Gaussian, per voxel
# Eccentricity of the Gaussian pRF centre (same as prfmodel.utils.calculate_eccentricity)
eccentricity = np.sqrt(gaussian_params["mu_x"] ** 2 + gaussian_params["mu_y"] ** 2).to_numpy()
sigma_gauss = gaussian_params["sigma"].to_numpy()

ecc_bins = np.arange(0, 12.5, 1.0)                 # 1-degree bins
sigma_bins = np.logspace(np.log10(0.05), np.log10(sigma_gauss.max()), 15)

colors = {"CSS": "tab:orange", "DoG": "tab:green", "DN": BLUE}
fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")

for ax, values, bins, label in [
    (axes[0], eccentricity, ecc_bins, "Eccentricity (deg), Gaussian pRF"),
    (axes[1], sigma_gauss, sigma_bins, "pRF size sigma (deg), Gaussian"),
]:
    groups = pd.cut(values, bins)
    for name in model_names[1:]:
        ax.scatter(values, diff_test[name], s=2, alpha=0.15, color=colors[name])
        binned = diff_test[name].groupby(groups, observed=True).mean()
        centers = [interval.mid for interval in binned.index]
        ax.plot(centers, binned.to_numpy(), "-o", color=colors[name], lw=2, label=name)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel=label, ylabel="Test R² improvement over Gaussian", ylim=(-0.1, 0.3))
    ax.legend()

# Edge of the stimulus: ~3.9° horizontally, ~2.8° vertically
axes[0].axvspan(2.8, 3.9, color="0.85", zorder=0, label="stimulus edge")
axes[1].set_xscale("log")
plt.show()

# How many pRF centres lie outside the stimulus?
outside = (np.abs(gaussian_params["mu_x"]) > 3.9) | (np.abs(gaussian_params["mu_y"]) > 2.8)
print(f"pRF centres outside the stimulus: {100 * outside.mean():.1f}% of voxels")
print("Mean improvement (inside vs outside):")
print(pd.DataFrame({"inside": diff_test[~outside.to_numpy()].mean(),
                    "outside": diff_test[outside.to_numpy()].mean()}))

# On the surface, with a symmetric diverging colour range
for name in model_names[1:]:
    plot_surf_stat_map_helper(
        project_to_surface(fill_selected_voxels(diff_test[name].to_numpy())),
        vmin=-0.1, vmax=0.1, cmap="RdBu_r",
        title=f"Test R² improvement of {name} over Gaussian",
    )
    plt.show()

# ------------------------------------------------------------------
# Exercise 16: fit all four models to the full run
# ------------------------------------------------------------------
response_full = response_valid[is_selected]   # both halves, selected voxels


def run_sgd(model, stim, data, init_params, log_names, fixed):
    fitter = SGDFitter(
        model=model, stimulus=stim,
        adapter=Adapter([ParameterTransform(log_names, transform_fun=ops.log, inverse_fun=ops.exp)]),
        optimizer=Adam(learning_rate=learning_rate), compile_step=True,
    )
    _, params = fitter.fit(data=data, init_parameters=init_params,
                           fixed_parameters=fixed, num_steps=num_steps)
    return params


def fit_gaussian_on(stim, data):
    grid_ranges = grid_values_2d_prf(stim, num_mu=11, mu_extent=1.0, num_sigma=10,
                                     sigma_range=(0.05, 4.0)) | {
        "weight_deriv": [-0.5], "baseline": [0.0], "amplitude": [1.0]}
    _, grid_p = GridFitter(model=gaussian_model, stimulus=stim, compile_step=True).fit(
        data=data, parameter_values=grid_ranges, batch_size=20)
    _, ls_p = LeastSquaresFitter(model=gaussian_model, stimulus=stim).fit(
        data=data, parameters=grid_p, slope_name="amplitude", intercept_name="baseline", batch_size=200)
    return run_sgd(gaussian_model, stim, data, ls_p, ["sigma"], ["weight_deriv"])


def fit_css_on(stim, data, g_params):
    _, init = LeastSquaresFitter(model=css_model, stimulus=stim).fit(
        data=data, parameters=init_css_from_gaussian(g_params),
        slope_name="amplitude", intercept_name="baseline", batch_size=200)
    return run_sgd(css_model, stim, data, init, ["sigma", "n"], ["weight_deriv"])


def fit_dog_on(stim, data, g_params):
    init = init_dog_from_gaussian(g_params, sigma_ratio=3.0)
    _, ls_p = LeastSquaresFitter(model=dog_model, stimulus=stim).fit(
        data=data, parameters=init, slope_name=["amplitude_center", "amplitude_surround"],
        intercept_name="baseline", batch_size=200)
    ok = ls_p["amplitude_surround"] > 0.0
    init.loc[ok] = ls_p.loc[ok]
    return run_sgd(dog_model, stim, data, init,
                   ["sigma_center", "sigma_surround", "amplitude_surround"], ["weight_deriv"])


def fit_dn_on(stim, data, g_params):
    init = pd.DataFrame({
        "mu_x": g_params["mu_x"], "mu_y": g_params["mu_y"],
        "sigma_activation": g_params["sigma"], "sigma_normalization": 3.0 * g_params["sigma"],
        "amplitude_activation": g_params["amplitude"], "amplitude_normalization": 1.0,
        "baseline_activation": 1e-3, "baseline_normalization": 1.0,
        "baseline": g_params["baseline"], "weight_deriv": g_params["weight_deriv"],
    })
    ls = LeastSquaresFitter(model=dn_model, stimulus=stim)
    _, ls_a = ls.fit(data=data, parameters=init, slope_name="amplitude_activation",
                     intercept_name="baseline", batch_size=200)
    _, ls_ab = ls.fit(data=data, parameters=init,
                      slope_name=["amplitude_activation", "baseline_activation"],
                      intercept_name="baseline", batch_size=200)
    ok = (ls_ab["amplitude_activation"] > 0) & (ls_ab["baseline_activation"] > 0)
    init = ls_a.copy()
    init.loc[ok] = ls_ab.loc[ok]
    return run_sgd(dn_model, stim, data, init,
                   ["sigma_activation", "sigma_normalization", "baseline_activation", "baseline_normalization"],
                   ["weight_deriv", "amplitude_normalization"])


full_params = {}
full_params["Gaussian"] = cached_fit("full_gaussian", lambda: fit_gaussian_on(stimulus, response_full))
g_full = full_params["Gaussian"]
full_params["CSS"] = cached_fit("full_css", lambda: fit_css_on(stimulus, response_full, g_full))
full_params["DoG"] = cached_fit("full_dog", lambda: fit_dog_on(stimulus, response_full, g_full))
full_params["DN"] = cached_fit("full_dn", lambda: fit_dn_on(stimulus, response_full, g_full))

# pRF centre of the selected model for each voxel
conditions = [best_model == name for name in model_names]
mu_x_selected = np.select(conditions, [full_params[m]["mu_x"].to_numpy() for m in model_names])
mu_y_selected = np.select(conditions, [full_params[m]["mu_y"].to_numpy() for m in model_names])

from prfmodel.utils import calculate_eccentricity, calculate_polar_angle


def plot_angle_ecc(mu_x, mu_y, label):
    # Project mu_x and mu_y separately, THEN compute angle/eccentricity per vertex
    # (averaging angles directly would be wrong: e.g. 350° and 10° would average to 180°)
    surf_x = project_to_surface(fill_selected_voxels(mu_x))
    surf_y = project_to_surface(fill_selected_voxels(mu_y))
    plot_surf_stat_map_helper(calculate_polar_angle(surf_x, surf_y), cmap="hsv",
                              title=f"Polar angle ({label})")
    plt.show()
    plot_surf_stat_map_helper(calculate_eccentricity(surf_x, surf_y), vmin=0.0, vmax=8.0,
                              title=f"Eccentricity ({label})")
    plt.show()


plot_angle_ecc(mu_x_selected, mu_y_selected, "selected models")
plot_angle_ecc(g_full["mu_x"].to_numpy(), g_full["mu_y"].to_numpy(), "Gaussian only")

# Numbers: how different are the centres?
ecc_sel = np.hypot(mu_x_selected, mu_y_selected)
ecc_gau = np.hypot(g_full["mu_x"], g_full["mu_y"]).to_numpy()
angle_diff = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(mu_y_selected, mu_x_selected)
                                                     - np.arctan2(g_full["mu_y"], g_full["mu_x"]))))))
inside = (np.abs(g_full["mu_x"]) <= 3.9) & (np.abs(g_full["mu_y"]) <= 2.8)
print(pd.DataFrame({
    "median |Δ eccentricity| (deg)": [np.median(np.abs(ecc_sel - ecc_gau)[inside]),
                                      np.median(np.abs(ecc_sel - ecc_gau)[~inside])],
    "median |Δ polar angle| (deg)": [np.median(angle_diff[inside]), np.median(angle_diff[~inside])],
}, index=["centre inside stimulus", "centre outside stimulus"]))

# ------------------------------------------------------------------
# Exercise 17
# ------------------------------------------------------------------
def binned_plot(ax, ecc, values, label, color):
    ax.scatter(ecc, values, s=6, alpha=0.3, color=color)
    bins = np.arange(0, 10.5, 1.0)
    med = pd.Series(values).groupby(pd.cut(ecc, bins), observed=True).median()
    ax.plot([b.mid for b in med.index], med.to_numpy(), "-o", color="black", lw=2, label="median per 1°")
    ax.axvspan(2.8, 3.9, color="0.85", zorder=0, label="stimulus edge")
    ax.set(xlabel="Eccentricity (deg)", ylabel=label, xlim=(0, 10))
    ax.legend(fontsize=8)


sel = {name: best_model == name for name in model_names}
p_css, p_dog, p_dn = (full_params[m] for m in ["CSS", "DoG", "DN"])

fig, axes = plt.subplots(1, 3, figsize=(16, 4), layout="constrained")

ecc = np.hypot(p_css["mu_x"], p_css["mu_y"]).to_numpy()[sel["CSS"]]
binned_plot(axes[0], ecc, p_css["n"].to_numpy()[sel["CSS"]], "CSS exponent n", "tab:orange")
axes[0].set_title(f"Compression (CSS voxels, n={sel['CSS'].sum()})")

ecc = np.hypot(p_dog["mu_x"], p_dog["mu_y"]).to_numpy()[sel["DoG"]]
ratio_dog = (p_dog["sigma_surround"] / p_dog["sigma_center"]).to_numpy()[sel["DoG"]]
binned_plot(axes[1], ecc, ratio_dog, "sigma_surround / sigma_center", "tab:green")
axes[1].set_title(f"Surround size (DoG voxels, n={sel['DoG'].sum()})")

ecc = np.hypot(p_dn["mu_x"], p_dn["mu_y"]).to_numpy()[sel["DN"]]
ratio_dn = (p_dn["sigma_normalization"] / p_dn["sigma_activation"]).to_numpy()[sel["DN"]]
binned_plot(axes[2], ecc, ratio_dn, "sigma_normalization / sigma_activation", BLUE)
axes[2].set_title(f"Normalization pool (DN voxels, n={sel['DN'].sum()})")
plt.show()

# Surround strength for DoG voxels
strength = (p_dog["amplitude_surround"] / p_dog["amplitude_center"]).to_numpy()[sel["DoG"]]
print(f"DoG voxels: median surround strength = {np.median(strength):.3f}")

# ------------------------------------------------------------------
# Exercise 18
# ------------------------------------------------------------------
from math import erf, sqrt


def fraction_on_screen(mu_x, mu_y, sigma):
    """Fraction of a 2D Gaussian's volume that lies on the (rectangular) stimulus screen."""
    def frac_1d(mu, s, lo, hi):
        return 0.5 * (erf((hi - mu) / (sqrt(2) * s)) - erf((lo - mu) / (sqrt(2) * s)))
    return frac_1d(mu_x, sigma, x_min, x_max) * frac_1d(mu_y, sigma, y_min, y_max)


coverage = {}
for name, sig_inner, sig_outer in [("DoG", "sigma_center", "sigma_surround"),
                                   ("DN", "sigma_activation", "sigma_normalization")]:
    p = full_params[name][sel[name]]
    coverage[name] = pd.DataFrame({
        "inner (centre/activation)": [fraction_on_screen(r.mu_x, r.mu_y, r[sig_inner]) for _, r in p.iterrows()],
        "outer (surround/normalization)": [fraction_on_screen(r.mu_x, r.mu_y, r[sig_outer]) for _, r in p.iterrows()],
    })
    print(f"\n{name} voxels: median fraction of the pRF volume on the screen")
    print(coverage[name].median().round(3))

# Picture for a few example voxels: the pRFs on a grid larger than the screen
xs = np.linspace(-15, 15, 301)
X, Y = np.meshgrid(xs, xs)


def gauss2d(mx, my, s):
    return np.exp(-((X - mx) ** 2 + (Y - my) ** 2) / (2 * s**2)) / (2 * np.pi * s**2)


for name, sig_inner, sig_outer, amp_outer in [
    ("DoG", "sigma_center", "sigma_surround", "amplitude_surround"),
    ("DN", "sigma_activation", "sigma_normalization", None),
]:
    voxels = np.flatnonzero(sel[name])[:3]
    fig, axes = plt.subplots(1, len(voxels), figsize=(5 * len(voxels), 4.5), layout="constrained")
    for ax, v in zip(np.atleast_1d(axes), voxels):
        r = full_params[name].iloc[v]
        if name == "DoG":
            prf = r["amplitude_center"] * gauss2d(r.mu_x, r.mu_y, r[sig_inner]) \
                  - r[amp_outer] * gauss2d(r.mu_x, r.mu_y, r[sig_outer])
            lim = np.abs(prf).max()
            im = ax.imshow(prf, extent=(-15, 15, -15, 15), origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        else:
            im = ax.imshow(gauss2d(r.mu_x, r.mu_y, r[sig_outer]), extent=(-15, 15, -15, 15),
                           origin="lower", cmap="Blues")
            ax.add_patch(plt.Circle((r.mu_x, r.mu_y), r[sig_inner], fill=False, color="red",
                                    label="activation (1 σ)"))
        for k, ls in [(1, "-"), (2, "--")]:
            ax.add_patch(plt.Circle((r.mu_x, r.mu_y), k * r[sig_outer], fill=False, color="black", ls=ls))
        ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   fill=False, color="gold", lw=2, label="stimulus screen"))
        ax.set(title=f"{name} voxel {v}", xlabel="x (deg)", ylabel="y (deg)")
        ax.legend(fontsize=7, loc="upper right")
        plt.colorbar(im, ax=ax, shrink=0.7)
    plt.show()
# parameter	intuition
# eccentricity	how far from fixation the voxel looks
# σ_center / σ_activation	how big its "window" is (drives the response up)
# σ_surround	how big the annoying neighbourhood is (pushes the response down)
# σ_normalization	how big the comparison crowd is (divides the response)

# Method: how to compare models fairly
# Use a noise ceiling to know what's achievable. You measure how reliable each voxel is (the correlation between repeated sweeps) and keep only reliable voxels. The noise ceiling then tells you how good a model could be, so you can judge model performance relative to it rather than to 1. It's only an estimate, and here a conservative one: some voxels scored above it.
# Never judge models on in-sample fit. More parameters always fit the training data better (DN > DoG > CSS > Gaussian). Only cross-validation shows which extra assumptions generalize.
# Make the comparison fair: same voxels, same data, same HRF, same SGD settings. Initialize complex models from simpler ones to avoid local minima.
# Prefer simpler models unless the gain is clear. Per-voxel selection with a minimum gain avoids picking complex models because of noise. The result depends strongly on that threshold, so check several values.
# Separate the jobs: split data to compare and select, then fit on all data to interpret parameters.
# Science: what this data showed
# Compression is the main thing the Gaussian misses. n < 1 in about 96% of voxels, CSS improves about 84% of voxels on test data, and its advantage doesn't shrink out of sample.
# DN is best on average, but only just. Its lead over CSS is tiny and mostly disappears with a minimum gain of 0.01. In your fits, its normalization pool behaved like compression.
# Suppression is barely detectable here. About half the voxels have no surround, DoG helps only about half of them, and surround parameters often stayed at their starting values.
# For most voxels, the Gaussian is good enough when you require a real improvement (77% at min_gain 0.05), especially for central, small pRFs.
# All models reach about 80% of the noise ceiling. The remaining gap is larger than the differences between models, and comes from things none of them capture (HRF timing, non-retinotopic responses).
# Biggest lesson: the stimulus limits what you can learn
# The models can only be judged where the stimulus samples the pRF.
# 43% of the pRF centres lie outside the screen. The complex models' largest gains occur there, exactly where estimates are least trustworthy.
# The surround is mostly outside the screen, so suppression can't be measured properly.
# Absence of evidence here isn't evidence of absence: testing suppression needs a larger or different stimulus.
# Parameters that pile up at their starting values weren't determined by the data, so don't interpret them. Also, parameters with the same name mean different things in different models: CSS σ isn't Gaussian σ.
# For your thesis

# The same recipe applies to comparing Gaussian vs sigmoid tuning in RTMs:

# fit on training images and evaluate on test images;
# relate the gains to a noise ceiling;
# prefer the simpler model unless the gain is clear;
# refit on all data for interpretation;
# check whether your stimulus set actually samples the part of the representational space where the models differ.