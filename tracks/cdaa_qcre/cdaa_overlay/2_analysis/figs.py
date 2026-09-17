"""
Provides figure-drawing functions for the figure script.
"""

import copy
import numpy as np
import scipy as sp
import shutil

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

from utils import get_proportional_changes
from utils import get_percent_REs
from utils import get_RE_quantiles
from utils import get_min_RE_metrics
from utils import get_REs_by_num_qubits
from utils import get_device_circtype_quantiles
from utils import get_identify_shortest_accuracies


# The upstream figures enable external LaTeX for two plots. Keep that output
# when LaTeX is installed, but fall back to Matplotlib's built-in mathtext so
# the artifact can be reproduced on a normal Python environment.
_USE_TEX = shutil.which("latex") is not None


def blend_hex(hex1, hex2, alpha):
    """
    Blends colors specified by hex code strings.

    Returns (list[float]): RGB vector of blended colors.

    Args:
        hex1 (str): Hex code string for color 1.

        hex2 (str): Hex code string for color 2.

        alpha (float): Blend proportion. between 0 and 1. Blended according to
            linear interpolation: alpha * hex1 + (1-alpha) * hex2
    """

    color_hexes = [hex1, hex2]
    color_rgbs = []

    # Convert hex to rgb
    for color_hex in color_hexes:
        color_hex = color_hex.lstrip("#")
        color_rgb = [int(color_hex[i : i + 2], 16) / 255 for i in (0, 2, 4)] + [1]
        color_rgbs.append(color_rgb)

    # Blend rgb
    blend_rgb = []
    for rgb1_val, rgb2_val in zip(color_rgbs[0], color_rgbs[1]):
        blend_rgb += [(alpha * rgb1_val) + ((1 - alpha) * rgb2_val)]

    return blend_rgb


def plot_case_study(data, metrics, comp_metric):
    """
    Plots the delta-Ds and delta-R for a single case-study comparison.

    In this case, we use the 75th highest out of 90 total traditional depth errors to show
    a worst-case scenario users may encounter. Using the original data, this corresponds to
    comparing the Qiskit-compiled version of the QFT64 against the SQGM-compiled version.

    Args:
        data (np.array): The metric values for all circuit versions compiled for a single device.

        metrics (list[str]): The names of the metrics to associate with the data values.

        comp_metric (str): The name of the metric to compare against trad and multi-qubit depth.
    """

    # Recreate the new view used internally by get_RE_quantiles()
    prop_changes = get_proportional_changes(data)
    percent_REs = get_percent_REs(prop_changes)

    # Switch percent_REs axes to:
    #   - Axis 0: Metrics
    #   - Axis 1: Compiled circuit version pairs
    new_view = np.moveaxis(percent_REs, [0, 1], [1, 0])

    # Find the circuit and compiler pair for which trad depth %RE is 75th worst
    trad_metric_pos = 2 * (len(metrics) - 1) - 1
    sorted_trad_errors = np.sort(new_view[trad_metric_pos])

    # TODO: Bug if multiple pairs have the same error for the specified metric. In that case,
    # only the first pair is returned. Note that other pairs may have different errors in the
    # other, non-specified metrics
    pair_position = np.where(new_view[trad_metric_pos] == sorted_trad_errors[75])[0][0]

    # Specify positions of the metrics along their axis in prop_changes
    trad_metric_pos = len(metrics) - 2
    multi_metric_pos = 0
    comp_metric_pos = metrics.index(comp_metric)
    runtime_pos = len(metrics) - 1

    # Find the prop changes for that circuit and compiler pair
    # In prop change data, circuit and compiler pairs are rows, so we use
    # circ_comp_pair_pos for the row index
    trad_prop_change = prop_changes[pair_position][trad_metric_pos]
    multi_prop_change = prop_changes[pair_position][multi_metric_pos]
    comp_prop_change = prop_changes[pair_position][comp_metric_pos]
    runtime_prop_change = prop_changes[pair_position][runtime_pos]

    # Initialize plot
    fig, ax = plt.subplots(dpi=300)
    plt.rcParams["text.usetex"] = _USE_TEX

    # Add bars
    x = [
        "Traditional depth",
        "Multi-qubit depth",
        "Gate-aware depth",
    ]
    y = [
        trad_prop_change,
        multi_prop_change,
        comp_prop_change,
    ]

    bar_list = ax.bar(x, y, alpha=0.65)

    # Set bar colors
    for i in range(2):
        bar_list[i].set_color("grey")
    bar_list[2].set_color("green")

    # Add bar labels
    for i in range(len(x)):
        plt.text(i, y[i] / 2 - 0.005, f"{y[i]:#.4g}", ha="center")

    # Add line for ground truth runtime change
    ax.plot(
        [-0.5, 2.5],
        [runtime_prop_change, runtime_prop_change],
        "--",
        color="green",
        label=rf"True relative runtime difference ($\Delta R$): {runtime_prop_change:#.4g}",
    )

    # Adjust bounding box and axes
    ax.spines[["top", "right"]].set_color(
        "none"
    )  # Hide top and right spine (axis/bounding box lines)
    ax.spines["bottom"].set_position(("data", 0))  # Place axis at y=0

    # Add labels & legend
    ax.xaxis.set_ticks_position("top")  # Show labels on top of plot
    ax.tick_params(axis="x", top=False)  # Hide top ticks

    ax.set_ylabel(r"Difference relative to to SQGM ($\Delta D$)")
    ax.legend()

    # Save plot
    plt.savefig("./out/case-study.png", bbox_inches="tight")

    # Close plot
    plt.rcParams["text.usetex"] = False
    plt.close()


def plot_RE_boxplots(datas, metrics):
    """
    Plots boxplots of percent relative error (%RE) of circuit comparisons on each device.

    Args:
        datas (dict[str: np.array]): Mapping from device names to device raw data.

        metrics (list[str]): The names of the metrics to associate with the data values.
    """

    fig, ax = plt.subplots(dpi=300)

    # Store default color and hatch list
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    hatches = [
        "///",
        "\\\\\\",
        "+++",
        "xxx",
        "...",
        "ooo",
    ]
    plt.rcParams["hatch.linewidth"] = 0.3

    for i, (device_name, device_data) in enumerate(datas.items()):

        # Collect data for plotting
        prop_changes = get_proportional_changes(device_data)
        percent_REs = get_percent_REs(prop_changes)

        # Switch percent_REs axes to:
        #   - Axis 0: Metrics
        #   - Axis 1: Compiled circuit version pairs
        new_view = np.moveaxis(percent_REs, [0, 1], [1, 0])

        data = [new_view[-1], new_view[len(metrics) - 1], new_view[-2]]

        # Obtain colors for plotting
        color = colors[i]
        fill_color = blend_hex(color, "#FFFFFF", 0.5)
        hatch = hatches[i]

        # Draw boxplots
        ax.boxplot(
            data,
            positions=[i, i + 7, i + 14],
            label=device_name,
            patch_artist=True,
            boxprops=dict(facecolor=fill_color, color=color, hatch=hatch),
            medianprops=dict(color=color),
            flierprops=dict(markeredgecolor=color),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
        )

    ax.set_xticks(
        [7 * x + 2.5 for x in range(3)],
        labels=["Traditional depth", "Multi-qubit depth", "Gate-aware depth"],
    )
    ax.tick_params(axis="x", bottom=False)

    ax.set_ylabel(r"% Relative error")
    ax.set_yscale("asinh")
    ax.set_ylim((-0.3, 5 * 10e3))
    ax.legend()

    plt.savefig("./out/REs-by-metric.png", bbox_inches="tight")
    plt.rcParams["hatch.linewidth"] = 1.0
    plt.close()


def plot_REs_by_weight(datas, metrics, weights):
    """
    Plots the median percent relative error (%RE) for gate-aware depth against the
    manually-set weight parameter for all devices.

    Args:
        datas (dict[str: np.array]): Mapping from device names to device raw data.

        metrics (list[str]): The names of the metrics to associate with the data values.

        weights (list[float]): The weight parameter used for gate-aware depth.
    """

    best_metrics = get_min_RE_metrics(datas, metrics, manual_weights_only=True)

    # Initialize plot
    fig, ax = plt.subplots(dpi=300)

    # Annotate parameter values matching the empirical weight maps
    ax.axvline(
        x=0.0942,
        color="grey",
        alpha=0.5,
        linewidth=3,
    )
    ax.annotate(
        xy=(0.105, 35),
        text="$w_s = 0.0942$",
        va="top",
        ha="left",
        fontsize=8,
    )
    ax.axvline(
        x=0.483,
        color="grey",
        alpha=0.5,
        linewidth=3,
    )
    ax.annotate(
        xy=(0.495, 35),
        text="$w_s = 0.483$",
        va="top",
        ha="left",
        fontsize=8,
    )

    # Set line attribute cycles
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    lines = [
        (0, ()),
        (0, (3, 2)),
        (0, (1, 1)),
        (0, (9, 1, 3, 1)),
        (0, (7, 1, 1, 1)),
        (0, (5, 1, 1, 1, 1, 1)),
        (0, (5, 2, 1, 2)),
    ]

    # Draw median %RE line for each device
    for i, (device_name, device_data) in enumerate(datas.items()):

        quantiles = get_RE_quantiles(device_data)

        # Switch quantiles axes to:
        #   - Axis 0: Quantiles (q0, q1, q2, q3, q4)
        #   - Axis 1: Metrics (see get_RE_quantiles for array structure)
        new_view = np.moveaxis(quantiles, [0, 1], [1, 0])

        # Create new %RE quantile array to manipulate for graphing
        # Slice from 0.5*new_view.shape[1] to eliminate first half, where abs errors are stored
        # Stop slice at -2 to eliminate gate_aware_depth_w=avg and trad_depth
        # Use j in [0, 2, 4] to read min, median, and max
        [min_RE, med_RE, max_RE] = [
            copy.deepcopy(new_view[j][int(0.5 * new_view.shape[1]) : -2])
            for j in [0, 2, 4]
        ]

        # Obtain best weight parameter for error bar construction.
        # Offset by 1 because metrics include multi-qubit at the start, while weights do not
        # include the corresponding w=0
        best_weight_pos = metrics.index(best_metrics[device_name])
        best_weight = weights[best_weight_pos - 1]

        # For each quantile, stop 2 before end to eliminate gate-aware and trad depth %RE.
        # Multi-qubit %RE is left in at the beginning because it is equal to
        # gate_aware_depth_w=0.
        x = [0] + weights

        # Add median %RE
        ax.plot(x, med_RE, color=colors[i], linestyle=lines[i], label=device_name)

        # Annotate best weight
        ax.vlines(
            x=best_weight,
            ymin=2e-2,
            ymax=0.8 * med_RE[best_weight_pos],
            color=colors[i],
            alpha=0.4,
            linewidth=0.75,
        )
        ax.annotate(
            xy=(best_weight + 0.015, 0.83 * med_RE[best_weight_pos]),
            text=f"$w_s = {best_weight}$",
            va="top",
            fontsize=8,
            color=colors[i],
            alpha=0.8,
        )

    # Adjust axis scale
    ax.set_yscale("log")

    # Add labels and legend
    plt.rcParams["text.usetex"] = _USE_TEX
    ax.set_xlabel("Non-$RZ$ single-qubit gate weight ($w_s$)")
    plt.rcParams["text.usetex"] = False

    ax.set_ylabel(r"Median % relative error")

    ax.legend()

    # Save plot
    plt.savefig("./out/RE-v-manual-weight.png", bbox_inches="tight")

    # Close plot
    plt.close()


def plot_RE_quantiles(datas, metrics, weights):
    """
    DEPRECATED

    Plots the 1st, 2nd, and 3rd quartile percent relative error (%RE) quantiles of
    each metric on all devices. Uses manually-set weight maps for gate-aware depth.

    Args:
        datas (dict[str: np.array]): Mapping from device names to device raw data.

        metrics (list[str]): The names of the metrics to associate with the data values.

        weights (list[float]): The weight parameter used for gate-aware depth.
    """

    # Initialize plot
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

    legend_marker_handles = []
    legend_labels = []

    num_devices = len(datas)

    # Draw quartiles for each device
    for i, (device_name, device_data) in enumerate(datas.items()):

        quantiles = get_RE_quantiles(device_data)

        # Switch quantiles axes to:
        #   - Axis 0: Quantiles (q0, q1, q2, q3, q4)
        #   - Axis 1: Metrics (see get_RE_quantiles for array structure)
        new_view = np.moveaxis(quantiles, [0, 1], [1, 0])

        # Create new %RE quantile array to manipulate for graphing
        # Slice from 0.5*new_view.shape[1] to eliminate first half, where abs errors are stored
        # Use range(1-4) to avoid reading min and max from 5-number summary
        [q1, q2, q3] = [
            copy.deepcopy(new_view[j][int(0.5 * new_view.shape[1]) :])
            for j in range(1, 4)
        ]

        # For each quantile, duplicate multi-qubit %RE, since it is equal to gate_aware_depth_w=0
        # and we will therefore want to plot that point for both multi-qubit and gate-aware depth
        # Multi-qubit %RE is stored in 0th column.
        [q1, q2, q3] = [np.insert(qn, 1, qn[0]) for qn in [q1, q2, q3]]

        # Rotate trad %RE to first column
        [q1, q2, q3] = [np.roll(qn, 1) for qn in [q1, q2, q3]]

        x = [-2 * num_devices, -1 * num_devices, 0] + weights

        # Add gate-aware plots
        (med,) = ax.plot(x[2:], q2[2:], "-")
        mid50 = ax.fill_between(x[2:], q1[2:], q3[2:], alpha=0.2)

        med_color = med.get_color()

        # Add legend entries
        legend_marker_handles.append((med, mid50))
        legend_labels.append(device_name)

        # "Box" plots for trad and multi depth

        x_group_range = 0.25
        gap = 0.05

        # Add trad "box" plot

        # Plot line over mid 50% range
        x_pos = -2 * (x_group_range + gap) + (x_group_range * i / num_devices)
        ax.vlines(
            x=x_pos,
            ymin=q1[0],
            ymax=q3[0],
            color=med_color,
            alpha=0.2,
        )
        # Plot dot over median
        ax.plot([x_pos], [q2[0]], ".", color=med_color)

        # Annotate Marrakesh example
        if device_name == "IBM Marrakesh":
            width = 0.04
            height = 25
            box = Rectangle(
                (x_pos - 0.5 * width, q2[0] - 0.43 * height),
                width,
                height,
                fill=False,
                edgecolor="black",
            )
            ax.add_patch(box)
            ax.annotate(
                "QAOA16 example",
                xy=(x_pos + 0.01, q2[0] + 0.57 * height),
                xytext=(x_pos + 0.03, q2[0] * 7),
                arrowprops=dict(
                    facecolor="black",
                    shrink=0.1,
                    width=0.1,
                    headwidth=3,
                    headlength=3,
                ),
            )

        # Add multi "box" plot

        # Plot line over mid 50% range
        x_pos = -1 * (x_group_range + gap) + (x_group_range * i / num_devices)
        ax.vlines(
            x=x_pos,
            ymin=q1[1],
            ymax=q3[1],
            color=med_color,
            alpha=0.2,
        )
        # Plot dot over median
        ax.plot([x_pos], [q2[1]], ".", color=med_color)

    # Add axes and legend

    # Create numbered x-ticks (primary)
    ax.set_xlabel("\nDepth")
    ax.set_xticks([x / 5 for x in range(0, 6)])

    # Create invisible x-ticks for text metric labels (secondary)
    sec_ax = ax.secondary_xaxis(location=0)
    sec_ax.set_xticks(
        [-0.4958, -0.1958, 0.5],
        labels=["\nTraditional", "\nMulti-qubit", "\nGate-aware (by weight)"],
    )
    sec_ax.tick_params(axis="x", color="white")

    ax.set_ylabel(r"% Relative error (%RE)")
    ax.set_yscale("log")
    ax.legend(tuple(legend_marker_handles), legend_labels)

    # Save plot
    plt.savefig("./out/RE-quantiles.png", bbox_inches="tight")

    # Close plot
    plt.close()


def plot_identify_shortest_accuracies(datas, metrics):
    """
    Plots each depth metric's accuracy at identifying the circuit version(s) with
    the shortest estimated runtime for each device.

    Args:
        datas (dict[str: np.array]): Mapping from device names to device raw data.

        metrics (list[str]): The names of the metrics to associate with the data values.
    """

    shared_metrics = {
        "IBM Sherbrooke": "gate_aware_depth_w=avg",
        "IBM Kyiv": "gate_aware_depth_w=avg",
        "IBM Brisbane": "gate_aware_depth_w=avg",
        "IBM Marrakesh": "gate_aware_depth_w=avg",
        "IBM Kingston": "gate_aware_depth_w=avg",
        "IBM Aachen": "gate_aware_depth_w=avg",
    }

    device_accuracies = get_identify_shortest_accuracies(datas, metrics, shared_metrics)

    # Initialize plot
    fig, ax = plt.subplots(dpi=300)

    # Create bar groups
    depth_labels = ("Traditional depth", "Multi-qubit depth", "Gate-aware depth")
    x = np.arange(len(depth_labels))  # Label locations
    width = 0.14  # Bar widths
    multiplier = 0

    # Set bar attribute cycles
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    hatches = [
        "///",
        "\\\\\\",
        "+++",
        "xxx",
        "...",
        "ooo",
    ]
    plt.rcParams["hatch.linewidth"] = 0.3

    # Draw bars for each device
    for i, (device_name, accuracy) in enumerate(device_accuracies.items()):

        color = colors[i]
        fill_color = blend_hex(color, "#FFFFFF", 0.65)
        hatch = hatches[i]

        offset = width * multiplier
        rects = ax.bar(
            x + offset,
            accuracy,
            width,
            label=device_name,
            facecolor=fill_color,
            edgecolor=color,
            hatch=hatch,
        )
        ax.bar_label(
            rects, fmt="{:#.4g}", padding=-30, rotation="vertical", color="white"
        )
        multiplier += 1

    # Add labels and legend
    ax.set_xticks(x + 2.5 * width, depth_labels)
    ax.set_ylabel(r"% Correct identifications")
    ax.legend(ncols=3, bbox_to_anchor=(0.5, 1.17), loc="upper center")

    # Save plot
    plt.savefig("./out/identify-shortest-accuracies.png", bbox_inches="tight")

    # Close Plot
    plt.rcParams["hatch.linewidth"] = 1.0
    plt.close()


def plot_REs_by_num_qubits(datas, metrics):
    """
    Plots boxplots of percent relative error (%RE) against number of qubits
    in circuit, aggregated across devices.

    Args:
        datas (dict[str: np.array]): Mapping from device names to device raw data.

        metrics (list[str]): The names of the metrics to associate with the data values.
    """

    # Link circuits of the same size
    # key: value = num_qubits: circuit_position in CSV, from top to bottom
    circ_positions_by_size = {
        4: [9, 2, 5, 9, 13],
        8: [11, 3, 6, 11, 14],
        16: [7, 0, 4, 7, 12],
        32: [8, 1, 8],
        64: [10],
    }

    # Obtain data
    REs_by_num_qubits = get_REs_by_num_qubits(datas, metrics, circ_positions_by_size)

    # Set boxplot attribute cycles
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    hatches = [
        "///",
        "\\\\\\",
        "+++",
        "xxx",
        "...",
        "ooo",
    ]
    plt.rcParams["hatch.linewidth"] = 0.3

    legend_marker_handles = []
    plotted_metrics = ["Traditional depth", "Multi-qubit depth", "Gate-aware depth"]

    # Initialize plot
    fig, ax = plt.subplots(dpi=300)

    # Draw boxplots for each metric
    for metric_pos, metric in enumerate(plotted_metrics):

        # Obtain colors for hatching
        color = colors[metric_pos]
        fill_color = blend_hex(color, "#FFFFFF", 0.5)
        hatch = hatches[metric_pos]

        # Draw boxplot for each size
        for j, num_qubits in enumerate(REs_by_num_qubits.keys()):
            data = REs_by_num_qubits[num_qubits][metric_pos]

            # Draw boxplots
            ax.boxplot(
                data,
                positions=[(metric_pos * 6) + j],
                patch_artist=True,
                boxprops=dict(facecolor=fill_color, color=color, hatch=hatch),
                medianprops=dict(color=color),
                flierprops=dict(markeredgecolor=color),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
                widths=[0.5],
            )

            # Store first boxplot style for legend
            if j == 0:
                patch = mpatches.Patch(
                    facecolor=fill_color, edgecolor=color, hatch=hatch
                )
                legend_marker_handles.append(patch)

    # Adjust axes
    ax.set_yscale("asinh")
    ax.set_ylim((-0.3, 5 * 10e3))

    # Add labels and legend
    ax.set_xticks(
        [x for x in range(17) if (x + 1) % 6 != 0],
        labels=[4, 8, 16, 32, 64] * 3,
    )

    sec = ax.secondary_xaxis(location=0)
    sec.set_xticks(
        [6 * 1 + 2],
        labels=["\n# Qubits in Circuit"],
    )

    ax.set_ylabel(r"% Relative error")
    ax.legend(tuple(legend_marker_handles), plotted_metrics)

    # Save plot
    plt.savefig("./out/REs-by-num-qubits.png", bbox_inches="tight")
    plt.rcParams["hatch.linewidth"] = 1.0

    # Close plot
    plt.close()


def plot_device_circtype_size_grid(datas, metrics):
    """
    Creates a grid of plots showing percent relative error (%RE) against number of
    qubits, disaggregated by circuit type and device.

    Each subplot in the grid plots median %RE against number of qubits for a single
    circuit type, on a single device, using all depth metrics. The median %RE is
    taken over the compiler comparison pairs for that circuit type at each size.

    Args:
        datas (dict[str: np.array]): Mapping from device names to device raw data.

        metrics (list[str]): The names of the metrics to associate with the data values.
    """

    # Obtain data
    type_pos_map = {
        "hamilsim": [2, 3, 0, 1],
        "qaoa": [5, 6, 4],
        "qft": [9, 11, 7, 8, 10],
        "vqe": [13, 14, 12],
    }

    device_circtype_quantiles = get_device_circtype_quantiles(
        datas, metrics, type_pos_map
    )

    metric_list = ["Traditional depth", "Multi-qubit depth", "Gate-aware depth"]
    num_circtypes = len(type_pos_map.keys())
    num_devices = len(datas.keys())
    log2_circ_sizes = [2, 3, 4, 5, 6]

    # Initialize plot
    fig, axs = plt.subplots(
        num_circtypes, num_devices, sharex=True, sharey=True, dpi=300
    )

    # For each circuit type and device, draw subplot
    for i, circ_type_name in enumerate(type_pos_map.keys()):
        for j, device_name in enumerate(datas.keys()):

            # Retrieve data for one circuit type and device pair
            ax_data = device_circtype_quantiles[(device_name, circ_type_name)]

            # For each metric, draw median %RE line
            for metric_pos in range(ax_data.shape[0]):

                # Store first subplot line styles for legend
                if i == 0 and j == 0:
                    label = metric_list[metric_pos]
                else:
                    label = "_nolegend_"

                # Draw line
                q2s = ax_data[metric_pos][2]
                axs[i, j].plot(
                    log2_circ_sizes[
                        : len(q2s)
                    ],  # Stop early if no data for largest sizes
                    q2s,
                    marker=".",
                    linestyle="-",
                    label=label,
                )

            # Adjust subplot axis
            axs[i, j].set_yscale("asinh")

            # Add subplot labels
            axs[i, j].tick_params(axis="both", which="major", labelsize=6)
            axs[i, j].set_xticks(log2_circ_sizes[::2], ["$2^2$", "$2^4$", "$2^6$"])
            axs[i, j].set_yticks([0, 1, 10, 100, 1000])

            # Set grid row and column labels
            if i == 0:
                axs[i, j].set_title(device_name, fontsize=6, fontweight="bold")
            if j == 0:
                axs[i, j].set_ylabel(circ_type_name, fontsize=6, fontweight="bold")

    # Add plot labels and legend
    fig.legend(ncols=3, loc="upper center")
    fig.supxlabel("# Qubits")
    fig.supylabel("% Relative error")

    # Save plot
    plt.savefig("./out/REs-by-num-qubits-grid.png", bbox_inches="tight")

    # Close plot
    plt.rcParams["hatch.linewidth"] = 1.0
    plt.close()


def plot_abs_runtime_v_metrics(data, metrics):
    """
    Plot absolute runtime against depth metrics and obtain linear regression.

    Args:
        data (np.array): The metric values for all circuit versions compiled for a single device.

        metrics (list[str]): The names of the metrics to associate with the data values.
    """

    trad_position = metrics.index("trad_depth")
    multi_position = metrics.index("multi_qubit_depth")
    gate_aware_position = metrics.index("gate_aware_depth_w=avg")
    runtime_position = metrics.index("runtime")

    # Switch percent_REs axes to:
    #   - Axis 0: Circuit
    #   - Axis 1: Metric
    #   - Axis 2: Compiler (or compiled version of circuit)
    new_view = np.moveaxis(data, [1, 2], [2, 1])

    metric_labels = ["Traditional depth", "Multi-qubit depth", "Gate-aware depth"]
    markers = ["o", "+", "s", "x", "v", "1", "d", "2", "*", "3", "d"]

    # Initialize plot
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(13, 4), dpi=300)

    # For each metric, create scatterplot
    for i, metric_position in enumerate(
        [trad_position, multi_position, gate_aware_position]
    ):

        # Plot metric against runtime, grouped/colored by base circuit
        for circ_position in range(new_view.shape[0]):

            new_x_vals = new_view[circ_position][metric_position]
            new_y_vals = new_view[circ_position][runtime_position]

            # Draw scatterplot
            ax[i].scatter(
                new_x_vals, new_y_vals, marker=markers[circ_position % len(markers)]
            )

            # Collect points for overall regression
            if circ_position == 0:
                x = new_x_vals
                y = new_y_vals
            else:
                x = np.concatenate((x, new_x_vals))
                y = np.concatenate((y, new_y_vals))

        # Construct regression line
        result = sp.stats.linregress(x, y)
        b = result.intercept
        m = result.slope
        r_squared = (result.rvalue) ** 2
        ax[i].axline(xy1=(0, b), slope=m)
        ax[i].annotate(
            f"slope = ${m:.2e}$\nintercept = ${b:.2e}$\n$R^2 = {r_squared:.10f}$",
            (8000, 0),
            fontsize="x-small",
        )

        # Add subplot label
        ax[i].set_xlabel(metric_labels[i])

    # Add plot label
    ax[0].set_ylabel(r"Runtime (s)")

    fig.suptitle("Depth v. runtime for IBM Marrakesh circuits")

    # Save plot
    plt.savefig("./out/abs-runtime-v-metrics.png", bbox_inches="tight")

    # Close plot
    plt.close()


def plot_prop_runtime_v_metrics(data, metrics):
    """
    Plot relative runtime difference against relative metric difference and obtain a
    linear regression.

    Args:
        data (np.array): The metric values for all circuit versions compiled for a single device.

        metrics (list[str]): The names of the metrics to associate with the data values.
    """

    trad_position = metrics.index("trad_depth")
    multi_position = metrics.index("multi_qubit_depth")
    gate_aware_position = metrics.index("gate_aware_depth_w=avg")
    runtime_position = metrics.index("runtime")

    prop_changes = get_proportional_changes(data)

    # Switch percent_REs axes to:
    #   - Axis 0: Circuit
    #   - Axis 1: Metric
    #   - Axis 2: Compiler (or compiled version of circuit)
    # new_view = np.moveaxis(data, [1,2],[2,1])

    # Set line style and label cycles
    metric_labels = ["Traditional depth", "Multi-qubit depth", "Gate-aware depth"]
    comp_pair_labels = ["SQGM", "Qiskit"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    lines = [
        (0, ()),
        (0, (3, 2)),
        (0, (1, 1)),
        (0, (9, 1, 3, 1)),
        (0, (7, 1, 1, 1)),
        (0, (5, 1, 1, 1, 1, 1)),
        (0, (5, 2, 1, 2)),
    ]
    markers = ["o", "+", "s", "x", "v", "1", "d", "2", "*", "3", "d"]

    # Initialize plot
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(13, 4), dpi=300)

    # For each metric, draw scatterplot
    for i, metric_position in enumerate(
        [trad_position, multi_position, gate_aware_position]
    ):

        # Draw x- and y-axis lines
        ax[i].axvline(x=0, lw=0.5, color="black", zorder=-2)
        ax[i].axhline(y=0, lw=0.5, color="black", zorder=-2)

        sqgm_v_sabre_offset = 0
        qiskit_v_sabre_offset = 2

        # For each compiler pair, draw scatterplot
        for j, compiler_pair_offset in enumerate(
            [sqgm_v_sabre_offset, qiskit_v_sabre_offset]
        ):

            # Obtain style and label from cycles
            color = colors[j % 2]
            linestyle = lines[j % 2]
            marker = markers[j % 2]
            label = comp_pair_labels[j % 2]

            # For each circuit, gather one data point
            for circ_position in range(data.shape[0]):

                circ_comp_pair_pos = circ_position * 6 + compiler_pair_offset

                delta_D = prop_changes[circ_comp_pair_pos][metric_position]
                delta_R = prop_changes[circ_comp_pair_pos][runtime_position]

                # Collect points for overall plot and regressionregression
                if circ_position == 0:
                    x = np.array([delta_D])
                    y = np.array([delta_R])
                else:
                    x = np.concatenate((x, np.array([delta_D])))
                    y = np.concatenate((y, np.array([delta_R])))

            # Draw compiler pair scatterplot
            ax[i].scatter(x, y, marker=marker)

            # Construct compiler pair regression line
            result = sp.stats.linregress(x, y)
            b = result.intercept
            m = result.slope
            r_squared = (result.rvalue) ** 2
            ax[i].axline(
                xy1=(0, b),
                slope=m,
                color=color,
                label=f"{label}\nslope = ${m:.2e}$",
                linestyle=linestyle,
                zorder=-1,
            )

        # Draw ideal regression line
        ax[i].axline(
            xy1=(0, 0),
            slope=1,
            color="black",
            label="Ideal\nslope = $1.00 e + 00$",
            linestyle=lines[2],
            zorder=-1,
        )

        # Add subplot labels and legend
        ax[i].set_xlabel(metric_labels[i], fontsize="small")
        ax[i].legend(fontsize="x-small")

    # Add figure labels
    fig.supylabel(r"$\Delta R$ relative to SABRE", x=0.07)
    fig.supxlabel(r"$\Delta D$ relative to SABRE", y=-0.05)
    fig.suptitle(r"$\Delta R$ v. $\Delta D$ for IBM Marrakesh circuits")

    # Save plot
    plt.savefig("./out/rel-runtime-v-metrics.png", bbox_inches="tight")

    # Close plot
    plt.close()
