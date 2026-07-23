import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from constants import K_PPV, N_PPV

Z_90 = norm.ppf(0.95)
Z_50 = norm.ppf(0.75)

COLOR_BLACK = "#000000"
COLOR_GREY = "#888888"
COLOR_BLUE = "#1a6faf"
COLOR_BLUE_LIGHT = "#a8cde8"
COLOR_BLUE_LIGHTER = "#d4e9f5"
COLOR_RED = "#c0392b"
COLOR_ORANGE = "#e67e22"
COLOR_GREEN = "#27ae60"
SPINE_COLOR = "#d0d0d0"

def _style_axes(ax):
    # Cleans up the plot aesthetics by removing top and right borders.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.tick_params(colors=COLOR_GREY, labelsize=9)

def plot_raw_data(data, outfile):
    # Generates Figure 1 to show the raw synthetic dataset versus the underlying true UFC curve.
    scaled_distance = data["scaled_distance"]
    observed = np.exp(data["log_observed"])
    true_velocity = data["true_velocity"]

    order = np.argsort(scaled_distance)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(scaled_distance, observed, c=COLOR_GREY, alpha=0.4, s=10,
               label="synthetic data (n=1000)", rasterized=True)
    ax.plot(scaled_distance[order], true_velocity[order], color=COLOR_BLACK, lw=2,
            label=f"UFC Eq. 2-85: V = {K_PPV:.0f} / Z^{N_PPV}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("scaled distance Z (ft / lb^(1/3))", fontsize=11)
    ax.set_ylabel("peak particle velocity V (in/sec)", fontsize=11)
    ax.set_title("ground shock training data", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.25, color=SPINE_COLOR)
    _style_axes(ax)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_gp_fit(data, mean_grid, sigma_grid, distance_grid, mean_test, sigma_test, outfile):
    # Generates Figure 2 showing the GP predictive mean, 90% confidence band, and test points.
    mean_linear = np.exp(mean_grid)
    lower_90 = np.exp(mean_grid - Z_90 * sigma_grid)
    upper_90 = np.exp(mean_grid + Z_90 * sigma_grid)

    test_distance = np.exp(data["x_test"])
    test_velocity = np.exp(data["y_test"])

    inside_90 = np.mean(
        (data["y_test"] >= mean_test - Z_90 * sigma_test) &
        (data["y_test"] <= mean_test + Z_90 * sigma_test)
    )

    true_grid = K_PPV * distance_grid ** (-N_PPV)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(distance_grid, lower_90, upper_90, color=COLOR_BLUE_LIGHT,
                     alpha=0.5, label="90% confidence band")
    ax.plot(distance_grid, mean_linear, color=COLOR_BLUE, lw=2, label="GP mean prediction")
    ax.plot(distance_grid, true_grid, color=COLOR_BLACK, lw=1.5, ls="--", label="UFC Eq. 2-85 (true)")
    ax.scatter(test_distance, test_velocity, c=COLOR_RED, s=20, alpha=0.8, zorder=5,
               label="test points (n=100)")

    ax.text(0.97, 0.95, f"90% coverage on test set: {inside_90 * 100:.1f}%",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="square,pad=0.4", facecolor="#1a1a1a", edgecolor="none"),
            color="white", family="monospace")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("scaled distance Z (ft / lb^(1/3))", fontsize=11)
    ax.set_ylabel("peak particle velocity V (in/sec)", fontsize=11)
    ax.set_title("GP forward prediction with 90% confidence band", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.25, color=SPINE_COLOR)
    _style_axes(ax)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_calibration(coverage, outfile):
    # Generates Figure 3 answering the assignment question do the 90% bands actually cover 90%.
    expected = np.array(sorted(coverage.keys()))
    observed = np.array([coverage[level] for level in expected])

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    diagonal = np.linspace(0, 1, 100)
    ax.fill_between(diagonal, np.clip(diagonal - 0.05, 0, 1), np.clip(diagonal + 0.05, 0, 1),
                     color=COLOR_GREEN, alpha=0.15, label="+/- 5% tolerance")
    ax.plot([0, 1], [0, 1], color=COLOR_GREY, lw=1.5, ls="--", label="perfect calibration")
    ax.plot(expected, observed, color=COLOR_BLUE, lw=2, label="GP calibration")
    ax.scatter(expected, observed, color=COLOR_BLUE, s=60, zorder=5)

    for level, obs in zip(expected, observed):
        ax.annotate(f"{obs * 100:.1f}%", xy=(level, obs), xytext=(6, -12),
                    textcoords="offset points", fontsize=8, color=COLOR_BLUE, family="monospace")

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("expected coverage level", fontsize=11)
    ax.set_ylabel("observed coverage on test set", fontsize=11)
    ax.set_title("Interval Calibration Check", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.25, color=SPINE_COLOR)
    _style_axes(ax)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_inversion(inversion_result, field_data, outfile):
    # Generates Figure 4 representing the Week 2 deliverable for inverse soil-parameter estimation.
    # Plots the few simulated field measurements against the recovered model curve.
    fig, ax = plt.subplots(figsize=(8, 5))
    
    scaled_dist = field_data["scaled_distance"]
    velocity = field_data["velocity"]
    
    dist_grid = np.logspace(np.log10(scaled_dist.min() * 0.5), np.log10(scaled_dist.max() * 2.0), 100)
    recovered_curve = inversion_result["recovered_k"] * dist_grid ** (-inversion_result["recovered_n"])
    true_curve = field_data["true_k"] * dist_grid ** (-field_data["true_n"])

    ax.scatter(scaled_dist, velocity, color=COLOR_ORANGE, s=40, zorder=5, label="field measurements")
    ax.plot(dist_grid, recovered_curve, color=COLOR_BLACK, lw=2, label="recovered fit")
    ax.plot(dist_grid, true_curve, color=COLOR_GREY, lw=1.5, ls="--", label="true underlying curve")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("scaled distance Z (ft / lb^(1/3))", fontsize=11)
    ax.set_ylabel("peak particle velocity V (in/sec)", fontsize=11)
    ax.set_title("Inverse Parameter Recovery", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.25, color=SPINE_COLOR)
    _style_axes(ax)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_comparison(data, mean_grid, sigma_grid, distance_grid, outfile):
    # Generates Figure 5 directly comparing the point estimates vs probabilistic bands
    # per the assignment requirements for the test scenario.
    mean_linear = np.exp(mean_grid)
    lower_95 = np.exp(mean_grid - norm.ppf(0.975) * sigma_grid)
    upper_95 = np.exp(mean_grid + norm.ppf(0.975) * sigma_grid)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(distance_grid, lower_95, upper_95, color=COLOR_BLUE_LIGHTER,
                     alpha=0.6, label="95% predictive interval")
    ax.plot(distance_grid, mean_linear, color=COLOR_BLUE, lw=2, label="probabilistic mean")
    ax.plot(distance_grid, K_PPV * distance_grid ** (-N_PPV), color=COLOR_BLACK, lw=1.5, 
            ls="--", label="deterministic point estimate")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("scaled distance Z (ft / lb^(1/3))", fontsize=11)
    ax.set_ylabel("peak particle velocity V (in/sec)", fontsize=11)
    ax.set_title("Probabilistic Prediction vs Point Estimate", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.25, color=SPINE_COLOR)
    _style_axes(ax)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)