"""

Builds the synthetic training data for Project.

The assignment says the noise should be treated as a range-dependent
band, not a fixed value, so the amount of scatter added to each point
depends on its scaled distance. Points close to the charge (small Z_G)
get more scatter, points far away get less. This mirrors how field
data behaves in practice: near-field readings are noisier because
local soil coupling effects are stronger there.
"""

import numpy as np
from constants import SEED, peak_velocity, NOISE_BASE, NOISE_DECAY


def noise_level(scaled_distance, min_distance):
    """
    Standard deviation of the log-space noise at a given scaled distance.

    Decays as a power law from the base noise level at the closest
    point in the dataset. This is the "heteroscedastic" part of the
    model: every point carries its own known noise level instead of
    one shared value.
    """
    return NOISE_BASE * (min_distance / scaled_distance) ** NOISE_DECAY


def generate_dataset(n_points=1000, seed=SEED):
    """
    Draws n_points scaled distances on a log scale between 1 and 100
    ft/lb^(1/3), computes the true UFC velocity at each one, then adds
    heteroscedastic log-normal noise to simulate field scatter.

    Returns a dict with the full dataset plus train/validation/test
    splits (80/10/10), all in log space since the GP is fit there.
    """
    rng = np.random.default_rng(seed)

    log_distance = rng.uniform(np.log(1.0), np.log(100.0), n_points)
    scaled_distance = np.exp(log_distance)

    true_velocity = peak_velocity(scaled_distance)
    min_distance = scaled_distance.min()
    sigma = noise_level(scaled_distance, min_distance)

    log_true = np.log(true_velocity)
    noise = rng.normal(0.0, sigma, n_points)
    log_observed = log_true + noise

    # shuffle and split 800 / 100 / 100
    order = rng.permutation(n_points)
    train_idx = order[:800]
    val_idx = order[800:900]
    test_idx = order[900:]

    return {
        "scaled_distance": scaled_distance,
        "true_velocity": true_velocity,
        "log_observed": log_observed,
        "sigma": sigma,
        "x_train": log_distance[train_idx],
        "y_train": log_observed[train_idx],
        "sigma_train": sigma[train_idx],
        "x_val": log_distance[val_idx],
        "y_val": log_observed[val_idx],
        "sigma_val": sigma[val_idx],
        "x_test": log_distance[test_idx],
        "y_test": log_observed[test_idx],
        "sigma_test": sigma[test_idx],
    }