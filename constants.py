"""

These are Physical constants and the UFC ground shock formulas.
Every other file in this project imports from here so the numbers
only live in one place.

Source: UFC, Eq. 2-85 (velocity) and Eq. 2-87 (acceleration).
"""

import numpy as np
from scipy.stats import norm

SEED = 42

# UFC Eq. 2-85: peak vertical velocity, V_V = K / Z_G^N  (in/sec)
K_PPV = 150.0
N_PPV = 1.5

# Reference charge weight used to convert the acceleration formula
# (Eq. 2-87) into a pure function of Z_G for this project.
REFERENCE_CHARGE_LB = 6000.0
REFERENCE_CHARGE_CUBE_ROOT = REFERENCE_CHARGE_LB ** (1.0 / 3.0)

K_ACCEL = 10000.0 / REFERENCE_CHARGE_CUBE_ROOT
N_ACCEL = 2.0

# Noise model: scatter shrinks as scaled distance grows.
# sigma(Z) = NOISE_BASE * (Z_min / Z)^NOISE_DECAY
NOISE_BASE = 0.30
NOISE_DECAY = 0.50

# z-scores for the coverage levels we check calibration against
COVERAGE_LEVELS = {
    0.50: norm.ppf(0.75),
    0.80: norm.ppf(0.90),
    0.90: norm.ppf(0.95),
    0.95: norm.ppf(0.975),
}


def peak_velocity(scaled_distance):
    """UFC Eq. 2-85: peak vertical particle velocity vs scaled distance."""
    return K_PPV * scaled_distance ** (-N_PPV)