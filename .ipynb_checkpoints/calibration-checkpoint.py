"""
calibration.py

Checks whether the GP's confidence intervals are trustworthy: does
the 90% band actually contain the true value 90% of the time on data
the model has not seen? This is the calibration test the assignment
asks for in Week 1.
"""

import numpy as np
from constants import COVERAGE_LEVELS


def check_coverage(mean, sigma, y_true):
    """
    For each coverage level (50%, 80%, 90%, 95%), builds the interval
    [mean - z*sigma, mean + z*sigma] and measures what fraction of
    y_true actually falls inside it. A well calibrated model should
    get an observed fraction close to the expected level.
    """
    coverage = {}
    for level, z_score in COVERAGE_LEVELS.items():
        lower = mean - z_score * sigma
        upper = mean + z_score * sigma
        inside = np.mean((y_true >= lower) & (y_true <= upper))
        coverage[level] = float(inside)
    return coverage


def print_coverage_report(coverage, tolerance=0.05):
    """Prints a pass/fail table against the expected coverage levels."""
    for level, observed in sorted(coverage.items()):
        passed = abs(observed - level) <= tolerance
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] expected {level * 100:4.0f}%   observed {observed * 100:5.1f}%")