"""
server.py

Thin Flask layer over the existing Project 3 code. Nothing about the
model, the noise formula, or the inversion routine is reimplemented
here — this file only imports gp_model.py, calibration.py,
constants.py, dataset.py, and inversion.py exactly as they are and
exposes them over HTTP so the front end has something to call.

On startup it runs phases 1-2 once (generate data, fit the GP) and
keeps the fitted GP and the held-out test set in memory, the same
split main.py already uses. Every request after that is just a
predict() or an inversion call against state that's already fitted.

i have to Run with: python server.py
Then open http://localhost:5000
"""

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from constants import SEED, K_PPV, N_PPV
from dataset import generate_dataset
from gp_model import HeteroscedasticGP
from calibration import check_coverage
from inversion import generate_field_data, recover_parameters

app = Flask(__name__, static_folder=".", static_url_path="")


# Startup: phases 1-2 from main.py, run once and cached in memory.

print("Generating dataset and fitting GP once at startup, this takes a few seconds...")
DATA = generate_dataset(n_points=1000, seed=SEED)

GP = HeteroscedasticGP()
GP.fit(DATA["x_train"], DATA["y_train"], DATA["sigma_train"], n_restarts=5)

MEAN_TEST, SIGMA_TEST = GP.predict(DATA["x_test"], DATA["sigma_test"])
COVERAGE = check_coverage(MEAN_TEST, SIGMA_TEST, DATA["y_test"])

# a smooth curve across the full Z range, reused by every /predict
# call and by the initial page load, same as main.py's plotting grid
_LOG_Z_MIN = np.log(DATA["scaled_distance"].min())
_LOG_Z_MAX = np.log(DATA["scaled_distance"].max())
_MEAN_SIGMA = float(DATA["sigma"].mean())


def _z_scores():
    # 50/80/90/95% two-sided normal z-scores, same levels calibration.py checks
    from scipy.stats import norm
    return {
        0.50: norm.ppf(0.75),
        0.80: norm.ppf(0.90),
        0.90: norm.ppf(0.95),
        0.95: norm.ppf(0.975),
    }


Z = _z_scores()


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/calibration-curve")
def calibration_curve():
    """
    Returns everything the page needs on load: the raw scatter, the
    true UFC curve, the GP mean + bands across the full Z range, the
    test points used for calibration, and the calibration table
    itself. This is main.py phases 1-3, packaged as JSON instead of
    matplotlib figures.
    """
    x_grid = np.linspace(_LOG_Z_MIN, _LOG_Z_MAX, 200)
    z_grid = np.exp(x_grid)
    sigma_grid_input = np.full_like(x_grid, _MEAN_SIGMA)
    mean_grid, sigma_grid = GP.predict(x_grid, sigma_grid_input)

    bands = {}
    for level, z in Z.items():
        bands[str(level)] = {
            "lower": np.exp(mean_grid - z * sigma_grid).tolist(),
            "upper": np.exp(mean_grid + z * sigma_grid).tolist(),
        }

    # subsample the raw scatter so the payload stays small; 1000 points
    # of context is more than a browser chart needs to show the shape
    scatter_idx = np.linspace(0, len(DATA["scaled_distance"]) - 1, 300).astype(int)

    return jsonify({
        "z_grid": z_grid.tolist(),
        "mean_grid": np.exp(mean_grid).tolist(),
        "true_grid": (K_PPV * z_grid ** (-N_PPV)).tolist(),
        "bands": bands,
        "scatter": {
            "z": DATA["scaled_distance"][scatter_idx].tolist(),
            "v": np.exp(DATA["log_observed"][scatter_idx]).tolist(),
        },
        "test_points": {
            "z": np.exp(DATA["x_test"]).tolist(),
            "v": np.exp(DATA["y_test"]).tolist(),
        },
        "coverage": {str(k): v for k, v in COVERAGE.items()},
    })


@app.route("/api/predict")
def predict():
    """
    Week 1 deliverable as a single query: given one scaled distance
    from the user, return the GP's mean PPV plus 50/80/90/95% bands,
    and where that query sits on the fitted noise curve for context.
    """
    try:
        z_query = float(request.args.get("z", ""))
    except (TypeError, ValueError):
        return jsonify({"error": "z must be a number"}), 400

    if z_query <= 0:
        return jsonify({"error": "scaled distance must be positive"}), 400

    log_z = np.log(np.array([z_query]))
    sigma_q = np.array([_MEAN_SIGMA])
    mean, sigma = GP.predict(log_z, sigma_q)
    mean, sigma = float(mean[0]), float(sigma[0])

    intervals = {}
    for level, z in Z.items():
        intervals[str(level)] = {
            "lower": float(np.exp(mean - z * sigma)),
            "upper": float(np.exp(mean + z * sigma)),
        }

    return jsonify({
        "z_query": z_query,
        "mean_velocity": float(np.exp(mean)),
        "true_ufc_velocity": float(K_PPV * z_query ** (-N_PPV)),
        "intervals": intervals,
        "in_training_range": bool(np.exp(_LOG_Z_MIN) <= z_query <= np.exp(_LOG_Z_MAX)),
    })


@app.route("/api/invert", methods=["POST"])
def invert():
    """
    Week 2 deliverable: recover n and k from field measurements using
    inversion.recover_parameters exactly as written. If the request
    body includes points, those are used (log-space velocity and a
    flat 0.25 measurement noise, matching what a user would plausibly
    supply for a handful of field readings). Otherwise it falls back
    to inversion.generate_field_data with a request-supplied seed, the
    same simulated scenario main.py runs.
    """
    body = request.get_json(silent=True) or {}
    seed = int(body.get("seed", 99))
    custom_points = body.get("points")

    if custom_points:
        try:
            z = np.array([float(p["z"]) for p in custom_points])
            v = np.array([float(p["v"]) for p in custom_points])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "each point needs numeric z and v"}), 400

        if len(z) < 2:
            return jsonify({"error": "need at least 2 field points to invert"}), 400
        if np.any(z <= 0) or np.any(v <= 0):
            return jsonify({"error": "z and v must be positive"}), 400

        field_data = {
            "scaled_distance": z,
            "velocity": v,
            "log_velocity": np.log(v),
            "true_n": None,
            "true_k": None,
        }
    else:
        field_data = generate_field_data(seed=seed, n_points=7)

    result = recover_parameters(field_data, n_epochs=300, learning_rate=0.01)

    dist_grid = np.logspace(
        np.log10(field_data["scaled_distance"].min() * 0.5),
        np.log10(field_data["scaled_distance"].max() * 2.0),
        100,
    )
    recovered_curve = result["recovered_k"] * dist_grid ** (-result["recovered_n"])
    ufc_curve = K_PPV * dist_grid ** (-N_PPV)

    response = {
        "recovered_n": result["recovered_n"],
        "recovered_k": result["recovered_k"],
        "loss_history": result["loss_history"][::5],  # thin for the chart
        "field_points": {
            "z": field_data["scaled_distance"].tolist(),
            "v": field_data["velocity"].tolist(),
        },
        "dist_grid": dist_grid.tolist(),
        "recovered_curve": recovered_curve.tolist(),
        "ufc_curve": ufc_curve.tolist(),
        "has_ground_truth": field_data["true_n"] is not None,
    }
    if field_data["true_n"] is not None:
        true_curve = field_data["true_k"] * dist_grid ** (-field_data["true_n"])
        response["true_curve"] = true_curve.tolist()
        response["true_n"] = field_data["true_n"]
        response["true_k"] = field_data["true_k"]

    return jsonify(response)


if __name__ == "__main__":
    print()
    print(f"Test-set calibration at startup: {COVERAGE}")
    print("Serving on http://localhost:5550")
    app.run(host="0.0.0.0", port=5550, debug=False)