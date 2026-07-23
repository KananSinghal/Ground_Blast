"""
server.py

Just a Flask wrapper around the stuff I already wrote for project 3.
I'm not redoing the model or the math here, this file just imports
gp_model.py, calibration.py, constants.py, dataset.py, and
inversion.py and calls them from HTTP routes so the frontend has
something to fetch() from.

At startup it does phase 1 and phase 2 from main.py one time (make
the dataset, fit the GP) and keeps the fitted GP + test set sitting
in memory. After that every request is just calling predict() or the
inversion function on stuff that's already fitted, so it should be
fast.

Run with: python server.py
then go to http://localhost:5550
(NOT 5000, mac's AirPlay Receiver grabs port 5000 by default so I
switched to 5550 to avoid that whole headache)
"""

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from constants import SEED, K_PPV, N_PPV
from dataset import generate_dataset
from gp_model import HeteroscedasticGP
from calibration import check_coverage
from inversion import generate_field_data, recover_parameters

app = Flask(__name__, static_folder=".", static_url_path="")


# ---- startup stuff, only runs once when the server starts ----
# this is basically phases 1-2 of main.py, just done here instead so
# the server can hold onto the result instead of re-fitting every request

print("Generating dataset and fitting GP once at startup, this takes a few seconds...")
DATA = generate_dataset(n_points=1000, seed=SEED)

GP = HeteroscedasticGP()
GP.fit(DATA["x_train"], DATA["y_train"], DATA["sigma_train"], n_restarts=5)

MEAN_TEST, SIGMA_TEST = GP.predict(DATA["x_test"], DATA["sigma_test"])
COVERAGE = check_coverage(MEAN_TEST, SIGMA_TEST, DATA["y_test"])

# min/max of log(Z) across the dataset, used to build the smooth grid
# for plotting and to check if a query is inside the training range
_LOG_Z_MIN = np.log(DATA["scaled_distance"].min())
_LOG_Z_MAX = np.log(DATA["scaled_distance"].max())
_MEAN_SIGMA = float(DATA["sigma"].mean())


def _z_scores():
    # these are just the two-sided normal z values for 50/80/90/95%,
    # same levels calibration.py already checks against
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
    # just serves index.html from the same folder, nothing fancy
    return send_from_directory(".", "index.html")


@app.route("/api/calibration-curve")
def calibration_curve():
    """
    Sends back everything the page needs on first load: the raw
    scatter points, the true UFC curve, the GP mean + bands over the
    whole Z range, the held out test points, and the calibration
    numbers. Basically phases 1-3 of main.py but as JSON instead of
    matplotlib pngs.
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

    # only sending 300 of the 1000 raw points back, a browser chart
    # doesn't need all 1000 to show the same shape and it keeps the
    # response smaller
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
    This is the week 1 deliverable, just as a single API call instead
    of a plot. Give it one Z value in the query string, it gives back
    the GP's mean PPV plus the 50/80/90/95% bands, and whether that Z
    was even inside the training range in the first place.
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
    Week 2 deliverable. Recovers n and k from field measurements using
    recover_parameters() from inversion.py exactly as is, I'm not
    touching that logic here. If the POST body has "points" in it,
    use those (treats them as log-space velocity with a flat 0.25
    noise, which is a reasonable guess for a handful of manually typed
    field readings). If there's no "points", just fall back to
    generate_field_data() with whatever seed came in the body, same
    simulated scenario main.py already runs.
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

        # no true_n / true_k here since these are made-up points the
        # user typed in, not one of my simulated soils
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

    # build a smooth grid a bit wider than the actual field points so
    # the recovered curve doesn't look like it's cut off right at the
    # edge of the data
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
        "loss_history": result["loss_history"][::5],  # only send every 5th point, chart doesn't need all 300
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