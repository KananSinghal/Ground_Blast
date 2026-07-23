# Ground_Blast
Built backend pipelines using Gaussian Process Regression and Bayesian Neural Networks to predict ground shock parameters (PPV and displacement) with uncertainty quantification. checks its own calibration against held-out data, and inverts field readings back into soil parameters (n, k).

# What's in here
dataset.py, constants.py — synthetic data generation
gp_model.py — the Matérn-5⁄2 heteroscedastic GP itself
calibration.py — held-out coverage checking
inversion.py — gradient-descent parameter recovery
plots.py, main.py — the original static-figure pipeline (python main.py, output in output_plots/)
server.py — a Flask API over the same code, for the interactive frontend
index.html — the frontend: raw data, the fitted model, calibration, a live predictor, the inversion tool, and several 3D visualizations, all by real API calls 

Running it locally
bash
pip install -r requirements.txt
python server.py

Then open http://localhost:5550 (not 5000 — see the note in server.py's docstring).
