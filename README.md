# Ground_Blast
Built backend pipelines using Gaussian Process Regression and Bayesian Neural Networks to predict ground shock parameters (PPV and displacement) with uncertainty quantification. checks its own calibration against held-out data, and inverts field readings back into soil parameters (n, k).

# What's in here
1. dataset.py, constants.py — synthetic data generation
2. gp_model.py — the Matérn-5⁄2 heteroscedastic GP itself
3. calibration.py — held-out coverage checking
4. inversion.py — gradient-descent parameter recovery
5. plots.py, main.py — the original static-figure pipeline (python main.py, output in output_plots/)
7. server.py — a Flask API over the same code, for the interactive frontend
8. index.html — the frontend: raw data, the fitted model, calibration, a live predictor, the inversion tool, and 9. several 3D visualizations, all by real API calls 

Running it locally:

1. pip install -r requirements.txt
2. python server.py
3. Then open http://localhost:5550 (not 5000 — see the note in server.py's docstring).
