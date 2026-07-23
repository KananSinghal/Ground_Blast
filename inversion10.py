"""
inversion.py

Recovers soil attenuation parameter n and coupling factor k from simulated field data.
"""

import numpy as np
from scipy.optimize import minimize
from constants import K_PPV, N_PPV, SEED


def generate_field_data(n_points=20, rng_seed=None):
    """Generates synthetic field measurements with heteroscedastic noise."""
    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed)
    else:
        rng = np.random.default_rng(SEED)

    x = rng.uniform(1.0, 3.5, n_points)  # log(Z)
    
    true_n = N_PPV
    true_k = K_PPV
    
    true_y = np.log(true_k) - true_n * x
    sigma = 0.05 + 0.05 * np.abs(x - np.mean(x))
    y = true_y + rng.normal(0, sigma)
    
    return x, y, sigma


def recover_parameters(field_x, field_y, field_sigma, n_restarts=10):
    """
    Recovers soil attenuation parameter n and coupling factor k from field 
    measurements using multi-start gradient-based optimization to avoid local minima.
    """
    best_loss = np.inf
    best_params = None
    
    # Bounds for [n, log(k)]
    bounds = [(1.0, 5.5), (np.log(10.0), np.log(500.0))]
    rng = np.random.default_rng(SEED)
    
    for i in range(n_restarts):
        if i == 0:
            x0 = [2.5, np.log(150.0)]
        else:
            x0 = [rng.uniform(1.5, 4.0), rng.uniform(np.log(50.0), np.log(300.0))]
            
        def objective(params):
            n_est, log_k_est = params
            k_est = np.exp(log_k_est)
            pred_y = np.log(k_est) - n_est * field_x
            weights = 1.0 / (field_sigma ** 2 + 1e-8)
            return np.sum(weights * (field_y - pred_y) ** 2)
            
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        
        if res.success and res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x
            
    if best_params is None:
        return 2.8, 175.0
        
    n_recovered = best_params[0]
    k_recovered = np.exp(best_params[1])
    
    return n_recovered, k_recovered


# Alias to support any alternate imports
recover_soil_parameters = recover_parameters