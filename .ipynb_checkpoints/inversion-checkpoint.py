"""
inversion.py

Week 2 deliverable: given only a handful of field measurements
(pretend real data, generated here with a different seed so it does
not overlap the training set), recover the underlying attenuation
exponent n and coupling constant k by gradient descent.

Uses Adam instead of plain gradient descent since the two parameters
here have very different natural scales (k is order 100, n is order
1), and Adam adapts the step size per parameter instead of needing a
separate learning rate tuned for each one by hand.
"""

import numpy as np


def generate_field_data(seed, n_points=7):
    """
    Simulates a small set of noisy field measurements from a soil
    with different true parameters than the training set, so
    recovery is a genuine test and not just re-deriving numbers
    already baked into the model.
    """
    rng = np.random.default_rng(seed)
    true_n = 2.80
    true_k = 175.0

    scaled_distance = np.exp(rng.uniform(np.log(2.0), np.log(50.0), size=n_points))
    true_velocity = true_k * scaled_distance ** (-true_n)

    measurement_noise = 0.25 * np.ones(n_points)
    log_velocity = np.log(true_velocity) + rng.normal(0.0, measurement_noise, n_points)

    return {
        "scaled_distance": scaled_distance,
        "velocity": np.exp(log_velocity),
        "log_velocity": log_velocity,
        "true_n": true_n,
        "true_k": true_k,
    }


def recover_parameters(field_data, n_epochs=300, learning_rate=0.01):
    """
    Fits log(k) and n by minimizing mean squared error in log space
    with the Adam optimizer, starting from rough initial guesses.
    Working in log space for k keeps it positive automatically and
    puts the loss landscape on a friendlier scale.
    """
    log_distance = np.log(field_data["scaled_distance"])
    log_velocity = field_data["log_velocity"]
    n_points = len(log_distance)

    # initial guesses, close to but not exactly the true values,
    # to simulate a reasonable engineering starting estimate
    log_k = np.log(160.0)
    n_exponent = 2.75
    params = np.array([log_k, n_exponent])

    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    first_moment = np.zeros(2)
    second_moment = np.zeros(2)
    loss_history = []

    for epoch in range(1, n_epochs + 1):
        log_k_current, n_current = params
        predicted_log_velocity = log_k_current - n_current * log_distance
        residual = predicted_log_velocity - log_velocity
        loss = np.mean(residual ** 2)

        grad_log_k = (2.0 / n_points) * np.sum(residual)
        grad_n = (2.0 / n_points) * np.sum(residual * (-log_distance))
        gradient = np.array([grad_log_k, grad_n])

        first_moment = beta1 * first_moment + (1 - beta1) * gradient
        second_moment = beta2 * second_moment + (1 - beta2) * gradient ** 2
        first_corrected = first_moment / (1 - beta1 ** epoch)
        second_corrected = second_moment / (1 - beta2 ** epoch)

        params = params - learning_rate * first_corrected / (np.sqrt(second_corrected) + epsilon)
        loss_history.append(loss)

    recovered_n = params[1]
    recovered_k = np.exp(params[0])

    return {
        "recovered_n": recovered_n,
        "recovered_k": recovered_k,
        "loss_history": loss_history,
    }


def recover_soil_parameters(field_x, field_y, field_sigma, n_restarts=10):
    
    best_loss = np.inf
    best_params = None
    
    # Define bounds for [n, log(k)]
    bounds = [(1.0, 5.0), (np.log(10.0), np.log(500.0))]
    rng = np.random.default_rng(42)
    
    for i in range(n_restarts):
        if i == 0:
            # Informed starting guess based on standard UFC baselines
            x0 = [2.5, np.log(150.0)]
        else:
            x0 = [rng.uniform(1.5, 4.0), rng.uniform(np.log(50.0), np.log(300.0))]
            
        def objective(params):
            n_est, log_k_est = params
            k_est = np.exp(log_k_est)
            
            # Theoretical model prediction in log-space
            # log(PPV) = ln(k) - n * ln(Z)
            pred_y = np.log(k_est) - n_est * field_x
            
            # Weighted least squares objective accounting for known field variance
            weights = 1.0 / (field_sigma ** 2 + 1e-8)
            return np.sum(weights * (field_y - pred_y) ** 2)
            
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        
        if res.success and res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x
            
    n_recovered = best_params[0]
    k_recovered = np.exp(best_params[1])

    return n_recovered, k_recovered