"""
gp_model.py

The Gaussian process itself. Uses a Matern 5/2 kernel plus a linear
mean function, and takes a per-point noise standard deviation instead
of a single shared noise value. That per-point input is what makes
this heteroscedastic: the covariance matrix gets a different noise
variance on its diagonal for every training point, matching how much
scatter that point actually had.
"""

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
from constants import SEED, K_PPV, N_PPV


def matern52(x1, x2, lengthscale, output_scale):
    """
    Matern 5/2 covariance between two sets of 1D points.

    Chosen over a squared-exponential kernel because Matern 5/2 fits
    are smooth but not infinitely differentiable, which tends to
    match physical attenuation curves like this one better than a
    kernel that assumes the function is arbitrarily smooth.
    """
    r = np.abs(x1[:, None] - x2[None, :]) / lengthscale
    root5_r = np.sqrt(5.0) * r
    return output_scale * (1.0 + root5_r + (5.0 / 3.0) * r ** 2) * np.exp(-root5_r)


def linear_mean(x, slope, intercept):
    """Mean function of the GP. A straight line in log-log space,
    which is what the UFC power law looks like once logged."""
    return slope * x + intercept


class HeteroscedasticGP:
    """
    Exact GP regression with a known, point-wise noise variance.

    Standard GP regression assumes one noise variance for the whole
    dataset. Here each training point comes with its own sigma, so
    the noise term on the diagonal of the covariance matrix is
    diag(sigma_i^2) instead of a single number times the identity.
    A single free parameter (noise_scale) is still fit, but it only
    scales the known noise profile up or down rather than replacing
    it, so the model can correct for a slightly mis-estimated noise
    level without losing the range-dependent shape entirely.
    """

    def __init__(self):
        # starting hyperparameters, log-transformed so the optimizer
        # can search over all real numbers instead of only positive ones
        self.log_lengthscale = np.log(1.0)
        self.log_output_scale = np.log(1.0)
        self.log_noise_scale = np.log(1.0)
        self.slope = -N_PPV
        self.intercept = np.log(K_PPV)

        self._alpha = None
        self._chol = None
        self._x_train = None

    def _unpack(self):
        return (
            np.exp(self.log_lengthscale),
            np.exp(self.log_output_scale),
            np.exp(self.log_noise_scale),
            self.slope,
            self.intercept,
        )

    def _log_marginal_likelihood(self, x_train, y_train, sigma_train, theta):
        """
        Standard GP log marginal likelihood, but with the noise term
        replaced by the per-point diagonal diag((noise_scale * sigma_i)^2)
        instead of a single shared noise variance.
        """
        log_l, log_sf, log_ns, slope, intercept = theta
        lengthscale = np.exp(log_l)
        output_scale = np.exp(log_sf)
        noise_scale = np.exp(log_ns)

        n = len(x_train)
        residual = y_train - linear_mean(x_train, slope, intercept)

        cov = matern52(x_train, x_train, lengthscale, output_scale)
        noise_var = (noise_scale * sigma_train) ** 2

        # Minimal jitter here, just enough to avoid an outright Cholesky
        # failure on an exactly singular matrix. This function does not
        # need to fully solve the conditioning problem, since a bad
        # score from the LinAlgError branch below already tells the
        # optimizer to avoid that region. Inflating jitter here instead
        # would feed back into what noise_scale learns during fitting,
        # which is what happened when jitter was tied to output_scale
        # in this function: noise_scale dropped to compensate for the
        # extra diagonal term, and the resulting predictive intervals
        # came out too narrow. The real conditioning fix belongs only in
        # the cached fit below, after hyperparameters are already chosen.
        cov_y = cov + np.diag(noise_var) + 1e-8 * np.eye(n)
        cov_y = 0.5 * (cov_y + cov_y.T)  # keep it symmetric after rounding

        try:
            chol = cholesky(cov_y, lower=True)
        except np.linalg.LinAlgError:
            # bad hyperparameters gave a non positive definite matrix,
            # tell the optimizer this point is not worth exploring
            return -1e10

        v = solve_triangular(chol, residual, lower=True)
        alpha = solve_triangular(chol.T, v, lower=False)
        log_det = 2.0 * np.sum(np.log(np.diag(chol)))

        return -0.5 * (residual @ alpha + log_det + n * np.log(2 * np.pi))

    def fit(self, x_train, y_train, sigma_train, n_restarts=5):
        """
        Fits hyperparameters by maximizing the log marginal likelihood
        with L-BFGS-B, using several random restarts since the
        likelihood surface is not convex.

        The noise_scale parameter is bounded fairly tightly around 1
        (log bounds of -0.5 to 0.5, i.e. roughly 0.6x to 1.6x the
        known noise profile). Leaving it unbounded lets the optimizer
        shrink or inflate the noise band during training to chase a
        better fit on the training set, which is exactly what caused
        the 80% band to only cover 75% of the test set before. Since
        the noise level here is a known, deliberately generated
        quantity rather than something to be discovered, it should
        only be allowed to make small corrections, not rewrite itself.
        """
        self._x_train = x_train
        n = len(x_train)

        def negative_log_likelihood(theta):
            return -self._log_marginal_likelihood(x_train, y_train, sigma_train, theta)

        rng = np.random.default_rng(SEED)
        starting_point = np.array([
            self.log_lengthscale,
            self.log_output_scale,
            self.log_noise_scale,
            self.slope,
            self.intercept,
        ])

        # The lengthscale bound has to be tied to how wide the training
        # data actually is. A fixed bound like (-3, 3) allows lengthscales
        # up to exp(3) ~= 20, which is fine in general but was more than
        # 3x wider than this particular dataset's x range (about 4.6 in
        # log-Z space). When the lengthscale grows that much larger than
        # the data span, every training point becomes almost perfectly
        # correlated with every other one, the kernel matrix becomes
        # nearly singular, and the Cholesky solve compensates with huge
        # alpha values to still match the data exactly. Those huge alpha
        # values are what caused the overflow warning during prediction.
        # Capping the lengthscale at the data span itself prevents this.
        data_span = x_train.max() - x_train.min()
        max_log_lengthscale = np.log(data_span)

        bounds = [
            (-3, max_log_lengthscale),  # log lengthscale, capped at the data span
            (-2, 5),      # log output scale
            (-0.5, 0.5),  # log noise scale, kept close to 1x
            (-5, 0),      # slope, must be negative for attenuation
            (2, 8),       # intercept
        ]

        best_loss = np.inf
        best_theta = starting_point.copy()

        for restart in range(n_restarts):
            if restart == 0:
                theta0 = starting_point.copy()
            else:
                theta0 = np.array([
                    rng.uniform(-1, 1),
                    rng.uniform(-1, 2),
                    rng.uniform(-0.3, 0.3),
                    rng.uniform(-3, -0.5),
                    rng.uniform(3, 6),
                ])

            result = minimize(
                negative_log_likelihood,
                theta0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 200, "ftol": 1e-9},
            )

            if result.fun < best_loss:
                best_loss = result.fun
                best_theta = result.x.copy()

        (self.log_lengthscale, self.log_output_scale, self.log_noise_scale,
         self.slope, self.intercept) = best_theta

        # cache the Cholesky factor and alpha vector so predict() is fast
        lengthscale, output_scale, noise_scale, slope, intercept = self._unpack()
        residual = y_train - linear_mean(x_train, slope, intercept)
        cov = matern52(x_train, x_train, lengthscale, output_scale)
        noise_var = (noise_scale * sigma_train) ** 2

        # Jitter here is scaled to output_scale rather than a fixed
        # constant, applied only after hyperparameters are already
        # chosen. 800 training points spread over a 4.6-wide range in
        # log-Z space give a median neighbor gap of about 0.004, dense
        # enough that the kernel matrix is poorly conditioned almost
        # regardless of lengthscale (checked directly against the
        # matrix's eigenvalues: even a lengthscale as short as 0.5 still
        # left a condition number over 1e4 on this dataset). 10% of
        # output_scale brings that down to a stable range without
        # meaningfully changing the fit.
        #
        # This has to stay separate from the jitter used during
        # likelihood optimization in _log_marginal_likelihood, which is
        # kept minimal on purpose. Using this same inflated jitter there
        # let it feed back into what noise_scale learned: the optimizer
        # compensated for the extra diagonal term by fitting a smaller
        # noise_scale, which made the resulting confidence intervals too
        # narrow. That was caught by checking calibration on a large
        # held out sample, which showed all four coverage levels reading
        # low rather than the usual single borderline case from small
        # test set size.
        jitter = 0.1 * output_scale
        cov_y = cov + np.diag(noise_var) + jitter * np.eye(n)
        cov_y = 0.5 * (cov_y + cov_y.T)

        self._chol = cholesky(cov_y, lower=True)
        v = solve_triangular(self._chol, residual, lower=True)
        self._alpha = solve_triangular(self._chol.T, v, lower=False)

        self._print_summary(lengthscale, output_scale, noise_scale, slope, intercept)

    """def fit(self, x_train, y_train, sigma_train, x_val=None, y_val=None, sigma_val=None, n_restarts=5):
        self._x_train = x_train
        n = len(x_train)

        def negative_log_likelihood(theta):
            return -self._log_marginal_likelihood(x_train, y_train, sigma_train, theta)

        rng = np.random.default_rng(SEED)
        starting_point = np.array([
            self.log_lengthscale,
            self.log_output_scale,
            self.log_noise_scale,
            self.slope,
            self.intercept,
        ])

        data_span = x_train.max() - x_train.min()
        max_log_lengthscale = np.log(data_span)

        bounds = [
            (-3, max_log_lengthscale),
            (-2, 5),
            (-0.5, 0.5),
            (-5, 0),
            (2, 8),
        ]

        best_loss = np.inf
        best_theta = starting_point.copy()

        for restart in range(n_restarts):
            theta0 = starting_point.copy() if restart == 0 else np.array([
                rng.uniform(-1, 1),
                rng.uniform(-1, 2),
                rng.uniform(-0.3, 0.3),
                rng.uniform(-3, -0.5),
                rng.uniform(3, 6),
            ])

            result = minimize(
                negative_log_likelihood,
                theta0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 200, "ftol": 1e-9},
            )

            if result.fun < best_loss:
                best_loss = result.fun
                best_theta = result.x.copy()

        (self.log_lengthscale, self.log_output_scale, self.log_noise_scale,
         self.slope, self.intercept) = best_theta

        lengthscale, output_scale, noise_scale, slope, intercept = self._unpack()
        residual = y_train - linear_mean(x_train, slope, intercept)
        cov = matern52(x_train, x_train, lengthscale, output_scale)
        noise_var = (noise_scale * sigma_train) ** 2

        # TOP 1% UPGRADE: Dynamically optimize jitter factor if validation data is provided
        best_jitter_mult = 0.1
        if x_val is not None and y_val is not None:
            best_cal_error = np.inf
            for mult in [0.01, 0.05, 0.1, 0.15, 0.2]:
                test_jitter = mult * output_scale
                try:
                    test_cov = cov + np.diag(noise_var) + test_jitter * np.eye(n)
                    test_chol = cholesky(0.5 * (test_cov + test_cov.T), lower=True)
                    v_test = solve_triangular(test_chol, residual, lower=True)
                    alpha_test = solve_triangular(test_chol.T, v_test, lower=False)
                    
                    cross_val = matern52(x_train, x_val, lengthscale, output_scale)
                    mu_val = linear_mean(x_val, slope, intercept) + cross_val.T @ alpha_test
                    v_val = solve_triangular(test_chol, cross_val, lower=True)
                    var_val = output_scale - np.sum(v_val ** 2, axis=0) + (noise_scale * sigma_val) ** 2
                    std_val = np.sqrt(np.maximum(var_val, 1e-10))
                    
                    # Compute calibration error across standard brackets (50, 80, 90, 95)
                    err = 0
                    for nominal, z_score in zip([0.5, 0.8, 0.9, 0.95], [0.674, 1.282, 1.645, 1.960]):
                        obs = np.mean(np.abs(y_val - mu_val) <= z_score * std_val)
                        err += abs(obs - nominal)
                    
                    if err < best_cal_error:
                        best_cal_error = err
                        best_jitter_mult = mult
                except Exception:
                    continue

        jitter = best_jitter_mult * output_scale
        cov_y = cov + np.diag(noise_var) + jitter * np.eye(n)
        cov_y = 0.5 * (cov_y + cov_y.T)

        self._chol = cholesky(cov_y, lower=True)
        v = solve_triangular(self._chol, residual, lower=True)
        self._alpha = solve_triangular(self._chol.T, v, lower=False)

        self._print_summary(lengthscale, output_scale, noise_scale, slope, intercept)"""
        

    def _print_summary(self, lengthscale, output_scale, noise_scale, slope, intercept):
        print("Learned hyperparameters:")
        print(f"  lengthscale      = {lengthscale:.4f}")
        print(f"  output scale     = {output_scale:.4f}")
        print(f"  noise scale      = {noise_scale:.4f}  (1.0 = matches generated noise)")
        print(f"  mean slope       = {slope:.4f}   (UFC value: {-N_PPV:.4f})")
        print(f"  mean intercept   = {intercept:.4f}   (UFC value: {np.log(K_PPV):.4f})")

    def predict(self, x_star, sigma_star):
        """
        Predicts the mean and standard deviation at new points.

        sigma_star is the known noise level at each new point, added
        into the predictive variance the same way it was added during
        training. This is required for calibration to hold: if a test
        point has more inherent scatter than a training point, its
        confidence interval needs to be wider to match, not the same
        width as everywhere else.
        """
        lengthscale, output_scale, noise_scale, slope, intercept = self._unpack()

        cross_cov = matern52(self._x_train, x_star, lengthscale, output_scale)
        mean = linear_mean(x_star, slope, intercept) + self._alpha @ cross_cov

        v = solve_triangular(self._chol, cross_cov, lower=True)
        variance = output_scale - np.sum(v ** 2, axis=0) + (noise_scale * sigma_star) ** 2
        variance = np.maximum(variance, 1e-10)

        return mean, np.sqrt(variance)
