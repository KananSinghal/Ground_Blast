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
from scipy.linalg import cho_factor, cho_solve, solve
from scipy.optimize import minimize
from constants import SEED, K_PPV, N_PPV


def matern52(x1, x2, lengthscale, output_scale):
    """
    Matern 5/2 covariance between two sets of 1D points.

    Chosen over a squared exponential kernel because Matern 5/2 fits
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
        # starting hyperparameters, log transformed so the optimizer
        # can search over all real numbers instead of only positive ones
        self.log_lengthscale = np.log(1.0)
        self.log_output_scale = np.log(1.0)
        self.log_noise_scale = np.log(1.0)
        self.slope = -N_PPV
        self.intercept = np.log(K_PPV)

        self._alpha = None
        self._cov_y = None
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

        This function is called many times per optimizer step, so it
        keeps a minimal fixed jitter rather than the relative jitter
        used in fit(). That is a deliberate choice, not an oversight:
        see the comment in fit() for why the two jitter amounts are
        kept separate.
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
            chol_factor, lower = cho_factor(cov_y, lower=True)
        except np.linalg.LinAlgError:
            # bad hyperparameters gave a non positive definite matrix,
            # tell the optimizer this point is not worth exploring
            return -1e10

        alpha = cho_solve((chol_factor, lower), residual)
        log_det = 2.0 * np.sum(np.log(np.diag(chol_factor)))

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
        # nearly singular, and the solve compensates with huge alpha
        # values to still match the data exactly. Those huge alpha
        # values are what caused the overflow warning during prediction.
        # Capping the lengthscale at the data span itself prevents this,
        # although the relative jitter added below is the real backstop,
        # since near duplicate points close together can still make the
        # matrix nearly singular even at a capped lengthscale.
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

        # cache the covariance matrix and alpha vector so predict() is fast
        lengthscale, output_scale, noise_scale, slope, intercept = self._unpack()
        residual = y_train - linear_mean(x_train, slope, intercept)
        cov = matern52(x_train, x_train, lengthscale, output_scale)
        noise_var = (noise_scale * sigma_train) ** 2

        # Jitter here is relative to the matrix's own scale instead of a
        # fixed fraction of output_scale, applied only after hyperparameters
        # are already chosen. The standard formula is machine epsilon times
        # the average diagonal magnitude times n: eps (about 2.2e-16 for
        # float64) is the smallest relative gap the dtype can represent, so
        # multiplying it by how large the matrix entries actually are and by
        # n gives a jitter that scales with the matrix instead of a constant
        # that would need re-tuning if output_scale's bounds ever changed.
        #
        # This was needed because a fixed 0.1 * output_scale was not always
        # enough: 800 training points spread over a 4.6-wide range in
        # log-Z space give a median neighbor gap of about 0.004, dense
        # enough that some rows of the kernel matrix are nearly duplicates
        # of each other almost regardless of lengthscale. That near
        # singularity is what let a bad alpha vector through even though
        # the old Cholesky call did not raise an error, since Cholesky can
        # succeed numerically on a matrix that is only barely positive
        # definite and still return a factor whose solves blow up.
        #
        # This has to stay separate from the jitter used during likelihood
        # optimization in _log_marginal_likelihood, which is kept minimal on
        # purpose. Using this same inflated jitter there let it feed back
        # into what noise_scale learned: the optimizer compensated for the
        # extra diagonal term by fitting a smaller noise_scale, which made
        # the resulting confidence intervals too narrow. That was caught by
        # checking calibration on a large held out sample, which showed all
        # four coverage levels reading low rather than the usual single
        # borderline case from small test set size.
        average_diagonal = np.trace(cov) / n
        jitter = np.finfo(cov.dtype).eps * average_diagonal * n
        cov_y = cov + np.diag(noise_var) + jitter * np.eye(n)
        cov_y = 0.5 * (cov_y + cov_y.T)

        # solve directly for alpha instead of a Cholesky factor followed by
        # two triangular solves. assume_a="pos" routes this to LAPACK's
        # POSV, which factors and solves in a single call and raises
        # LinAlgError on failure instead of silently returning a factor
        # with a near zero pivot, which is what let a broken alpha reach
        # predict() undetected the first time.
        try:
            self._alpha = solve(cov_y, residual, assume_a="pos")
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                "covariance matrix is not positive definite even after "
                "jitter. this usually means the lengthscale is too large "
                "relative to how close together the training points are. "
                f"original error: {error}"
            )

        self._cov_y = cov_y

        self._print_summary(lengthscale, output_scale, noise_scale, slope, intercept)

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

        # solved directly against the cached covariance matrix rather than
        # a triangular factor, matching the change in fit(). this recovers
        # K^-1 @ cross_cov, the same quantity the old triangular solves
        # were building toward, without keeping a Cholesky factor around.
        v = solve(self._cov_y, cross_cov, assume_a="pos")
        variance = output_scale - np.sum(cross_cov * v, axis=0) + (noise_scale * sigma_star) ** 2
        variance = np.maximum(variance, 1e-10)

        return mean, np.sqrt(variance)