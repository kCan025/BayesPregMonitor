"""
bootstrap_pf.py -- Bootstrap Particle Filter for latent risk state estimation
============================================================================
Estimates eta_i(t) = E[theta_i(t) | y_{1:t}] for each subject using a
Bootstrap Particle Filter with N_particles=100.

State space model:
  State equation:   theta_i(t) = theta_i(t-1) + omega_i(t),  omega ~ N(0, sigma2_omega)
  Observation eq:   y_i(t) = alpha * theta_i(t) + eps_i(t), eps ~ N(0, Sigma_eps)

Output: eta_i(t) -- scalar dynamic risk score per subject per time point.
Also stores particle weights for downstream multiple imputation.
"""

import numpy as np
from typing import Tuple


class BootstrapPF:
    """
    Bootstrap Particle Filter for scalar state + Gaussian observations.

    Parameters:
        alpha: (D,) observation loadings
        sigma2_omega: scalar, state noise variance
        Sigma_eps: (D, D) diagonal observation noise covariance
        n_particles: int, number of particles (default 100)
    """

    def __init__(self, alpha: np.ndarray, sigma2_omega: float,
                 Sigma_eps: np.ndarray, n_particles: int = 100):
        self.alpha = alpha
        self.sigma2_omega = sigma2_omega
        self.Sigma_eps = Sigma_eps
        self.n_particles = n_particles
        self.D = len(alpha)

        # Precompute inverse and log-determinant of Sigma_eps
        self._Sigma_eps_inv = np.linalg.inv(Sigma_eps)
        self._log_det_Sigma = np.log(np.linalg.det(Sigma_eps))

    def _log_likelihood(self, y: np.ndarray, theta_particles: np.ndarray) -> np.ndarray:
        """
        Compute log N(y; alpha*theta, Sigma_eps) for each particle.

        Args:
            y: (D,) observation at time t for one subject
            theta_particles: (n_particles,) particle states

        Returns:
            log_lik: (n_particles,) log-likelihood values
        """
        # Predicted observations: alpha * theta for each particle
        y_pred = self.alpha[:, None] * theta_particles[None, :]  # (D, n_particles)
        residuals = y[:, None] - y_pred  # (D, n_particles)

        # Mahalanobis distance for each particle
        mahal = np.sum(residuals * (self._Sigma_eps_inv @ residuals), axis=0)

        log_lik = -0.5 * (self.D * np.log(2 * np.pi) +
                           self._log_det_Sigma + mahal)
        return log_lik

    def filter_subject(self, Y_i: np.ndarray,
                       prior_mean: float = 0.0,
                       prior_var: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Bootstrap PF for a single subject.

        Args:
            Y_i: (T, D) observations for subject i
            prior_mean: prior mean for theta_i(0)
            prior_var: prior variance for theta_i(0)

        Returns:
            eta: (T,) filtered posterior mean eta_i(t) = E[theta_i(t)|y_{1:t}]
            particles: (T, n_particles) all particle states
            weights: (T, n_particles) normalized weights at each time
        """
        T = Y_i.shape[0]
        Np = self.n_particles
        rng = np.random.RandomState()

        # Storage
        particles = np.zeros((T, Np))
        weights = np.zeros((T, Np))
        eta = np.zeros(T)

        # Initialize particles from prior
        theta_p = rng.normal(prior_mean, np.sqrt(prior_var), Np)
        w = np.ones(Np) / Np

        for t in range(T):
            # --- Predict (only for t > 0) ---
            if t > 0:
                theta_p = theta_p + rng.normal(0, np.sqrt(self.sigma2_omega), Np)

            # --- Update ---
            log_lik = self._log_likelihood(Y_i[t], theta_p)
            # Log-sum-exp for numerical stability
            log_w = np.log(w + 1e-300) + log_lik
            max_log = np.max(log_w)
            log_w = log_w - max_log - np.log(np.sum(np.exp(log_w - max_log)))
            w = np.exp(log_w)

            # Store
            particles[t] = theta_p
            weights[t] = w
            eta[t] = np.sum(w * theta_p)

            # --- Resample (systematic resampling) ---
            if self._effective_sample_size(w) < Np / 2:
                theta_p = self._systematic_resample(theta_p, w, rng)
                w = np.ones(Np) / Np

        return eta, particles, weights

    def filter_all(self, Y: np.ndarray,
                   prior_mean: float = 0.0,
                   prior_var: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Bootstrap PF for all subjects.

        Args:
            Y: (T, N, D) observations for all subjects

        Returns:
            eta_all: (T, N) filtered posterior means
            particles_all: (T, N, n_particles) particles
            weights_all: (T, N, n_particles) weights
        """
        T, N, D = Y.shape
        eta_all = np.zeros((T, N))
        particles_all = np.zeros((T, N, self.n_particles))
        weights_all = np.zeros((T, N, self.n_particles))

        for i in range(N):
            eta, parts, wts = self.filter_subject(Y[:, i, :],
                                                 prior_mean, prior_var)
            eta_all[:, i] = eta
            particles_all[:, i, :] = parts
            weights_all[:, i, :] = wts

        return eta_all, particles_all, weights_all

    @staticmethod
    def _effective_sample_size(w: np.ndarray) -> float:
        """ESS = 1 / sum(w_j^2)."""
        return 1.0 / np.sum(w ** 2)

    @staticmethod
    def _systematic_resample(particles: np.ndarray, weights: np.ndarray,
                             rng: np.random.RandomState) -> np.ndarray:
        """Systematic resampling -- lower variance than multinomial."""
        Np = len(particles)
        cumw = np.cumsum(weights)
        u = (rng.uniform() + np.arange(Np)) / Np
        indices = np.searchsorted(cumw, u)
        indices = np.clip(indices, 0, Np - 1)
        return particles[indices]


def marginal_multiple_imputation(particles: np.ndarray,
                                 weights: np.ndarray,
                                 M: int = 5,
                                 seed: int = 42) -> np.ndarray:
    """
    Marginal multiple imputation from filtered posterior.

    At each time point t, draw M values by weighted resampling
    from the particle set. This is a marginal (pointwise) imputation
    -- it does NOT produce jointly smooth trajectories.

    NOTE: This is an approximation. The filtering distributions
    p(theta_t | y_{1:t}) are marginal at each t, not the joint smoothing
    distribution p(theta_{1:T} | y_{1:T}). Future work: backward simulation
    for joint smoothing.

    Args:
        particles: (T, n_particles) for one subject
        weights: (T, n_particles) normalized weights
        M: number of imputations
        seed: random seed

    Returns:
        eta_m: (M, T) imputed trajectories
    """
    rng = np.random.RandomState(seed)
    T, Np = particles.shape
    eta_m = np.zeros((M, T))

    for m in range(M):
        for t in range(T):
            idx = rng.choice(Np, size=Np, replace=True, p=weights[t])
            eta_m[m, t] = particles[t, idx].mean()

    return eta_m


def trajectory_multiple_imputation(particles: np.ndarray,
                                    weights: np.ndarray,
                                    M: int = 5,
                                    seed: int = 42) -> np.ndarray:
    """
    Trajectory-level multiple imputation from filtered posterior.
    
    KEY FIX: At each time t, draw ONE particle (not mean of Np).
    This preserves full posterior variance, preventing attenuation
    bias in downstream Cox regression.
    
    Each imputed trajectory is a valid random draw from p(theta_t | y_{1:t}).
    With M such trajectories, between-imputation variance B > 0,
    and Rubin's rules properly account for measurement uncertainty.
    
    Reference: Rubin (1987), "Multiple Imputation for Nonresponse";
    Doucet & Johansen (2011), "A Tutorial on Particle Filtering".
    
    Args:
        particles: (T, n_particles) for one subject
        weights: (T, n_particles) normalized weights
        M: number of imputed trajectories
        seed: random seed
    
    Returns:
        eta_m: (M, T) imputed trajectories
    """
    rng = np.random.RandomState(seed)
    T, Np = particles.shape
    eta_m = np.zeros((M, T))
    
    for m in range(M):
        for t in range(T):
            j = rng.choice(Np, p=weights[t])
            eta_m[m, t] = particles[t, j]
    
    return eta_m


def trajectory_mi_all_subjects(particles_all: np.ndarray,
                                weights_all: np.ndarray,
                                M: int = 5,
                                seed: int = 42) -> list:
    """
    Generate MI trajectories for all subjects.
    
    Args:
        particles_all: (T, N, n_particles)
        weights_all: (T, N, n_particles)
        M: number of imputed trajectories
        seed: random seed
    
    Returns:
        eta_m_list: list of M arrays, each (T, N)
    """
    T, N, Np = particles_all.shape
    eta_m_list = []
    
    for m in range(M):
        rng = np.random.RandomState(seed + m)
        eta_m = np.zeros((T, N))
        for i in range(N):
            for t in range(T):
                j = rng.choice(Np, p=weights_all[t, i])
                eta_m[t, i] = particles_all[t, i, j]
        eta_m_list.append(eta_m)
    
    return eta_m_list


if __name__ == "__main__":
    # Quick test with simulated data
    from simulate_data import simulate, DGPParams

    data = simulate()
    params = data.params

    pf = BootstrapPF(
        alpha=params.alpha,
        sigma2_omega=params.sigma2_omega,
        Sigma_eps=params.Sigma_eps,
        n_particles=100
    )

    print("\nRunning Bootstrap PF on N=500 subjects...")
    eta_all, particles_all, weights_all = pf.filter_all(data.Y)

    print(f"eta shape: {eta_all.shape}")
    print(f"eta mean range: [{eta_all.mean(axis=0).min():.3f}, {eta_all.mean(axis=0).max():.3f}]")

    # Correlation between true theta and estimated eta
    from scipy.stats import pearsonr
    true_flat = data.theta.flatten()
    est_flat = eta_all.flatten()
    r, p = pearsonr(true_flat, est_flat)
    print(f"Correlation(theta, eta): r = {r:.4f} (p < {p:.2e})")

    # Test multiple imputation for subject 0
    eta_m = marginal_multiple_imputation(
        particles_all[:, 0, :], weights_all[:, 0, :], M=5)
    print(f"MI trajectories shape: {eta_m.shape}")
    print(f"MI trajectory std at t=20: {eta_m[:, 20].std():.4f}")
    print("\nBootstrap PF completed successfully.")
