"""
simulate_data.py — Data Generating Process (DGP) for maternal risk simulation
=============================================================================
Generates N=500 synthetic subjects with:
  - Scalar latent risk state theta_i(t) evolving as Gaussian random walk
  - 4-dim wearable observations y_i(t) = alpha * theta_i(t) + eps_i(t)
  - Event times from Weibull proportional hazards model
  - Random censoring + administrative censoring at week 40

DGP v3.4: adjusted event rate to ~9% (realistic for high-risk cohort),
added random dropout censoring mechanism.

All parameters are hardcoded for reproducibility.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DGPParams:
    """Complete DGP parameter specification (v3.4)."""
    # State evolution
    sigma2_omega: float = 0.04       # State noise variance

    # Observation model
    alpha: np.ndarray = None         # Observation loadings (4x1)
    Sigma_eps: np.ndarray = None     # Observation noise covariance (4x4 diagonal)

    # Event hazard
    gamma: float = 0.8               # Latent state -> hazard link (v3.5: from 0.5)
    k_weibull: float = 1.2           # Weibull shape parameter
    lambda_weibull: float = 0.0015   # Weibull scale parameter (v3.5: event rate ~10-15%)
    beta: np.ndarray = None          # Baseline covariate effects (2x1)

    # Simulation design
    N: int = 500                     # Number of subjects
    T_weeks: int = 40                # Total gestational weeks
    censoring_week: int = 40         # Administrative censoring time

    # Random censoring (v3.4)
    lambda_dropout: float = 0.0      # Exponential dropout rate; 0 = no dropout
                                     # Set ~0.02-0.05 for 15-30% dropout

    def __post_init__(self):
        if self.alpha is None:
            self.alpha = np.array([1.0, -0.8, -0.6, -0.7])
        if self.Sigma_eps is None:
            self.Sigma_eps = np.diag([0.25, 0.16, 0.36, 0.09])
        if self.beta is None:
            self.beta = np.array([0.005, 0.005])  # v3.4: reduced from [0.03, 0.05]


def generate_baseline_covariates(N: int, seed: int = 42) -> np.ndarray:
    """
    Generate baseline covariates X_i = (age, BMI).
    - age ~ N(28, 5^2), clipped to [18, 45]
    - BMI ~ N(26, 5^2), clipped to [18, 45]
    """
    rng = np.random.RandomState(seed)
    age = np.clip(rng.normal(28, 5, N), 18, 45)
    bmi = np.clip(rng.normal(26, 5, N), 18, 45)
    return np.column_stack([age, bmi])  # (N, 2)


def generate_latent_trajectories(N: int, T: int, sigma2_omega: float,
                                  seed: int = 42) -> np.ndarray:
    """
    Generate latent risk state theta_i(t) as Gaussian random walk.
    theta_i(0) ~ N(0, 0.1)
    theta_i(t) = theta_i(t-1) + omega_i(t),  omega_i(t) ~ N(0, sigma2_omega)

    Returns:
        theta: np.ndarray, shape (T, N)
    """
    rng = np.random.RandomState(seed)
    theta = np.zeros((T, N))
    theta[0, :] = rng.normal(0, np.sqrt(0.1), N)
    for t in range(1, T):
        theta[t, :] = theta[t - 1, :] + rng.normal(0, np.sqrt(sigma2_omega), N)
    return theta


def generate_observations(theta: np.ndarray, alpha: np.ndarray,
                          Sigma_eps: np.ndarray,
                          seed: int = 42) -> np.ndarray:
    """
    Generate 4-dim wearable observations y_i(t) = alpha * theta_i(t) + eps_i(t).

    Returns:
        Y: np.ndarray, shape (T, N, 4) -- wearable data
    """
    rng = np.random.RandomState(seed)
    T, N = theta.shape
    D = len(alpha)
    Y = np.zeros((T, N, D))
    sigma_eps = np.sqrt(np.diag(Sigma_eps))  # (D,)
    for d in range(D):
        Y[:, :, d] = alpha[d] * theta + rng.normal(0, sigma_eps[d], (T, N))
    return Y


def generate_event_times(theta: np.ndarray, X: np.ndarray,
                          params: DGPParams,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate event times from Weibull proportional hazards model:
      lambda_i(t) = lambda_0(t) * exp(gamma * theta_i(t) + beta'X_i)
      lambda_0(t) = k * lambda * t^(k-1)

    Uses inverse CDF method with discretized cumulative hazard.
    v3.4: Fixed event generation -- no event => censoring (not event at T).
    v3.4: Added random dropout censoring C_i ~ Exponential(lambda_dropout).

    Returns:
        observed_time: (N,) observed time = min(event_time, censoring_time)
        event_indicator: (N,) 1=event occurred, 0=censored
    """
    rng = np.random.RandomState(seed)
    T, N = theta.shape
    k = params.k_weibull
    lam = params.lambda_weibull
    gamma = params.gamma
    beta = params.beta

    # Linear predictor: beta'X_i (constant per subject)
    linear_pred = X @ beta  # (N,)

    # Hazard at each time point
    t_grid = np.arange(1, T + 1).astype(float)  # weeks 1..40
    hazard_at_t = np.zeros((T, N))
    for t_idx in range(T):
        hazard_at_t[t_idx, :] = (k * lam * t_grid[t_idx] ** (k - 1) *
                                  np.exp(gamma * theta[t_idx, :] + linear_pred))

    # Cumulative hazard
    cum_hazard = np.cumsum(hazard_at_t, axis=0)  # (T, N)

    # --- v3.4: Fixed event generation ---
    # Generate event times via inverse CDF
    U = rng.uniform(0, 1, N)
    event_times_raw = np.full(N, T + 1, dtype=int)  # default: no event
    event_indicators_raw = np.zeros(N, dtype=int)    # default: censored

    for i in range(N):
        surv = np.exp(-cum_hazard[:, i])
        below = np.where(surv <= U[i])[0]
        if len(below) > 0:
            event_times_raw[i] = below[0] + 1  # 1-indexed week
            event_indicators_raw[i] = 1         # event occurred
        else:
            event_times_raw[i] = T + 1          # no event by end
            event_indicators_raw[i] = 0         # censored

    # --- v3.4: Random dropout censoring ---
    if params.lambda_dropout > 0:
        dropout_times = rng.exponential(1.0 / params.lambda_dropout, N)
        dropout_times = np.ceil(dropout_times).astype(int)
        dropout_times = np.clip(dropout_times, 1, T)
    else:
        dropout_times = np.full(N, T + 1, dtype=int)

    # Administrative censoring
    admin_censor = params.censoring_week

    # Observed time = min(event_time, dropout_time, admin_censor)
    effective_censor = np.minimum(dropout_times, admin_censor)
    observed_time = np.minimum(event_times_raw, effective_censor)
    event_indicator = event_indicators_raw.copy()
    # If event time exceeds censoring, mark as censored
    event_indicator[event_times_raw > effective_censor] = 0

    return observed_time, event_indicator


@dataclass
class SimulatedData:
    """Container for all simulated data."""
    theta: np.ndarray       # (T, N) latent state
    Y: np.ndarray           # (T, N, 4) wearable observations
    X: np.ndarray           # (N, 2) baseline covariates
    event_times: np.ndarray # (N,) observed event/censoring times
    event_indicators: np.ndarray  # (N,) 1=event, 0=censored
    params: DGPParams

    def summary(self):
        """Print data summary."""
        n_events = self.event_indicators.sum()
        n_censored = (self.event_indicators == 0).sum()
        print(f"N = {self.params.N} subjects")
        print(f"T = {self.params.T_weeks} weeks")
        print(f"Event rate: {self.event_indicators.mean():.1%}")
        print(f"  Events: {n_events}, Censored: {n_censored}")
        if n_events > 0:
            print(f"  Mean event time (events only): "
                  f"{self.event_times[self.event_indicators == 1].mean():.1f} weeks")
        if n_censored > 0:
            print(f"  Mean censoring time (censored only): "
                  f"{self.event_times[self.event_indicators == 0].mean():.1f} weeks")
        print(f"Baseline covariates: age={self.X[:, 0].mean():.1f}+/-{self.X[:, 0].std():.1f}, "
              f"BMI={self.X[:, 1].mean():.1f}+/-{self.X[:, 1].std():.1f}")
        print(f"Latent state range: [{self.theta.min():.2f}, {self.theta.max():.2f}]")
        print(f"Observations shape: {self.Y.shape}")


def simulate(seed: int = 42) -> SimulatedData:
    """Run full simulation pipeline."""
    params = DGPParams()
    print("=" * 60)
    print("Simulating maternal risk data (DGP v3.4)")
    print("=" * 60)

    # 1. Baseline covariates
    X = generate_baseline_covariates(params.N, seed=seed)

    # 2. Latent trajectories
    theta = generate_latent_trajectories(params.N, params.T_weeks,
                                          params.sigma2_omega, seed=seed)

    # 3. Observations
    Y = generate_observations(theta, params.alpha, params.Sigma_eps, seed=seed)

    # 4. Event times
    event_times, event_indicators = generate_event_times(
        theta, X, params, seed=seed)

    data = SimulatedData(
        theta=theta, Y=Y, X=X,
        event_times=event_times,
        event_indicators=event_indicators,
        params=params
    )
    data.summary()
    return data


if __name__ == "__main__":
    data = simulate()
    # Quick sanity check
    assert data.theta.shape == (40, 500)
    assert data.Y.shape == (40, 500, 4)
    assert data.X.shape == (500, 2)
    assert data.event_times.shape == (500,)
    assert data.event_indicators.shape == (500,)
    print("\nAll shape checks passed.")
