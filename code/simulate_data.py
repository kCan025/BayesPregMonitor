"""
simulate_data.py — Data Generating Process (DGP) for maternal risk simulation
=============================================================================
Generates synthetic subjects with:
  - Scalar latent risk state theta_i(t) evolving as Gaussian random walk
  - 4-dim wearable observations y_i(t) = alpha * theta_i(t) + eps_i(t)
  - Event times from Weibull proportional hazards model
  - Random censoring + administrative censoring at week 40

DGP v4.0 (2026-10-25, after Dr. Sundaram's calibration guidance):
  - N = 2000 (v3.4 was 500); enlarged so Plan-C 4-week evaluation windows
    keep >=5 cases under the realistic ~4% event rate.
  - Baseline covariates now (age, bmi, age_ge33, prior_pe_history).
      age_ge33 = 1{age>=33}, HR 1.5 (WORKING ASSUMPTION, to be confirmed)
      prior_pe_history ~ Bernoulli(0.10) on an INDEPENDENT stream (seed+4),
      HR 3.0 (WORKING ASSUMPTION, to be confirmed). Prevalence 0.10 also tbc.
    The two new columns are APPENDED; X[:,0]=age, X[:,1]=bmi unchanged and the
    age/bmi RNG stream is untouched, so v3.4 streams remain reproducible.
  - beta = [0.005, 0.005, ln 1.5, ln 3.0].
  - Overall event rate calibrated to ~4% (US preeclampsia prevalence):
      lambda_weibull = 0.0002176 for linear latent process (baseline/heavy_tail)
      lambda_weibull = 0.0001227 for nonlinear latent process (nonlinear/combined)
    calibrated separately, N=2000, 80-rep verification (4.07% / 4.14%).
  - v3.4 N=500 production values were lambda=0.0015 (rate 18.87% linear /
    28.00% nonlinear over 80 reps). The N=500 4%-calibration lambdas
    (0.000313 / 0.000173) are NOT reused at N=2000.

All parameters are hardcoded for reproducibility.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

# ---- v4.0 calibration constants (N=2000, target ~4%) -----------------------
N_V4 = 2000
LAMBDA_V4_LINEAR = 0.0002176      # baseline / heavy_tail (linear latent process)
LAMBDA_V4_NONLINEAR = 0.0001227   # nonlinear / combined (nonlinear latent process)
PRIOR_PE_PREVALENCE = 0.10        # WORKING ASSUMPTION (to be confirmed)
AGE_GE33_CUTOFF = 33.0


@dataclass
class DGPParams:
    """Complete DGP parameter specification (v4.0)."""
    # State evolution
    sigma2_omega: float = 0.04       # State noise variance

    # Observation model
    alpha: np.ndarray = None         # Observation loadings (4x1)
    Sigma_eps: np.ndarray = None     # Observation noise covariance (4x4 diagonal)

    # Event hazard
    gamma: float = 0.8               # Latent state -> hazard link
    k_weibull: float = 1.2           # Weibull shape parameter
    lambda_weibull: float = LAMBDA_V4_LINEAR  # v4.0: calibrated to ~4% at N=2000 (linear)
    beta: np.ndarray = None          # Covariate effects (4x1); set in __post_init__

    # Simulation design
    N: int = N_V4                    # v4.0: 500 -> 2000 (window adequacy at ~4%)
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
            # v4.0: append effects for age_ge33 (HR 1.5) and prior_pe (HR 3.0).
            # HRs/prevalence are WORKING ASSUMPTIONS (to be confirmed by collaborator).
            self.beta = np.array([
                0.005, 0.005,
                np.log(1.5),           # age_ge33
                np.log(3.0),           # prior_pe_history
            ])


def generate_baseline_covariates(N: int, seed: int = 42) -> np.ndarray:
    """
    Generate baseline covariates X_i = (age, BMI, age_ge33, prior_pe_history).
    - age ~ N(28, 5^2), clipped to [18, 45]
    - BMI ~ N(26, 5^2), clipped to [18, 45]
      (age/BMI use the SAME RandomState(seed) stream as v3.4, so the first two
       columns are bit-for-bit unchanged -> old streams stay reproducible)
    - age_ge33 = 1{age >= 33} (deterministic function of age)
    - prior_pe_history ~ Bernoulli(0.10) on an INDEPENDENT stream RandomState(seed+4)
      (prevalence 0.10 is a working assumption, to be confirmed).
    New columns are APPENDED: X[:,0]=age, X[:,1]=bmi are never moved.
    """
    rng = np.random.RandomState(seed)
    age = np.clip(rng.normal(28, 5, N), 18, 45)
    bmi = np.clip(rng.normal(26, 5, N), 18, 45)
    age_ge33 = (age >= AGE_GE33_CUTOFF).astype(float)
    rng_pe = np.random.RandomState(seed + 4)  # independent stream; do NOT reuse seed/+1/+3
    prior_pe = (rng_pe.uniform(0, 1, N) < PRIOR_PE_PREVALENCE).astype(float)
    return np.column_stack([age, bmi, age_ge33, prior_pe])  # (N, 4)


def generate_latent_trajectories(N: int, T: int, sigma2_omega: float,
                                  seed: int = 42, mode: str = "linear",
                                  nonlinear_amp: float = 0.08) -> np.ndarray:
    """
    Generate latent risk state theta_i(t).

    mode="linear":   Gaussian random walk theta_t = theta_{t-1} + omega_t
    mode="nonlinear": random walk + sin perturbation
                      theta_t = theta_{t-1} + amp * sin(theta_{t-1}) + omega_t
    Nonlinear amplitude is small so the perturbation is a mild misspecification,
    not a qualitative change in dynamics.

    Returns:
        theta: np.ndarray, shape (T, N)
    """
    rng = np.random.RandomState(seed)
    theta = np.zeros((T, N))
    theta[0, :] = rng.normal(0, np.sqrt(0.1), N)
    for t in range(1, T):
        omega = rng.normal(0, np.sqrt(sigma2_omega), N)
        if mode == "nonlinear":
            theta[t, :] = theta[t - 1, :] + nonlinear_amp * np.sin(theta[t - 1, :]) + omega
        else:
            theta[t, :] = theta[t - 1, :] + omega
    return theta


def generate_observations(theta: np.ndarray, alpha: np.ndarray,
                          Sigma_eps: np.ndarray,
                          seed: int = 42) -> np.ndarray:
    """
    Generate 4-dim wearable observations y_i(t) = alpha * theta_i(t) + eps_i(t).
    eps ~ N(0, Sigma_eps).

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


def generate_observations_heavy_tail(theta: np.ndarray, alpha: np.ndarray,
                                     Sigma_eps: np.ndarray, df: float = 3.0,
                                     seed: int = 42) -> np.ndarray:
    """
    Generate observations with heavy-tailed (Student-t) noise.
    Noise variance is scaled so that Var(eps_d) matches Sigma_eps[d,d] exactly,
    i.e. each dimension is sigma_d * t(df) / sqrt(df/(df-2)).

    This is a model misspecification scenario: the Bootstrap PF assumes Gaussian
    likelihood but the true emissions have heavier tails.

    Args:
        theta: (T, N) latent states
        alpha: (D,) observation loadings
        Sigma_eps: (D, D) diagonal noise covariance (target variance)
        df: degrees of freedom for t-distribution (default 3, moderately heavy)
        seed: random seed

    Returns:
        Y: np.ndarray, shape (T, N, D)
    """
    rng = np.random.RandomState(seed)
    T, N = theta.shape
    D = len(alpha)
    Y = np.zeros((T, N, D))
    sigma_eps = np.sqrt(np.diag(Sigma_eps))               # (D,) target std
    scale = sigma_eps * np.sqrt((df - 2.0) / df)          # scale to match target variance
    for d in range(D):
        Y[:, :, d] = alpha[d] * theta + rng.standard_t(df, size=(T, N)) * scale[d]
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
    X: np.ndarray           # (N, 4) baseline covariates: age, bmi, age_ge33, prior_pe
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


def simulate(seed: int = 42, theta_mode: str = "linear",
             obs_noise: str = "gaussian", heavy_tail_df: float = 3.0,
             verbose: bool = True) -> SimulatedData:
    """
    Run full simulation pipeline.

    Args:
        seed: random seed
        theta_mode: "linear" (Gaussian random walk) or "nonlinear" (sin perturbation)
        obs_noise: "gaussian" or "heavy_tail" (Student-t)
        heavy_tail_df: degrees of freedom for heavy-tailed noise
        verbose: if True, print summary (set False for batch MC runs)
    """
    params = DGPParams()
    # v4.0: lambda calibrated per latent-process mode so each yields ~4% at N=2000.
    if theta_mode == "nonlinear":
        params.lambda_weibull = LAMBDA_V4_NONLINEAR
    else:
        params.lambda_weibull = LAMBDA_V4_LINEAR
    if verbose:
        print("=" * 60)
        print(f"Simulating maternal risk data (DGP v4.0, {theta_mode}/{obs_noise})")
        print("=" * 60)

    # 1. Baseline covariates
    X = generate_baseline_covariates(params.N, seed=seed)

    # 2. Latent trajectories (independent random stream — Bugfix #1)
    theta = generate_latent_trajectories(params.N, params.T_weeks,
                                          params.sigma2_omega, seed=seed + 1,
                                          mode=theta_mode)

    # 3. Observations (independent random stream — Bugfix #1)
    if obs_noise == "heavy_tail":
        Y = generate_observations_heavy_tail(theta, params.alpha,
                                             params.Sigma_eps, df=heavy_tail_df,
                                             seed=seed + 2)
    else:
        Y = generate_observations(theta, params.alpha, params.Sigma_eps,
                                  seed=seed + 2)

    # 4. Event times (independent random stream — Bugfix #1)
    event_times, event_indicators = generate_event_times(
        theta, X, params, seed=seed + 3)

    data = SimulatedData(
        theta=theta, Y=Y, X=X,
        event_times=event_times,
        event_indicators=event_indicators,
        params=params
    )
    if verbose:
        data.summary()
    return data


if __name__ == "__main__":
    data = simulate()
    # Quick sanity check (v4.0: N=2000, 4 covariates)
    assert data.theta.shape == (40, N_V4)
    assert data.Y.shape == (40, N_V4, 4)
    assert data.X.shape == (N_V4, 4)
    assert set(np.unique(data.X[:, 2])).issubset({0.0, 1.0})   # age_ge33 binary
    assert set(np.unique(data.X[:, 3])).issubset({0.0, 1.0})   # prior_pe binary
    assert data.event_times.shape == (N_V4,)
    assert data.event_indicators.shape == (N_V4,)
    print(f"prior_pe prevalence: {data.X[:, 3].mean():.3f} (target {PRIOR_PE_PREVALENCE})")
    print(f"age_ge33 prevalence: {data.X[:, 2].mean():.3f}")
    print("\nAll shape checks passed.")
