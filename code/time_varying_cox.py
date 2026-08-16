"""
time_varying_cox.py -- Time-varying Cox model with eta_i(t) as covariate
========================================================================
Fits extended Cox proportional hazards model:
  lambda_i(t) = lambda_0(t) * exp(beta_1 * eta_i(t) + beta_2' X_i)

Uses counting process format (start, stop, event) for time-varying covariates.
Supports multiple imputation (M=5) with Rubin's rules for pooling.

v3.4: Fixed risk set definition -- at_risk = (start < t_k) & (stop >= t_k)
"""

import numpy as np
from typing import Tuple
import warnings


def prepare_counting_process_data(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    eta: np.ndarray,
    X: np.ndarray,
    max_time: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert subject-level data to counting process (start, stop, event) format.

    Each subject i contributes T_i rows (one per time interval).

    Args:
        event_times: (N,) observed event/censoring times
        event_indicators: (N,) 1=event, 0=censored
        eta: (T_eta, N) time-varying risk scores
        X: (N, p) baseline covariates
        max_time: int or None. If None, inferred from eta.shape[0].
                  If provided, capped at eta.shape[0] to prevent IndexError.

    Returns:
        start_times, stop_times, event_flags, eta_values, X_values
    """
    N = len(event_times)
    p = X.shape[1]

    # Robustly determine max time: infer from eta shape, never exceed it
    T_eta = eta.shape[0]
    if max_time is None:
        max_time = T_eta
    else:
        max_time = min(max_time, T_eta)

    start_list = []
    stop_list = []
    event_list = []
    eta_list = []
    X_list = []

    for i in range(N):
        t_obs = min(int(event_times[i]), max_time)

        for t in range(1, t_obs + 1):
            start_list.append(t - 1)
            stop_list.append(t)

            # Event indicator: only 1 in the last interval if event occurred
            if t == t_obs and event_indicators[i] == 1:
                event_list.append(1)
            else:
                event_list.append(0)

            # eta value at start of interval (use t-1 index into 0-based eta)
            eta_list.append(eta[t - 1, i])

            # Baseline covariates (constant)
            X_list.append(X[i])

    return (
        np.array(start_list),
        np.array(stop_list),
        np.array(event_list),
        np.array(eta_list),
        np.array(X_list)
    )


def fit_cox_model(
    start_times: np.ndarray,
    stop_times: np.ndarray,
    event_flags: np.ndarray,
    eta_values: np.ndarray,
    X_values: np.ndarray
) -> dict:
    """
    Fit Cox PH model using partial likelihood (Breslow method).
    Model: h(t|z) = h_0(t) * exp(beta_eta * eta + beta_X' X)

    v3.4: Fixed risk set -- at_risk = (start < t_k) & (stop >= t_k)

    Returns:
        dict with keys: 'beta_eta', 'beta_X', 'se_eta', 'se_X',
                        'log_likelihood', 'converged'
    """
    from scipy.optimize import minimize

    n_rows = len(start_times)
    p_X = X_values.shape[1]
    n_params = 1 + p_X  # beta_eta + beta_X

    # Design matrix: [eta, X]
    Z = np.column_stack([eta_values, X_values])  # (n_rows, n_params)

    # Get unique event times for partial likelihood
    event_mask = event_flags == 1
    event_stop_times = stop_times[event_mask]
    unique_event_times = np.sort(np.unique(event_stop_times))

    def neg_partial_log_likelihood(beta):
        """Negative partial log-likelihood (Breslow method)."""
        risk_scores = Z @ beta

        log_lik = 0.0
        for t_k in unique_event_times:
            # v3.4: Correct risk set definition
            # Includes individuals who are still under observation at t_k,
            # including those who experience the event at t_k
            at_risk = (start_times < t_k) & (stop_times >= t_k)

            if not np.any(at_risk):
                continue

            events_here = event_mask & (stop_times == t_k)
            n_events = np.sum(events_here)

            log_risk_sum = np.log(np.sum(np.exp(risk_scores[at_risk])) + 1e-300)
            event_risk_sum = np.sum(risk_scores[events_here])

            log_lik += event_risk_sum - n_events * log_risk_sum

        return -log_lik

    def neg_gradient(beta):
        """Gradient of negative partial log-likelihood."""
        risk_scores = Z @ beta
        grad = np.zeros(n_params)

        for t_k in unique_event_times:
            # v3.4: Correct risk set
            at_risk = (start_times < t_k) & (stop_times >= t_k)
            events_here = event_mask & (stop_times == t_k)
            n_events = np.sum(events_here)

            if not np.any(at_risk):
                continue

            exp_risk = np.exp(risk_scores[at_risk])
            risk_sum = np.sum(exp_risk) + 1e-300

            weighted_Z = np.sum(Z[at_risk] * exp_risk[:, None], axis=0) / risk_sum
            event_Z_sum = np.sum(Z[events_here], axis=0)

            grad += n_events * weighted_Z - event_Z_sum

        return grad

    beta_init = np.zeros(n_params)

    result = minimize(
        neg_partial_log_likelihood,
        beta_init,
        jac=neg_gradient,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-8}
    )

    beta_hat = result.x

    # Standard errors from observed information (numerical Hessian)
    try:
        from scipy.optimize import approx_fprime
        eps = 1e-5
        hessian = np.zeros((n_params, n_params))
        for j in range(n_params):
            def grad_j(b):
                return neg_gradient(b)[j]
            hessian[j] = approx_fprime(beta_hat, grad_j, eps)
        cov_matrix = np.linalg.inv(hessian + 1e-8 * np.eye(n_params))
        se = np.sqrt(np.abs(np.diag(cov_matrix)))
    except Exception:
        se = np.full(n_params, np.nan)
        cov_matrix = np.full((n_params, n_params), np.nan)

    return {
        'beta_eta': beta_hat[0],
        'beta_X': beta_hat[1:],
        'se_eta': se[0],
        'se_X': se[1:],
        'cov_matrix': cov_matrix,
        'log_likelihood': -result.fun,
        'converged': result.success
    }


def fit_with_multiple_imputation(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    particles_all: np.ndarray,
    weights_all: np.ndarray,
    X: np.ndarray,
    M: int = 5,
    max_time: int = 40
) -> dict:
    """
    Fit time-varying Cox model with marginal multiple imputation.
    Pool via Rubin's rules.
    """
    T, N, Np = particles_all.shape
    p_X = X.shape[1]
    rng = np.random.RandomState(42)

    beta_eta_list = []
    beta_X_list = []
    var_eta_list = []
    var_X_list = []

    for m in range(M):
        # Construct imputed eta via trajectory-level MI
        # KEY FIX: draw ONE particle per (t,i), not mean of Np
        # This preserves posterior variance -> prevents Cox attenuation
        eta_imputed = np.zeros((T, N))
        for i in range(N):
            for t in range(T):
                j = rng.choice(Np, p=weights_all[t, i])
                eta_imputed[t, i] = particles_all[t, i, j]

        start, stop, events, eta_vals, X_vals = prepare_counting_process_data(
            event_times, event_indicators, eta_imputed, X, max_time)

        result = fit_cox_model(start, stop, events, eta_vals, X_vals)
        beta_eta_list.append(result['beta_eta'])
        beta_X_list.append(result['beta_X'])
        var_eta_list.append(result['se_eta'] ** 2)
        var_X_list.append(result['se_X'] ** 2)

        status = "converged" if result['converged'] else "NOT converged"
        print(f"  MI [{m+1}/{M}]: beta_eta={result['beta_eta']:.4f} "
              f"(se={result['se_eta']:.4f}), {status}")

    # Rubin's rules
    beta_eta_arr = np.array(beta_eta_list)
    beta_X_arr = np.array(beta_X_list)
    var_eta_arr = np.array(var_eta_list)
    var_X_arr = np.array(var_X_list)

    beta_eta_pooled = np.mean(beta_eta_arr)
    beta_X_pooled = np.mean(beta_X_arr, axis=0)

    W_eta = np.mean(var_eta_arr)
    W_X = np.mean(var_X_arr, axis=0)

    B_eta = np.var(beta_eta_arr, ddof=1)
    B_X = np.var(beta_X_arr, axis=0, ddof=1)

    T_eta = W_eta + (1 + 1.0 / M) * B_eta
    T_X = W_X + (1 + 1.0 / M) * B_X

    se_eta_pooled = np.sqrt(T_eta)
    se_X_pooled = np.sqrt(T_X)

    fmi_eta = ((1 + 1/M) * B_eta) / T_eta if T_eta > 0 else 0

    print(f"\n=== Rubin's Rules Pooling (M={M}) ===")
    print(f"Pooled beta_eta: {beta_eta_pooled:.4f} (SE={se_eta_pooled:.4f})")
    print(f"Within-imputation var: {W_eta:.6f}")
    print(f"Between-imputation var: {B_eta:.6f}")
    print(f"Total var: {T_eta:.6f}")
    print(f"FMI (eta): {fmi_eta:.3f}")

    return {
        'beta_eta': beta_eta_pooled,
        'beta_X': beta_X_pooled,
        'se_eta': se_eta_pooled,
        'se_X': se_X_pooled,
        'W_eta': W_eta,
        'B_eta': B_eta,
        'T_eta': T_eta,
        'fmi_eta': fmi_eta,
        'individual_results': {
            'beta_eta': beta_eta_arr,
            'beta_X': beta_X_arr,
        }
    }


def fit_naive_cox(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    eta: np.ndarray,
    X: np.ndarray,
    max_time: int = 40
) -> dict:
    """Fit Cox model using point estimates of eta (no MI)."""
    start, stop, events, eta_vals, X_vals = prepare_counting_process_data(
        event_times, event_indicators, eta, X, max_time)
    return fit_cox_model(start, stop, events, eta_vals, X_vals)
