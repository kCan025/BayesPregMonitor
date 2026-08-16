"""
benchmarks.py -- Benchmark methods for comparison with the high-frequency pipeline
==================================================================================
Implements four methods for time-dependent risk prediction:

  1. High-Freq PF + Cox (OUR METHOD): weekly data, Bootstrap PF -> extended Cox
  2. Static Cox: baseline covariates only (age, BMI), no longitudinal data
  3. Landmarking: discrete-time prediction at weeks 20, 28, 32
  4. Low-Freq PF + Cox: 4-week downsampled data, same PF pipeline

Primary metric: incident/dynamic time-dependent AUC (Heagerty & Zheng, 2005)
  - At each evaluation time t, predict event in (t, t+delta_t]
  - Cases: event in window; Controls: survived past window
  - Integrated AUC (iAUC): weighted average across evaluation times

Also computes:
  - Bootstrap 95% CI for iAUC
  - Calibration (observed vs predicted event rates)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

from simulate_data import SimulatedData, DGPParams
from bootstrap_pf import BootstrapPF


# =============================================================================
# TIME-DEPENDENT AUC
# =============================================================================

def incident_dynamic_auc_single(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    risk_scores: np.ndarray,
    eval_time: int,
    delta_t: int = 4
) -> Tuple[float, int, int]:
    """
    Compute incident/dynamic AUC at a single evaluation time.

    At time t:
      Cases: subjects with event in (t, t+delta_t]
      Controls: subjects with T > t+delta_t (survived past window)
      Excluded: censored before t, censored in (t, t+delta_t]

    Args:
        event_times: (N,) observed times
        event_indicators: (N,) 1=event, 0=censored
        risk_scores: (N,) risk scores at eval_time (higher = higher risk)
        eval_time: evaluation time point
        delta_t: prediction window width

    Returns:
        (auc, n_cases, n_controls)
    """
    t = eval_time
    t_end = t + delta_t

    # Cases: event in (t, t+delta_t]
    case_mask = ((event_times > t) & (event_times <= t_end) &
                 (event_indicators == 1))

    # Controls: survived past t+delta_t
    control_mask = event_times > t_end

    case_scores = risk_scores[case_mask]
    control_scores = risk_scores[control_mask]

    n_cases = len(case_scores)
    n_controls = len(control_scores)

    if n_cases == 0 or n_controls == 0:
        return np.nan, n_cases, n_controls

    # Concordance probability (Wilcoxon-Mann-Whitney)
    concordant = 0.0
    tied = 0.0
    for cs in case_scores:
        concordant += np.sum(cs > control_scores)
        tied += np.sum(cs == control_scores)

    auc = (concordant + 0.5 * tied) / (n_cases * n_controls)
    return auc, n_cases, n_controls


def time_dependent_auc(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    risk_scores: np.ndarray,
    eval_times: Optional[List[int]] = None,
    delta_t: int = 4,
    min_pairs: int = 5
) -> Dict:
    """
    Compute time-dependent AUC at multiple evaluation times.

    Args:
        event_times: (N,) observed times
        event_indicators: (N,) 1=event, 0=censored
        risk_scores: (N,) for static methods, or (T, N) for time-varying
                     If 2D, uses risk_scores[t-1, :] at each eval_time t
        eval_times: list of evaluation times (default: every 4 weeks from 12 to 36)
        delta_t: prediction window (default: 4 weeks)
        min_pairs: minimum cases/controls needed to compute AUC

    Returns:
        dict with:
            'eval_times': list of evaluation times
            'auc_values': AUC at each time
            'iAUC': integrated AUC (weighted average)
            'n_cases': number of cases at each time
            'n_controls': number of controls at each time
            'n_pairs': total comparable pairs at each time
    """
    if eval_times is None:
        # Evaluate every 4 weeks from week 12 to week 36
        eval_times = list(range(12, 37, 4))

    # If risk_scores is 2D, filter eval_times to those available in the array
    if risk_scores.ndim == 2:
        max_t = risk_scores.shape[0]  # number of time points available
        # Only keep eval_times that are <= max_t
        valid_eval_times = [t for t in eval_times if t <= max_t]
        if len(valid_eval_times) < len(eval_times):
            warnings.warn(f"Some evaluation times exceed risk_scores time dimension "
                          f"({max_t}); they will be skipped.")
        eval_times = valid_eval_times

    auc_values = []
    n_cases_list = []
    n_controls_list = []
    n_pairs_list = []
    valid_times = []

    for t in eval_times:
        # Get risk scores at this time
        if risk_scores.ndim == 2:
            # Index is t-1 because weeks are 1-indexed
            idx = t - 1
            scores_t = risk_scores[idx, :]
        else:
            scores_t = risk_scores

        auc, n_cases, n_controls = incident_dynamic_auc_single(
            event_times, event_indicators, scores_t, t, delta_t)

        if not np.isnan(auc) and n_cases >= min_pairs and n_controls >= min_pairs:
            auc_values.append(auc)
            valid_times.append(t)
            n_cases_list.append(n_cases)
            n_controls_list.append(n_controls)
            n_pairs_list.append(n_cases * n_controls)

    if len(auc_values) == 0:
        return {
            'eval_times': [],
            'auc_values': [],
            'iAUC': np.nan,
            'n_cases': [],
            'n_controls': [],
            'n_pairs': []
        }

    # Integrated AUC: weighted average (weight by number of comparable pairs)
    weights = np.array(n_pairs_list, dtype=float)
    iAUC = np.average(auc_values, weights=weights)

    return {
        'eval_times': valid_times,
        'auc_values': auc_values,
        'iAUC': iAUC,
        'n_cases': n_cases_list,
        'n_controls': n_controls_list,
        'n_pairs': n_pairs_list
    }


def bootstrap_auc_ci(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    risk_scores: np.ndarray,
    eval_times: Optional[List[int]] = None,
    delta_t: int = 4,
    n_bootstrap: int = 200,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap 95% CI for iAUC.

    Returns:
        (iAUC, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    N = len(event_times)
    iAUCs = []

    for b in range(n_bootstrap):
        idx = rng.choice(N, size=N, replace=True)
        auc_result = time_dependent_auc(
            event_times[idx], event_indicators[idx],
            risk_scores[:, idx] if risk_scores.ndim == 2 else risk_scores[idx],
            eval_times=eval_times, delta_t=delta_t)
        if not np.isnan(auc_result['iAUC']):
            iAUCs.append(auc_result['iAUC'])

    if len(iAUCs) == 0:
        return np.nan, np.nan, np.nan

    iAUCs = np.array(iAUCs)
    return (np.mean(iAUCs),
            np.percentile(iAUCs, 2.5),
            np.percentile(iAUCs, 97.5))


# =============================================================================
# BENCHMARK 1: STATIC COX (baseline covariates only)
# =============================================================================

def static_cox_benchmark(data: SimulatedData) -> Dict:
    from time_varying_cox import fit_cox_model

    N = len(data.event_times)

    start_list = []
    stop_list = []
    event_list = []
    X_list = []

    for i in range(N):
        t_obs = min(int(data.event_times[i]), data.params.T_weeks)
        for t in range(1, t_obs + 1):
            start_list.append(t - 1)
            stop_list.append(t)
            if t == t_obs and data.event_indicators[i] == 1:
                event_list.append(1)
            else:
                event_list.append(0)
            X_list.append(data.X[i])

    start = np.array(start_list)
    stop = np.array(stop_list)
    events = np.array(event_list)
    X_vals = np.array(X_list)

    # dummy_eta length must match X_vals rows (counting process format)
    dummy_eta = np.zeros(len(X_vals))

    result = fit_cox_model(start, stop, events, dummy_eta, X_vals)

    # Risk score: exp(beta_X' X_i) for each subject
    beta_X = result['beta_X']
    risk_scores = np.exp(data.X @ beta_X)  # (N,) constant over time

    # Compute time-dependent AUC
    auc_result = time_dependent_auc(
        data.event_times, data.event_indicators, risk_scores)

    # Bootstrap CI
    rng = np.random.RandomState(42)
    N = len(data.event_times)
    iAUCs = []
    for b in range(200):
        idx = rng.choice(N, size=N, replace=True)
        auc_b = time_dependent_auc(
            data.event_times[idx], data.event_indicators[idx],
            risk_scores[idx])
        if not np.isnan(auc_b['iAUC']):
            iAUCs.append(auc_b['iAUC'])

    ci_lower = np.percentile(iAUCs, 2.5) if iAUCs else np.nan
    ci_upper = np.percentile(iAUCs, 97.5) if iAUCs else np.nan

    print(f"\n--- Static Cox (baseline only) ---")
    print(f"  beta_X = {beta_X} (age, BMI)")
    print(f"  iAUC = {auc_result['iAUC']:.4f} "
          f"[95% CI: {ci_lower:.4f}, {ci_upper:.4f}]")

    return {
        'risk_scores': risk_scores,  # (N,) constant
        'auc_result': auc_result,
        'beta_X': beta_X,
        'se_X': result['se_X'],
        'iAUC': auc_result['iAUC'],
        'ci_95': (ci_lower, ci_upper),
        'method': 'Static Cox'
    }


# =============================================================================
# BENCHMARK 2: LANDMARKING
# =============================================================================

def landmarking_benchmark(
    data: SimulatedData,
    eta_all: np.ndarray,
    landmark_times: List[int] = None,
    delta_t: int = 4
) -> Dict:
    """
    Landmark analysis at specified time points.

    At each landmark t_L:
      1. Restrict to subjects at risk at t_L (T_i > t_L)
      2. Predictor: eta_i(t_L) (current risk score)
      3. Outcome: event in (t_L, t_L + delta_t]
      4. Compute AUC for this binary classification

    Returns:
        dict with per-landmark AUCs, average AUC, and risk scores
    """
    if landmark_times is None:
        landmark_times = [20, 28, 32]

    landmark_aucs = []
    landmark_details = []

    for t_L in landmark_times:
        t_end = t_L + delta_t

        # Cases: event in (t_L, t_L + delta_t]
        case_mask = ((data.event_times > t_L) &
                     (data.event_times <= t_end) &
                     (data.event_indicators == 1))

        # Controls: survived past t_L + delta_t
        control_mask = data.event_times > t_end

        # Risk scores at landmark time (1-indexed: week t_L -> index t_L-1)
        t_idx = min(t_L, eta_all.shape[0])
        scores_at_landmark = eta_all[t_idx - 1, :]

        # AUC for this landmark
        case_scores = scores_at_landmark[case_mask]
        control_scores = scores_at_landmark[control_mask]

        n_cases = len(case_scores)
        n_controls = len(control_scores)

        if n_cases > 0 and n_controls > 0:
            concordant = sum(np.sum(cs > control_scores) for cs in case_scores)
            tied = sum(np.sum(cs == control_scores) for cs in case_scores)
            auc = (concordant + 0.5 * tied) / (n_cases * n_controls)
        else:
            auc = np.nan

        landmark_aucs.append(auc)
        landmark_details.append({
            'landmark': t_L,
            'n_cases': n_cases,
            'n_controls': n_controls,
            'AUC': auc
        })

        print(f"  Landmark t={t_L}w: cases={n_cases}, controls={n_controls}, AUC={auc:.4f}")

    # Average AUC across landmarks
    valid_aucs = [a for a in landmark_aucs if not np.isnan(a)]
    avg_auc = np.mean(valid_aucs) if valid_aucs else np.nan

    # Use eta_all as risk scores for time-dependent AUC (already 40 weeks)
    risk_scores_tv = eta_all.copy()  # (T, N)

    # Compute overall time-dependent AUC using landmark-style evaluation
    auc_result = time_dependent_auc(
        data.event_times, data.event_indicators, risk_scores_tv,
        eval_times=landmark_times, delta_t=delta_t)

    print(f"\n--- Landmarking ---")
    print(f"  Landmark AUCs: {[f'{a:.4f}' for a in landmark_aucs]}")
    print(f"  Average landmark AUC: {avg_auc:.4f}")
    print(f"  iAUC (at landmarks): {auc_result['iAUC']:.4f}")

    return {
        'landmark_aucs': landmark_aucs,
        'landmark_details': landmark_details,
        'avg_auc': avg_auc,
        'auc_result': auc_result,
        'risk_scores': risk_scores_tv,
        'iAUC': auc_result['iAUC'],
        'method': 'Landmarking'
    }


# =============================================================================
# BENCHMARK 3: LOW-FREQUENCY PIPELINE (4-week downsampled)
# =============================================================================

def downsample_observations(Y: np.ndarray, interval: int = 4) -> np.ndarray:
    """
    Downsample wearable observations from weekly to every `interval` weeks.

    Args:
        Y: (T, N, D) weekly observations
        interval: downsampling interval (default: 4 weeks)

    Returns:
        Y_down: (T_ds, N, D) downsampled observations
    """
    T, N, D = Y.shape
    # Select time points: 0, interval, 2*interval, ...
    time_indices = list(range(0, T, interval))
    return Y[time_indices, :, :]


def low_freq_pipeline_benchmark(
    data: SimulatedData,
    interval: int = 4
) -> Dict:
    """
    Run the same PF + Cox pipeline on downsampled (4-weekly) data.
    This isolates the effect of data frequency from methodology.

    Returns:
        dict with AUC results and risk scores (interpolated to weekly grid)
    """
    print(f"\n--- Low-Freq Pipeline ({interval}-week intervals) ---")

    # 1. Downsample observations
    Y_down = downsample_observations(data.Y, interval=interval)
    T_ds = Y_down.shape[0]
    print(f"  Downsampled: {data.Y.shape[0]} -> {T_ds} time points")

    # 2. Run PF on downsampled data
    pf = BootstrapPF(
        alpha=data.params.alpha,
        sigma2_omega=data.params.sigma2_omega,
        Sigma_eps=data.params.Sigma_eps,
        n_particles=100
    )

    eta_down, particles_down, weights_down = pf.filter_all(Y_down)
    print(f"  PF on downsampled data completed")

    # Correlation check
    from scipy.stats import pearsonr
    ds_time_indices = list(range(0, data.params.T_weeks, interval))
    theta_ds = data.theta[ds_time_indices, :]
    r_down, _ = pearsonr(theta_ds.flatten(), eta_down.flatten())
    print(f"  Correlation(theta, eta_down): r = {r_down:.4f}")

    # 3. Fit Cox model on downsampled eta
    from time_varying_cox import (prepare_counting_process_data,
                                   fit_cox_model,
                                   fit_with_multiple_imputation)

    # Naive Cox
    start, stop, events, eta_vals, X_vals = prepare_counting_process_data(
        data.event_times, data.event_indicators, eta_down, data.X)
    cox_result = fit_cox_model(start, stop, events, eta_vals, X_vals)
    print(f"  Cox beta_eta = {cox_result['beta_eta']:.4f} "
          f"(true gamma = {data.params.gamma})")

    # 4. Interpolate eta to weekly grid for AUC computation
    T_full = data.params.T_weeks
    eta_interp = np.zeros((T_full, data.params.N))
    # Assign each downsampled value to all weeks until next downsampled point
    for j in range(T_ds):
        t_start = ds_time_indices[j]  # 0,4,8,...
        if j + 1 < T_ds:
            t_end = ds_time_indices[j + 1]
        else:
            t_end = T_full
        eta_interp[t_start:t_end, :] = eta_down[j, :]

    # 5. Compute time-dependent AUC
    auc_result = time_dependent_auc(
        data.event_times, data.event_indicators, eta_interp)

    # Bootstrap CI
    rng = np.random.RandomState(42)
    N = len(data.event_times)
    iAUCs = []
    for b in range(200):
        idx = rng.choice(N, size=N, replace=True)
        auc_b = time_dependent_auc(
            data.event_times[idx], data.event_indicators[idx],
            eta_interp[:, idx])
        if not np.isnan(auc_b['iAUC']):
            iAUCs.append(auc_b['iAUC'])

    ci_lower = np.percentile(iAUCs, 2.5) if iAUCs else np.nan
    ci_upper = np.percentile(iAUCs, 97.5) if iAUCs else np.nan

    print(f"  iAUC = {auc_result['iAUC']:.4f} "
          f"[95% CI: {ci_lower:.4f}, {ci_upper:.4f}]")

    return {
        'eta_down': eta_down,
        'eta_interp': eta_interp,
        'auc_result': auc_result,
        'correlation_r': r_down,
        'cox_beta_eta': cox_result['beta_eta'],
        'iAUC': auc_result['iAUC'],
        'ci_95': (ci_lower, ci_upper),
        'method': 'Low-Freq PF+Cox'
    }


# =============================================================================
# BENCHMARK 4: HIGH-FREQ PIPELINE (already computed in Step 1)
# =============================================================================

def high_freq_pipeline_benchmark(
    data: SimulatedData,
    eta_all: np.ndarray
) -> Dict:
    """
    Compute time-dependent AUC for the high-frequency PF + Cox pipeline.

    Uses the eta trajectories from Step 1.

    Returns:
        dict with AUC results
    """
    print(f"\n--- High-Freq PF + Cox (OUR METHOD) ---")

    # Compute time-dependent AUC using eta as time-varying risk score
    auc_result = time_dependent_auc(
        data.event_times, data.event_indicators, eta_all)

    # Bootstrap CI
    rng = np.random.RandomState(42)
    N = len(data.event_times)
    iAUCs = []
    for b in range(200):
        idx = rng.choice(N, size=N, replace=True)
        auc_b = time_dependent_auc(
            data.event_times[idx], data.event_indicators[idx],
            eta_all[:, idx])
        if not np.isnan(auc_b['iAUC']):
            iAUCs.append(auc_b['iAUC'])

    ci_lower = np.percentile(iAUCs, 2.5) if iAUCs else np.nan
    ci_upper = np.percentile(iAUCs, 97.5) if iAUCs else np.nan

    # Correlation
    from scipy.stats import pearsonr
    r, _ = pearsonr(data.theta.flatten(), eta_all.flatten())

    print(f"  PF correlation: r = {r:.4f}")
    print(f"  iAUC = {auc_result['iAUC']:.4f} "
          f"[95% CI: {ci_lower:.4f}, {ci_upper:.4f}]")

    return {
        'auc_result': auc_result,
        'iAUC': auc_result['iAUC'],
        'ci_95': (ci_lower, ci_upper),
        'correlation_r': r,
        'method': 'High-Freq PF+Cox'
    }


# =============================================================================
# SUMMARY COMPARISON
# =============================================================================

def print_comparison_table(results: Dict):
    """Print formatted comparison table."""
    print("\n" + "=" * 75)
    print("PERFORMANCE COMPARISON")
    print("=" * 75)
    print(f"{'Method':<25} {'iAUC':>8} {'95% CI':>18} {'Delta_t':>8}")
    print("-" * 75)

    for name, res in results.items():
        iAUC = res['iAUC']
        if 'ci_95' in res and not np.isnan(res['ci_95'][0]):
            ci_str = f"[{res['ci_95'][0]:.3f}, {res['ci_95'][1]:.3f}]"
        else:
            ci_str = "[N/A, N/A]"
        print(f"{name:<25} {iAUC:>8.4f} {ci_str:>18} {'4 wks':>8}")

    print("-" * 75)

    # Compute improvement over best baseline
    methods = list(results.keys())
    our_method = [m for m in methods if 'High-Freq' in m]
    baselines = [m for m in methods if 'High-Freq' not in m]

    if our_method and baselines:
        our_iAUC = results[our_method[0]]['iAUC']
        best_baseline = max(results[m]['iAUC'] for m in baselines)
        improvement = (our_iAUC - best_baseline) / best_baseline * 100
        print(f"\nImprovement over best baseline: +{improvement:.1f}%")


def run_all_benchmarks(
    data: SimulatedData,
    eta_all: np.ndarray,
    particles_all: np.ndarray = None,
    weights_all: np.ndarray = None
) -> Dict:
    """
    Run all benchmark methods and return comparison.

    Args:
        data: SimulatedData from Step 1
        eta_all: (T, N) filtered risk scores from Step 1
        particles_all: (T, N, n_particles) from Step 1 (optional)
        weights_all: (T, N, n_particles) from Step 1 (optional)

    Returns:
        dict with results for each method
    """
    results = {}

    # 1. High-freq pipeline (our method)
    results['High-Freq PF+Cox'] = high_freq_pipeline_benchmark(data, eta_all)

    # 2. Static Cox (baseline only)
    results['Static Cox'] = static_cox_benchmark(data)

    # 3. Landmarking
    results['Landmarking'] = landmarking_benchmark(data, eta_all)

    # 4. Low-freq pipeline
    results['Low-Freq PF+Cox'] = low_freq_pipeline_benchmark(data, interval=4)

    # Print comparison
    print_comparison_table(results)

    return results
