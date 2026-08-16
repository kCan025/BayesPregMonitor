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

def static_cox_benchmark(data: SimulatedData, eval_times: List[int] = None) -> Dict:
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
        data.event_times, data.event_indicators, risk_scores,
        eval_times=eval_times)

    # Bootstrap CI
    rng = np.random.RandomState(42)
    N = len(data.event_times)
    iAUCs = []
    for b in range(200):
        idx = rng.choice(N, size=N, replace=True)
        auc_b = time_dependent_auc(
            data.event_times[idx], data.event_indicators[idx],
            risk_scores[idx], eval_times=eval_times)
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

    # Use eta_all as risk scores (PF posterior mean — no Cox adjustment)
    risk_scores_tv = eta_all.copy()  # (T, N)

    # Compute time-dependent AUC at landmarks (primary iAUC, pair-weighted)
    auc_result = time_dependent_auc(
        data.event_times, data.event_indicators, risk_scores_tv,
        eval_times=landmark_times, delta_t=delta_t)

    # Per-landmark case/control counts for diagnostics
    landmark_aucs = []
    landmark_details = []
    print(f"\n--- Landmarking ---")
    for t_L in landmark_times:
        t_end = t_L + delta_t
        case_mask = ((data.event_times > t_L) &
                     (data.event_times <= t_end) &
                     (data.event_indicators == 1))
        control_mask = data.event_times > t_end
        n_cases = int(case_mask.sum())
        n_controls = int(control_mask.sum())
        print(f"  Landmark t={t_L}w: cases={n_cases}, controls={n_controls}")
        landmark_details.append({
            'landmark': t_L,
            'n_cases': n_cases,
            'n_controls': n_controls,
        })

    print(f"  iAUC (at landmarks, pair-weighted): {auc_result['iAUC']:.4f}")

    return {
        'landmark_aucs': landmark_aucs,
        'landmark_details': landmark_details,
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
    interval: int = 4,
    eval_times: List[int] = None
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
        data.event_times, data.event_indicators, eta_interp,
        eval_times=eval_times)

    # Bootstrap CI
    rng = np.random.RandomState(42)
    N = len(data.event_times)
    iAUCs = []
    for b in range(200):
        idx = rng.choice(N, size=N, replace=True)
        auc_b = time_dependent_auc(
            data.event_times[idx], data.event_indicators[idx],
            eta_interp[:, idx], eval_times=eval_times)
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
    eta_all: np.ndarray,
    particles_all: np.ndarray = None,
    weights_all: np.ndarray = None,
    M_mi: int = 5,
    eval_times: List[int] = None
) -> Dict:
    """
    Full high-freq pipeline: PF -> MI Cox -> risk score -> AUC.

    Risk score = beta_eta * eta(t) + beta_X' * (X - X_bar), incorporating
    both the latent state trajectory and baseline covariate adjustment.
    This differentiates HF from Landmarking, which uses raw eta only.

    MI (M=5) propagates PF uncertainty into Cox coefficients.
    """
    print(f"\n--- High-Freq PF + Cox (OUR METHOD) ---")

    from scipy.stats import pearsonr
    r, _ = pearsonr(data.theta.flatten(), eta_all.flatten())
    print(f"  PF correlation: r = {r:.4f}")

    # ---- MI Cox: propagate PF uncertainty into coefficients ----
    from time_varying_cox import fit_with_multiple_imputation
    mi_result = fit_with_multiple_imputation(
        data.event_times, data.event_indicators,
        particles_all, weights_all, data.X, M=M_mi)
    beta_eta = mi_result['beta_eta']
    beta_X = mi_result['beta_X']

    # ---- Construct Cox risk score: z(t) = beta_eta*eta(t) + beta_X'*(X-Xbar) ----
    X_centered = data.X - data.X.mean(axis=0)
    risk_score = beta_eta * eta_all + (X_centered @ beta_X)[None, :]  # (T, N)

    # ---- Primary AUC: Cox risk score ----
    auc_result = time_dependent_auc(
        data.event_times, data.event_indicators, risk_score,
        eval_times=eval_times)

    # Bootstrap 95% CI
    rng = np.random.RandomState(42)
    N = len(data.event_times)
    iAUCs = []
    for b in range(200):
        idx = rng.choice(N, size=N, replace=True)
        auc_b = time_dependent_auc(
            data.event_times[idx], data.event_indicators[idx],
            risk_score[:, idx], eval_times=eval_times)
        if not np.isnan(auc_b['iAUC']):
            iAUCs.append(auc_b['iAUC'])

    ci_lower = np.percentile(iAUCs, 2.5) if iAUCs else np.nan
    ci_upper = np.percentile(iAUCs, 97.5) if iAUCs else np.nan

    tag = f" at landmarks {eval_times}" if eval_times else ""
    print(f"  iAUC (Cox risk score{tag}) = {auc_result['iAUC']:.4f} "
          f"[95% CI: {ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  beta_eta={beta_eta:.4f}, beta_X={beta_X}")

    # ---- Supplementary: MI trajectory AUC variability ----
    # Uses same Cox coefficients, but eta_m trajectories vary -> shows PF uncertainty
    eta_m_list = None
    if particles_all is not None and weights_all is not None and M_mi >= 2:
        from bootstrap_pf import trajectory_mi_all_subjects
        eta_m_list = trajectory_mi_all_subjects(particles_all, weights_all, M=M_mi, seed=42)

        mi_aucs = []
        for eta_m in eta_m_list:
            rs_m = beta_eta * eta_m + (X_centered @ beta_X)[None, :]
            auc_m = time_dependent_auc(
                data.event_times, data.event_indicators, rs_m,
                eval_times=eval_times)
            if not np.isnan(auc_m['iAUC']):
                mi_aucs.append(auc_m['iAUC'])

        if mi_aucs:
            print(f"  [Supplementary] MI-pooled iAUC (M={M_mi}): "
                  f"{np.mean(mi_aucs):.4f} (SD={np.std(mi_aucs, ddof=1):.4f})")

    return {
        'auc_result': auc_result,
        'iAUC': auc_result['iAUC'],
        'ci_95': (ci_lower, ci_upper),
        'correlation_r': r,
        'beta_eta': beta_eta,
        'beta_X': beta_X,
        'method': 'High-Freq PF+Cox',
        'eta_m_list': eta_m_list,
        'eval_times': eval_times,
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
        if name.startswith('__'):  # skip internal metrics
            continue
        iAUC = res['iAUC']
        if 'ci_95' in res and not np.isnan(res['ci_95'][0]):
            ci_str = f"[{res['ci_95'][0]:.3f}, {res['ci_95'][1]:.3f}]"
        else:
            ci_str = "[N/A, N/A]"
        print(f"{name:<25} {iAUC:>8.4f} {ci_str:>18} {'4 wks':>8}")

    print("-" * 75)

    # Compute improvement over best baseline
    methods = [k for k in results.keys() if not k.startswith('__')]
    our_method = [m for m in methods if 'High-Freq' in m]
    baselines = [m for m in methods if 'High-Freq' not in m]

    if our_method and baselines:
        our_iAUC = results[our_method[0]]['iAUC']
        best_baseline = max(results[m]['iAUC'] for m in baselines)
        improvement = (our_iAUC - best_baseline) / best_baseline * 100
        print(f"\nImprovement over best baseline: +{improvement:.1f}%")


# =============================================================================
# CONTINUOUS MONITORING METRICS (HF's unique value proposition)
# =============================================================================

def continuous_monitoring_metrics(
    data: SimulatedData,
    eta_all: np.ndarray,
    eval_times: List[int] = None,
    delta_t: int = 4
) -> Dict:
    """
    Quantify High-Freq's unique advantage: continuous risk updating
    between clinic visits, and earlier detection of high-risk subjects.

    Key metrics:
    1. Weekly AUC trajectory: HF provides a valid risk score every week,
       while LM only scores at discrete landmarks.
    2. Detection lead time: for subjects who eventually experience the
       event, how many weeks earlier does HF flag them as high-risk
       compared to the nearest landmark assessment?
    3. Coverage gap: fraction of event-window weeks where LM has no
       score but HF does.

    Args:
        data: SimulatedData
        eta_all: (T, N) weekly risk scores from PF
        eval_times: landmark times (default [20, 28, 32])
        delta_t: prediction window

    Returns:
        dict with continuous monitoring metrics
    """
    if eval_times is None:
        eval_times = [20, 28, 32]

    T, N = eta_all.shape

    # ---- 1. Weekly AUC trajectory ----
    # Compute AUC at every week from 12 to 36 (where data supports it)
    weekly_eval = list(range(12, 37))
    weekly_aucs = []
    for t in weekly_eval:
        auc_t, n_cases, n_controls = incident_dynamic_auc_single(
            data.event_times, data.event_indicators,
            eta_all[t - 1, :], t, delta_t)
        if not np.isnan(auc_t) and n_cases >= 2 and n_controls >= 2:
            weekly_aucs.append({'time': t, 'auc': auc_t,
                                'n_cases': n_cases, 'n_controls': n_controls})

    # AUC at landmark vs between landmarks
    landmark_set = set(eval_times)
    lm_aucs = [w['auc'] for w in weekly_aucs if w['time'] in landmark_set]
    between_lm_aucs = [w['auc'] for w in weekly_aucs
                       if w['time'] not in landmark_set
                       and min(eval_times) <= w['time'] <= max(eval_times)]

    # ---- 2. Detection lead time ----
    # For each event subject: find earliest week where they exceed a
    # risk threshold (median of event subjects at that week), compare
    # to nearest landmark
    threshold_prob = 0.5  # use median as threshold
    event_mask = data.event_indicators == 1
    event_indices = np.where(event_mask)[0]

    lead_times = []
    for i in event_indices:
        event_time = int(data.event_times[i])
        # Find earliest week where this subject's eta exceeds median
        # among all subjects at that week
        flagged_week = None
        for t in range(12, min(event_time, T) + 1):
            eta_t = eta_all[t - 1, :]
            threshold = np.median(eta_t)
            if eta_all[t - 1, i] > threshold:
                flagged_week = t
                break

        if flagged_week is not None:
            # Compare to nearest landmark BEFORE the flagged week
            nearest_lm_before = max([lm for lm in eval_times if lm <= flagged_week],
                                    default=None)
            if nearest_lm_before is not None:
                lead_times.append(flagged_week - nearest_lm_before)

    mean_lead_time = np.mean(lead_times) if lead_times else 0.0
    median_lead_time = np.median(lead_times) if lead_times else 0.0

    # ---- 3. Coverage gap ----
    # Weeks in [min(eval_times), max(eval_times)] NOT covered by landmarks
    all_weeks_range = list(range(min(eval_times), max(eval_times) + 1))
    uncovered = [w for w in all_weeks_range if w not in landmark_set]
    coverage_gap_pct = len(uncovered) / len(all_weeks_range) * 100

    # ---- Print summary ----
    print(f"\n{'='*60}")
    print("CONTINUOUS MONITORING ADVANTAGE (High-Freq vs Landmarking)")
    print(f"{'='*60}")
    print(f"  Landmark times: {eval_times}")
    print(f"  Weeks in landmark range: {len(all_weeks_range)}")
    print(f"  Weeks covered by LM: {len(eval_times)}")
    print(f"  Weeks LM is silent: {len(uncovered)} ({coverage_gap_pct:.0f}%)")
    print(f"\n  Weekly AUC at landmarks:     "
          f"{np.mean(lm_aucs):.4f} (n={len(lm_aucs)})" if lm_aucs else
          "  Weekly AUC at landmarks:     N/A")
    print(f"  Weekly AUC between landmarks: "
          f"{np.mean(between_lm_aucs):.4f} (n={len(between_lm_aucs)})"
          if between_lm_aucs else
          "  Weekly AUC between landmarks: N/A")
    print(f"\n  Detection lead time (event subjects flagged early):")
    print(f"    Mean: {mean_lead_time:.1f} weeks")
    print(f"    Median: {median_lead_time:.1f} weeks")
    print(f"    n subjects: {len(lead_times)}")

    return {
        'weekly_aucs': weekly_aucs,
        'lm_week_aucs': lm_aucs,
        'between_lm_aucs': between_lm_aucs,
        'mean_lead_time_weeks': float(mean_lead_time),
        'median_lead_time_weeks': float(median_lead_time),
        'n_leading_subjects': len(lead_times),
        'coverage_gap_pct': coverage_gap_pct,
        'landmark_times': eval_times,
    }


def run_all_benchmarks(
    data: SimulatedData,
    eta_all: np.ndarray,
    particles_all: np.ndarray = None,
    weights_all: np.ndarray = None,
    M_mi: int = 5
) -> Dict:
    """
    Run all benchmark methods and return comparison.

    Fair comparison: all methods evaluated at the SAME landmark times
    [20, 28, 32].  Additionally, continuous monitoring metrics quantify
    High-Freq's unique value between landmarks.
    """
    # Unified evaluation times for fair comparison
    landmark_times = [20, 28, 32]
    results = {}

    # 1. High-freq pipeline (our method)
    results['High-Freq PF+Cox'] = high_freq_pipeline_benchmark(
        data, eta_all, particles_all, weights_all, M_mi=M_mi,
        eval_times=landmark_times)

    # 2. Static Cox (baseline only)
    results['Static Cox'] = static_cox_benchmark(data, eval_times=landmark_times)

    # 3. Landmarking (already uses [20, 28, 32] by default)
    results['Landmarking'] = landmarking_benchmark(
        data, eta_all, landmark_times=landmark_times)

    # 4. Low-freq pipeline
    results['Low-Freq PF+Cox'] = low_freq_pipeline_benchmark(
        data, interval=4, eval_times=landmark_times)

    # Print fair comparison table
    print_comparison_table(results)

    # 5. Continuous monitoring metrics (HF's unique advantage)
    cm_metrics = continuous_monitoring_metrics(
        data, eta_all, eval_times=landmark_times)

    results['__continuous_monitoring__'] = cm_metrics

    return results
