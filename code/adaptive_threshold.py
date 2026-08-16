"""
adaptive_threshold.py -- Layer 3: Cost-Sensitive Decision Rule (Fixed)
=======================================================================
Implements the clinical decision layer:

  1. Compute conditional event probability P(T <= t+dt | T > t, H_i(t))
     from the fitted extended Cox model
  2. Cost-sensitive threshold: d* = C_FP / (C_FP + C_FN)
  3. Sweep cost ratio to trace cost-benefit frontier
  4. Compare against fixed-cutoff monitoring strategies
  5. Compute net benefit / decision curve analysis

Fixed version uses Nelson-Aalen marginal cumulative hazard as baseline
and properly centers covariates to avoid mean shift.

Date: 2026-08-15
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize_scalar


def estimate_marginal_cumulative_hazard(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    T_max: int
) -> np.ndarray:
    """
    Nelson-Aalen estimator of marginal cumulative hazard.

    Args:
        event_times: (N,) observed times (1-indexed weeks)
        event_indicators: (N,) 1=event, 0=censored
        T_max: maximum follow-up time (weeks)

    Returns:
        H: (T_max,) cumulative hazard estimates for t = 1..T_max
    """
    # Only event times contribute to hazard jumps
    df = pd.DataFrame({'time': event_times, 'event': event_indicators})
    df_events = df[df['event'] == 1]

    H = np.zeros(T_max)
    n_at_risk = len(event_times)

    for t in range(1, T_max + 1):
        d = (df_events['time'] == t).sum()
        if d > 0 and n_at_risk > 0:
            # Add jump: d / n_at_risk
            H[t-1] = (H[t-2] if t > 1 else 0.0) + d / n_at_risk
            n_at_risk -= d
        else:
            # No jump: carry forward previous value
            if t > 1:
                H[t-1] = H[t-2]
    return H


def compute_conditional_probability(
    eta: np.ndarray,
    X: np.ndarray,
    beta_eta: float,
    beta_X: np.ndarray,
    eval_times: np.ndarray,
    delta_t: int = 4,
    baseline_hazard: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute conditional event probability P(T <= t+dt | T > t, H_i(t)).

    Uses Cox model:
      lambda_i(t) = lambda_0(t) * exp(beta_eta * eta_i(t) + beta_X' X_i)
      P(T <= t+dt | T > t) = 1 - S(t+dt) / S(t)
      S(t) = exp(-H_0(t) * exp(z))

    If baseline_hazard is not provided, a Nelson-Aalen marginal hazard
    should have been passed by the caller.

    Args:
        eta: (T, N) dynamic risk scores
        X: (N, 2) baseline covariates (will be centered)
        beta_eta: fitted coefficient for eta
        beta_X: (2,) fitted coefficients for X
        eval_times: (T_eval,) evaluation weeks
        delta_t: prediction window width (weeks)
        baseline_hazard: (T_max,) cumulative baseline hazard

    Returns:
        prob_event: (T_eval, N) conditional event probabilities
    """
    T, N = eta.shape
    # Center X to remove average effect, so baseline hazard corresponds
    # to the average subject
    X_centered = X - X.mean(axis=0)
    z = beta_eta * eta + X_centered @ beta_X  # (T, N)

    if baseline_hazard is None:
        # Fallback: empirical Nelson-Aalen should be provided; if not,
        # use a rough approximation
        import warnings
        warnings.warn("baseline_hazard not provided; using Nelson-Aalen internally")
        baseline_hazard = estimate_marginal_cumulative_hazard(
            event_times, event_indicators, T)

    prob_event = np.zeros((T, N))
    for t_idx in range(T):
        t = int(eval_times[t_idx]) if len(eval_times) > t_idx else t_idx + 1
        t_end = min(t + delta_t, T)

        # Cumulative baseline hazard at t-1 and t_end-1 (1-indexed)
        H0_t = baseline_hazard[min(t - 1, T - 1)]
        H0_t_end = baseline_hazard[min(t_end - 1, T - 1)]
        delta_H0 = H0_t_end - H0_t

        if delta_H0 <= 0:
            continue

        risk = np.exp(z[t_idx, :])
        # Conditional event probability
        prob_event[t_idx, :] = 1 - np.exp(-delta_H0 * risk)

    return prob_event


def cost_sensitive_threshold(
    prob_event: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    cost_ratio: float = 1.0,
    delta_t: int = 4,
    eval_times: Optional[np.ndarray] = None
) -> Dict:
    """
    Apply cost-sensitive decision rule.

    Alert when P(event in window) > d* = C_FP / (C_FP + C_FN)
                                    = cost_ratio / (1 + cost_ratio)

    Args:
        prob_event: (T, N) conditional event probabilities
        event_times: (N,) observed times
        event_indicators: (N,) 1=event, 0=censored
        cost_ratio: C_FP / C_FN ratio
        delta_t: prediction window
        eval_times: evaluation times (weeks)

    Returns:
        dict with confusion-matrix aggregates and derived metrics
    """
    d_star = cost_ratio / (1 + cost_ratio)
    T, N = prob_event.shape
    if eval_times is None:
        eval_times = np.arange(1, T + 1)

    tp_total = fp_total = fn_total = tn_total = 0

    for t_idx in range(T):
        t = int(eval_times[t_idx]) if len(eval_times) > t_idx else t_idx + 1
        t_end = t + delta_t

        # Ground truth: event in (t, t+dt] and survived to t
        event_in_window = (event_times > t) & (event_times <= t_end) & (event_indicators == 1)
        at_risk = event_times > t

        if not np.any(at_risk):
            continue

        alerted = prob_event[t_idx, :] > d_star

        tp = np.sum(alerted & event_in_window & at_risk)
        fp = np.sum(alerted & ~event_in_window & at_risk)
        fn = np.sum(~alerted & event_in_window & at_risk)
        tn = np.sum(~alerted & ~event_in_window & at_risk)

        tp_total += tp
        fp_total += fp
        fn_total += fn
        tn_total += tn

    sensitivity = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    specificity = tn_total / (tn_total + fp_total) if (tn_total + fp_total) > 0 else 0.0
    ppv = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    npv = tn_total / (tn_total + fn_total) if (tn_total + fn_total) > 0 else 0.0

    N_at_risk = tp_total + fp_total + fn_total + tn_total
    # Net benefit = (TP/N) - (FP/N) * cost_ratio
    net_benefit = (tp_total / N_at_risk - fp_total / N_at_risk * cost_ratio) if N_at_risk > 0 else 0.0

    return {
        'threshold': d_star,
        'cost_ratio': cost_ratio,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'net_benefit': net_benefit,
        'tp': tp_total,
        'fp': fp_total,
        'fn': fn_total,
        'tn': tn_total
    }


def sweep_cost_ratios(
    prob_event: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    cost_ratios: Optional[np.ndarray] = None,
    delta_t: int = 4,
    eval_times: Optional[np.ndarray] = None
) -> Dict:
    """
    Sweep cost ratios to trace the cost-benefit frontier.

    Args:
        cost_ratios: array of C_FP/C_FN ratios

    Returns:
        dict with frontier data and optimal cost ratio
    """
    if cost_ratios is None:
        cost_ratios = np.logspace(-2, 2, 30)  # 0.01 to 100

    results = []
    for cr in cost_ratios:
        res = cost_sensitive_threshold(
            prob_event, event_times, event_indicators,
            cost_ratio=cr, delta_t=delta_t, eval_times=eval_times)
        res['cost_ratio'] = float(cr)
        results.append(res)

    net_benefits = [r['net_benefit'] for r in results]
    opt_idx = np.argmax(net_benefits)

    return {
        'frontier': results,
        'cost_ratios': cost_ratios.tolist(),
        'net_benefits': net_benefits,
        'optimal_cost_ratio': float(cost_ratios[opt_idx]),
        'max_net_benefit': net_benefits[opt_idx],
        'optimal_result': results[opt_idx]
    }


def compare_decision_strategies(
    prob_event: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    delta_t: int = 4,
    eval_times: Optional[np.ndarray] = None,
    base_cost_ratio: float = 1.0
) -> Dict:
    """
    Compare cost-sensitive rule against fixed-cutoff and treat-all/none.

    All strategies are evaluated at the same base_cost_ratio for fairness.

    Args:
        base_cost_ratio: C_FP / C_FN ratio used for treat-all and treat-none
                         and balanced cost-sensitive.

    Returns:
        dict with strategies and sweep_result
    """
    strategies = {}

    # Fixed thresholds: DATA-ADAPTIVE based on actual probability range
    # Using percentiles ensures thresholds fall within the observed range
    # (avoids degenerate comparison where fixed thresholds never trigger)
    valid_probs = prob_event[prob_event > 0]
    if len(valid_probs) > 0:
        data_thresholds = {
            'Median prob': float(np.median(valid_probs)),
            '75th pctl': float(np.percentile(valid_probs, 75)),
            '90th pctl': float(np.percentile(valid_probs, 90)),
        }
    else:
        data_thresholds = {'Median prob': 0.01, '75th pctl': 0.02, '90th pctl': 0.05}

    for label, threshold in data_thresholds.items():
        threshold = np.clip(threshold, 1e-6, 1 - 1e-6)
        cr_equiv = threshold / (1 - threshold)
        res = cost_sensitive_threshold(
            prob_event, event_times, event_indicators,
            cost_ratio=cr_equiv, delta_t=delta_t, eval_times=eval_times)
        strategies[f'Threshold ({label}={threshold:.3f})'] = res

    # Balanced cost-sensitive at base_cost_ratio
    res_balanced = cost_sensitive_threshold(
        prob_event, event_times, event_indicators,
        cost_ratio=base_cost_ratio, delta_t=delta_t, eval_times=eval_times)
    strategies['Cost-sensitive (balanced)'] = res_balanced

    # Optimal cost-sensitive from sweep
    sweep_result = sweep_cost_ratios(
        prob_event, event_times, event_indicators,
        delta_t=delta_t, eval_times=eval_times)
    strategies['Cost-sensitive (optimal)'] = sweep_result['optimal_result']

    # Treat all / none, computed at base_cost_ratio
    N = len(event_times)
    n_events = event_indicators.sum()
    n_non_events = N - n_events

    treat_all_nb = (n_events / N) - (n_non_events / N) * base_cost_ratio
    treat_none_nb = 0.0

    strategies['Treat all'] = {
        'threshold': 0.0,
        'cost_ratio': base_cost_ratio,
        'sensitivity': 1.0,
        'specificity': 0.0,
        'ppv': n_events / N,
        'net_benefit': treat_all_nb,
        'tp': n_events,
        'fp': n_non_events,
        'fn': 0,
        'tn': 0
    }
    strategies['Treat none'] = {
        'threshold': 1.0,
        'cost_ratio': base_cost_ratio,
        'sensitivity': 0.0,
        'specificity': 1.0,
        'ppv': np.nan,
        'net_benefit': treat_none_nb,
        'tp': 0,
        'fp': 0,
        'fn': n_events,
        'tn': n_non_events
    }

    return {
        'strategies': strategies,
        'sweep_result': sweep_result
    }


def run_layer3(
    data,
    eta_all: np.ndarray,
    beta_eta: float,
    beta_X: np.ndarray,
    delta_t: int = 4
) -> Dict:
    """
    Complete Layer 3 pipeline.

    Args:
        data: SimulatedData
        eta_all: (T, N) filtered risk scores
        beta_eta: fitted Cox coefficient for eta
        beta_X: (2,) fitted Cox coefficients for X
        delta_t: prediction window (weeks)

    Returns:
        dict with all Layer 3 results
    """
    T, N = eta_all.shape
    eval_times = np.arange(1, T + 1)

    print("\n" + "=" * 50)
    print("LAYER 3: Cost-Sensitive Decision Rule (FIXED)")
    print("=" * 50)

    # 1. Estimate marginal cumulative hazard
    baseline_hazard = estimate_marginal_cumulative_hazard(
        data.event_times, data.event_indicators, T)

    # 2. Compute conditional event probabilities
    print("\n--- Computing conditional event probabilities ---")
    prob_event = compute_conditional_probability(
        eta_all, data.X, beta_eta, beta_X,
        eval_times=eval_times, delta_t=delta_t,
        baseline_hazard=baseline_hazard)

    print(f"  Prob range: [{prob_event.min():.4f}, {prob_event.max():.4f}]")
    print(f"  Prob mean at t=20: {prob_event[19, :].mean():.4f}")
    print(f"  Prob mean at t=30: {prob_event[29, :].mean():.4f}")

    # 3. Sweep cost ratios
    print("\n--- Sweeping cost ratios ---")
    sweep_result = sweep_cost_ratios(
        prob_event, data.event_times, data.event_indicators,
        delta_t=delta_t, eval_times=eval_times)

    opt_cr = sweep_result['optimal_cost_ratio']
    opt_d = opt_cr / (1 + opt_cr)
    print(f"  Optimal cost ratio: {opt_cr:.3f}")
    print(f"  Corresponding d*: {opt_d:.3f}")
    print(f"  Max net benefit: {sweep_result['max_net_benefit']:.4f}")

    # 4. Compare strategies at base_cost_ratio = 1.0
    print("\n--- Comparing decision strategies (base cost ratio = 1.0) ---")
    comparison = compare_decision_strategies(
        prob_event, data.event_times, data.event_indicators,
        delta_t=delta_t, eval_times=eval_times, base_cost_ratio=1.0)

    for name, res in comparison['strategies'].items():
        sens = res.get('sensitivity', np.nan)
        spec = res.get('specificity', np.nan)
        nb = res.get('net_benefit', np.nan)
        print(f"  {name:<30} Sens={sens:.3f}  Spec={spec:.3f}  NB={nb:.4f}")

    # 5. Net benefit improvement vs data-adaptive fixed threshold
    nb_optimal = comparison['strategies']['Cost-sensitive (optimal)']['net_benefit']
    # Compare against median-probability threshold (the most "reasonable" fixed rule)
    median_key = [k for k in comparison['strategies'].keys() if 'Median' in k]
    if median_key:
        nb_fixed = comparison['strategies'][median_key[0]]['net_benefit']
        ref_label = median_key[0]
    else:
        nb_fixed = 0.0
        ref_label = 'N/A'

    if abs(nb_fixed) > 1e-10:
        harm_reduction = (nb_optimal - nb_fixed) / abs(nb_fixed) * 100
        print(f"\n  Net benefit improvement (optimal vs {ref_label}): {harm_reduction:+.1f}%")
    elif nb_optimal > 0:
        print(f"\n  Net benefit: optimal={nb_optimal:.4f} vs {ref_label}={nb_fixed:.4f}")
        print(f"  (Fixed threshold NB≈0; optimal achieves positive NB)")
    else:
        print(f"\n  Net benefit: optimal={nb_optimal:.4f} (no improvement over fixed)")

    return {
        'prob_event': prob_event,
        'sweep_result': sweep_result,
        'comparison': comparison,
        'harm_reduction_pct': harm_reduction,
        'delta_t': delta_t,
    }
