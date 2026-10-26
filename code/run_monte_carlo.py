"""
run_monte_carlo.py -- Monte Carlo replication for Step 2 + Robustness
=======================================================================
Runs N_MC independent simulation replicates to assess:
  1. Whether High-Freq PF+Cox consistently outperforms baselines
  2. Stable iAUC estimates with confidence intervals
  3. Paired comparison: method A vs method B across same datasets
  4. [v4] Model misspecification robustness analysis across 4 scenarios

Scenarios (theta_mode / obs_noise):
  baseline    : linear   / gaussian   — original setup
  nonlinear   : nonlinear/ gaussian   — sin perturbation in state transition
  heavy_tail  : linear   / heavy_tail — Student-t(df=3) observation noise
  combined    : nonlinear/ heavy_tail — both misspecifications together

Usage:
    python run_monte_carlo.py --scenario baseline --n_mc 80
    python run_monte_carlo.py --run_all --n_mc 80
"""

import numpy as np
import json
import time
import sys
import os
from datetime import datetime

os.makedirs('../results', exist_ok=True)
# results go to results/<subdir> so a re-run can never overwrite another version's files
RESULTS_SUBDIR = os.environ.get('RESULTS_SUBDIR', 'v4.0')
RESULTS_DIR = f'../results/{RESULTS_SUBDIR}'
os.makedirs(RESULTS_DIR, exist_ok=True)

from simulate_data import simulate
from bootstrap_pf import BootstrapPF
from benchmarks import (time_dependent_auc, static_cox_benchmark,
                        downsample_observations, landmarking_benchmark,
                        BootstrapPF as BPF2)
from time_varying_cox import prepare_counting_process_data, fit_cox_model


# ---------------------------------------------------------------------------
# Scenario definitions: each is a config dict passed to simulate()
# ---------------------------------------------------------------------------
SCENARIOS = {
    "baseline":    {"theta_mode": "linear",    "obs_noise": "gaussian"},
    "nonlinear":   {"theta_mode": "nonlinear", "obs_noise": "gaussian"},
    "heavy_tail":  {"theta_mode": "linear",    "obs_noise": "heavy_tail"},
    "combined":    {"theta_mode": "nonlinear", "obs_noise": "heavy_tail"},
}

# Warning thresholds for particle degeneracy
ESS_WARN_THRESHOLD = 10.0   # median ESS below this = severe degeneracy


def _json_safe(obj):
    """Recursively convert numpy types / NaN -> JSON-safe Python types.

    Handles both numpy scalar types and Python-native float('nan')/inf
    that can hide inside lists/tuples/dicts from Cox model returns.
    """
    import math
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, str)) or obj is None:
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return str(obj)


def run_landmarking_auc_only(data, eta_all, landmark_times=[20, 28, 32],
                              delta_t=4, min_pairs=5):
    """
    Lightweight landmarking AUC computation.
    Uses eta (PF-estimated state) at landmark times -- FAIR comparison
    with High-Freq (both use estimated eta, not true theta).
    min_pairs matches time_dependent_auc default for consistency.
    """
    aucs = []
    for t_L in landmark_times:
        t_end = t_L + delta_t
        case_mask = ((data.event_times > t_L) & (data.event_times <= t_end) &
                     (data.event_indicators == 1))
        control_mask = data.event_times > t_end

        t_idx = min(t_L, eta_all.shape[0]) - 1
        t_idx = max(0, t_idx)
        scores = eta_all[t_idx, :]

        n_cases = case_mask.sum()
        n_controls = control_mask.sum()

        if n_cases >= min_pairs and n_controls >= min_pairs:
            cs = scores[case_mask]
            ct = scores[control_mask]
            conc = sum(np.sum(c > ct) for c in cs)
            tied = sum(np.sum(c == ct) for c in cs)
            auc = (conc + 0.5 * tied) / (n_cases * n_controls)
            aucs.append(auc)
    return aucs


def run_one_iter(seed, scenario="baseline", interval=4):
    """Run one complete Monte Carlo iteration for a given scenario."""
    cfg = SCENARIOS[scenario]
    result = {"scenario": scenario}

    # 1. Generate independent dataset under this scenario
    data = simulate(seed=seed, theta_mode=cfg["theta_mode"],
                    obs_noise=cfg["obs_noise"], verbose=False)
    result["event_rate"] = float(data.event_indicators.mean())

    # 2. Particle filter
    pf = BootstrapPF(
        alpha=data.params.alpha,
        sigma2_omega=data.params.sigma2_omega,
        Sigma_eps=data.params.Sigma_eps,
        n_particles=100
    )
    eta_all, particles_all, weights_all = pf.filter_all(data.Y, seed=seed + 1000)

    # Diagnostics: PF quality + particle health
    from scipy.stats import pearsonr
    r, _ = pearsonr(data.theta.flatten(), eta_all.flatten())
    result['pf_r'] = r
    median_ess = float(np.median(pf.ess_all))
    p5_ess = float(np.percentile(pf.ess_all, 5))
    result["median_ess"] = median_ess
    result["p5_ess"] = p5_ess
    if median_ess < ESS_WARN_THRESHOLD:
        result["ess_warning"] = True
    else:
        result["ess_warning"] = False

    # 3. High-Freq PF + Cox: full pipeline — Cox risk score -> AUC
    landmark_times = [20, 28, 32]
    try:
        from time_varying_cox import fit_with_multiple_imputation
        mi_cox = fit_with_multiple_imputation(
            data.event_times, data.event_indicators,
            particles_all, weights_all, data.X, M=5, seed=seed + 3000)
        beta_eta = mi_cox['beta_eta']
        beta_X = mi_cox['beta_X']
        result['cox_beta_eta'] = beta_eta

        # Cox risk score: z(t) = beta_eta*eta(t) + beta_X'*(X - Xbar)
        X_centered = data.X - data.X.mean(axis=0)
        risk_score = beta_eta * eta_all + (X_centered @ beta_X)[None, :]

        auc_hf = time_dependent_auc(
            data.event_times, data.event_indicators, risk_score,
            eval_times=landmark_times)
        result['iAUC_highfreq'] = auc_hf['iAUC']

        # === Plan C: clean frequency comparison ===
        # Same MI-Cox β as HF. LM-clean freezes η at the last landmark
        # (stale between landmarks). Eval times moved BETWEEN landmarks
        # so that HF uses fresh η(t) while LM-clean uses stale η(t_L).
        eval_times_planC = [22, 24, 26, 30, 34]
        risk_score_lm_clean = np.zeros_like(risk_score)
        T_weeks = data.params.T_weeks
        for t_week in range(1, T_weeks + 1):
            lms_before = [lm for lm in landmark_times if lm <= t_week]
            last_lm = lms_before[-1] if lms_before else landmark_times[0]
            risk_score_lm_clean[t_week - 1, :] = (
                beta_eta * eta_all[last_lm - 1, :]
                + (X_centered @ beta_X)
            )
        auc_hf_planC = time_dependent_auc(
            data.event_times, data.event_indicators, risk_score,
            eval_times=eval_times_planC)
        auc_lm_clean = time_dependent_auc(
            data.event_times, data.event_indicators, risk_score_lm_clean,
            eval_times=eval_times_planC)
        result['iAUC_hf_planC'] = auc_hf_planC['iAUC']
        result['iAUC_lm_clean'] = auc_lm_clean['iAUC']

        # Lead time: for event subjects, earliest week flagged above median
        # relative to nearest prior landmark
        lead_times_list = []
        event_indices = np.where(data.event_indicators == 1)[0]
        for i in event_indices:
            event_time = int(data.event_times[i])
            flagged_week = None
            for t in range(12, min(event_time, data.params.T_weeks) + 1):
                eta_t = eta_all[t - 1, :]
                threshold = np.median(eta_t)
                if eta_all[t - 1, i] > threshold:
                    flagged_week = t
                    break
            if flagged_week is not None:
                lms_before = [lm for lm in landmark_times if lm <= flagged_week]
                if lms_before:
                    nearest_lm = max(lms_before)
                    lead_times_list.append(flagged_week - nearest_lm)
        result['mean_lead_time'] = np.mean(lead_times_list) if lead_times_list else np.nan
        result['n_lead_subjects'] = len(lead_times_list)
    except Exception:
        result['iAUC_highfreq'] = np.nan
        result['cox_beta_eta'] = np.nan
        result['mean_lead_time'] = np.nan
        result['n_lead_subjects'] = 0
        result['iAUC_hf_planC'] = np.nan
        result['iAUC_lm_clean'] = np.nan

    # 4. Static Cox (baseline only) — same eval_times for fair comparison
    try:
        res_sc = static_cox_benchmark(data, eval_times=landmark_times)
        result['iAUC_static'] = res_sc['iAUC']
    except Exception:
        result['iAUC_static'] = np.nan

    # 5. Landmarking -- TRUE independent Cox refit at each landmark (Bugfix #2b)
    #    NOTE: This arm has known in-sample optimism (same subjects fit & scored).
    #    Retained for reference only. Primary frequency comparison is Plan C above.
    #    Risk score held constant between landmarks (discrete monitoring).
    try:
        lm_res = landmarking_benchmark(data, eta_all,
                                       landmark_times=landmark_times)
        result['landmark_aucs'] = lm_res['landmark_aucs']
        result['iAUC_landmark'] = lm_res['iAUC']
        result['iAUC_landmark_mean_auc'] = (
            float(np.mean(lm_res['landmark_aucs']))
            if lm_res['landmark_aucs'] else np.nan)
        result['landmark_coxfit'] = [d['coxfit'] for d in lm_res['landmark_details']]
    except Exception as e:
        import traceback
        print(f"  [LM] landmarking_benchmark FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        result['landmark_aucs'] = []
        result['iAUC_landmark'] = np.nan
        result['iAUC_landmark_mean_auc'] = np.nan
        result['landmark_coxfit'] = []

    # 6. Low-Freq PF + Cox
    try:
        Y_down = downsample_observations(data.Y, interval=interval)
        pf_low = BootstrapPF(
            alpha=data.params.alpha,
            sigma2_omega=data.params.sigma2_omega,
            Sigma_eps=data.params.Sigma_eps,
            n_particles=100
        )
        eta_down, _, _ = pf_low.filter_all(Y_down, seed=seed + 2000)

        start, stop, events, eta_vals, X_vals = prepare_counting_process_data(
            data.event_times, data.event_indicators, eta_down, data.X)
        cox_res = fit_cox_model(start, stop, events, eta_vals, X_vals)
        result['cox_beta_eta_lowfreq'] = cox_res['beta_eta']

        # Interpolate to weekly grid
        T_full = data.params.T_weeks
        ds_indices = list(range(0, T_full, interval))
        T_ds = len(ds_indices)
        eta_interp = np.zeros((T_full, data.params.N))
        for j in range(T_ds):
            t_s = ds_indices[j]
            t_e = ds_indices[j + 1] if j + 1 < T_ds else T_full
            eta_interp[t_s:t_e, :] = eta_down[j, :]

        auc_lf = time_dependent_auc(
            data.event_times, data.event_indicators, eta_interp,
            eval_times=landmark_times)
        result['iAUC_lowfreq'] = auc_lf['iAUC']
    except Exception:
        result['iAUC_lowfreq'] = np.nan
        result['cox_beta_eta_lowfreq'] = np.nan

    return result


def run_scenario(scenario, n_mc, run_timestamp=None):
    """Run n_mc replicates for a single scenario, return results list + total_time."""
    cfg = SCENARIOS[scenario]

    print("\n" + "=" * 75)
    print(f"SCENARIO: {scenario}")
    print(f"  theta_mode = {cfg['theta_mode']}")
    print(f"  obs_noise  = {cfg['obs_noise']}")
    print(f"  N_MC       = {n_mc}")
    print("=" * 75)

    t0 = time.time()
    all_results = []
    ess_warnings = 0

    for mc in range(n_mc):
        t_iter = time.time()
        try:
            res = run_one_iter(seed=42 + mc, scenario=scenario)
            if res.get("ess_warning", False):
                ess_warnings += 1
        except Exception as e:
            print(f"  [Iter {mc+1}] FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            res = {'scenario': scenario,
                   'iAUC_highfreq': np.nan, 'iAUC_static': np.nan,
                   'iAUC_landmark': np.nan, 'iAUC_lowfreq': np.nan,
                   'iAUC_hf_planC': np.nan, 'iAUC_lm_clean': np.nan,
                   'pf_r': np.nan, 'cox_beta_eta': np.nan,
                   'landmark_aucs': [], 'event_rate': np.nan,
                   'median_ess': np.nan, 'p5_ess': np.nan,
                   'ess_warning': False,
                   'mean_lead_time': np.nan, 'n_lead_subjects': 0}
        all_results.append(res)

        elapsed_iter = time.time() - t_iter
        eta_sec = elapsed_iter * (n_mc - mc - 1)

        if (mc + 1) % 10 == 0 or mc == 0:
            hf = res.get('iAUC_highfreq', float('nan'))
            lm = res.get('iAUC_landmark', float('nan'))
            lmc = res.get('iAUC_lm_clean', float('nan'))
            ess = res.get('median_ess', float('nan'))
            print(f"  [{mc+1:3d}/{n_mc}] HF={hf:.4f}  LM={lm:.4f}  "
                  f"LMc={lmc:.4f}  medESS={ess:.1f}  "
                  f"({elapsed_iter:.1f}s/iter, ETA {eta_sec/60:.1f}min)")

    total_time = time.time() - t0
    n_valid = sum(1 for r in all_results
                  if not np.isnan(r.get('iAUC_highfreq', np.nan)))
    print(f"\n  Scenario '{scenario}' complete: {n_valid}/{n_mc} valid, "
          f"{ess_warnings} ESS warnings, {total_time/60:.1f} min total")

    # Save per-iteration results (needed for cross-scenario paired tests)
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    iters_path = f'{RESULTS_DIR}/monte_carlo_{scenario}_iters_{run_timestamp}.json'
    with open(iters_path, 'w') as f:
        json.dump(_json_safe(all_results), f, indent=2, allow_nan=False)
    # Also save to standard name for convenience
    iters_std = f'{RESULTS_DIR}/monte_carlo_{scenario}_iters.json'
    with open(iters_std, 'w') as f:
        json.dump(_json_safe(all_results), f, indent=2, allow_nan=False)
    print(f"  Saved per-iter: {iters_path}")
    print(f"  Saved per-iter (std): {iters_std}")

    return all_results, total_time


def summarize_results(all_results, scenario):
    """Produce per-scenario summary table + paired comparisons + diagnostics."""
    methods = ['iAUC_highfreq', 'iAUC_static', 'iAUC_landmark', 'iAUC_lowfreq',
               'iAUC_hf_planC', 'iAUC_lm_clean']
    labels = ['High-Freq PF+Cox', 'Static Cox', 'Landmarking', 'Low-Freq PF+Cox',
              'HF (planC eval)', 'LM-clean (planC eval)']

    print(f"\n{'='*75}")
    print(f"MONTE CARLO RESULTS — scenario: {scenario}")
    print(f"{'='*75}")
    print(f"{'Method':<25} {'Mean':>8} {'SD':>8} {'95% CI':>18} "
          f"{'Valid':>6} {'Median':>8}")
    print("-" * 75)

    summary = {}
    for method, label in zip(methods, labels):
        values = np.array([r[method] for r in all_results
                          if not np.isnan(r.get(method, np.nan))])
        if len(values) == 0:
            print(f"{label:<25} {'NO DATA':>8}")
            summary[label] = {'n_valid': 0}
            continue
        mean_v = np.mean(values)
        sd_v = np.std(values, ddof=1) if len(values) > 1 else 0
        ci_lo = np.percentile(values, 2.5)
        ci_hi = np.percentile(values, 97.5)
        med_v = np.median(values)
        print(f"{label:<25} {mean_v:>8.4f} {sd_v:>8.4f} "
              f"[{ci_lo:.4f}, {ci_hi:.4f}] {len(values):>6d} {med_v:>8.4f}")
        summary[label] = {
            'n_valid': int(len(values)),
            'mean': float(mean_v), 'sd': float(sd_v),
            'ci_95': [float(ci_lo), float(ci_hi)],
            'median': float(med_v),
        }

    # --- Diagnostics ---
    event_rates = [r['event_rate'] for r in all_results
                   if not np.isnan(r.get('event_rate', np.nan))]
    pf_rs = [r['pf_r'] for r in all_results if not np.isnan(r.get('pf_r', np.nan))]
    med_esses = [r['median_ess'] for r in all_results
                 if not np.isnan(r.get('median_ess', np.nan))]
    p5_esses = [r['p5_ess'] for r in all_results
                if not np.isnan(r.get('p5_ess', np.nan))]
    ess_warn_count = sum(1 for r in all_results if r.get('ess_warning', False))
    cox_betas = [r['cox_beta_eta'] for r in all_results
                 if not np.isnan(r.get('cox_beta_eta', np.nan))]
    lead_times = [r['mean_lead_time'] for r in all_results
                  if not np.isnan(r.get('mean_lead_time', np.nan))]

    print(f"\n--- Diagnostics ---")
    if event_rates:
        print(f"  Event rate:    {np.mean(event_rates):.3f} +/- "
              f"{np.std(event_rates, ddof=1):.3f}")
    if pf_rs:
        print(f"  PF corr(θ,η):  {np.mean(pf_rs):.4f} +/- "
              f"{np.std(pf_rs, ddof=1):.4f}")
    if med_esses:
        print(f"  Median ESS:    {np.mean(med_esses):.1f} +/- "
              f"{np.std(med_esses, ddof=1):.1f}  "
              f"(warn <{ESS_WARN_THRESHOLD:.0f}: {ess_warn_count} iters)")
    if p5_esses:
        print(f"  P5 ESS:        {np.mean(p5_esses):.1f} +/- "
              f"{np.std(p5_esses, ddof=1):.1f}")
    if cox_betas:
        print(f"  Cox beta_eta:  {np.mean(cox_betas):.4f} "
              f"(true gamma=0.8)")
    if lead_times:
        print(f"  Lead time:     {np.mean(lead_times):.2f} +/- "
              f"{np.std(lead_times, ddof=1):.2f} weeks")

    # --- Paired HF vs LM (primary comparison) ---
    hf_vals = np.array([r['iAUC_highfreq'] for r in all_results
                        if (not np.isnan(r.get('iAUC_highfreq', np.nan)) and
                            not np.isnan(r.get('iAUC_landmark', np.nan)))])
    lm_vals = np.array([r['iAUC_landmark'] for r in all_results
                        if (not np.isnan(r.get('iAUC_highfreq', np.nan)) and
                            not np.isnan(r.get('iAUC_landmark', np.nan)))])
    paired = {}
    if len(hf_vals) > 0:
        hf_wins = int(np.sum(hf_vals > lm_vals))
        ties = int(np.sum(hf_vals == lm_vals))
        lm_wins = int(np.sum(hf_vals < lm_vals))
        mean_diff = float(np.mean(hf_vals - lm_vals))
        delta = hf_vals - lm_vals
        if np.all(delta == 0):
            w_stat, w_p = float('nan'), float('nan')
        else:
            from scipy.stats import wilcoxon
            w_stat, w_p = wilcoxon(delta)
        print(f"\n  HF vs LM (n={len(hf_vals)} paired): "
              f"wins {hf_wins}/{lm_wins}/{ties}, mean diff {mean_diff:+.4f}, "
              f"Wilcoxon p={w_p:.4g}")
        paired['HF_vs_LM'] = {
            'n_paired': int(len(hf_vals)),
            'HF_wins': hf_wins, 'LM_wins': lm_wins, 'ties': ties,
            'mean_diff': mean_diff,
            'wilcoxon_stat': None if np.isnan(w_stat) else float(w_stat),
            'wilcoxon_p': None if np.isnan(w_p) else float(w_p),
        }

    # --- Paired HF vs LM-clean (Plan C: clean frequency comparison) ---
    # Same β, same X. Only difference: HF uses fresh η(t), LM-clean uses
    # stale η(last_landmark). Eval at [22,24,26,30,34] (between landmarks).
    hf_pC = np.array([r['iAUC_hf_planC'] for r in all_results
                      if (not np.isnan(r.get('iAUC_hf_planC', np.nan)) and
                          not np.isnan(r.get('iAUC_lm_clean', np.nan)))])
    lm_pC = np.array([r['iAUC_lm_clean'] for r in all_results
                      if (not np.isnan(r.get('iAUC_hf_planC', np.nan)) and
                          not np.isnan(r.get('iAUC_lm_clean', np.nan)))])
    if len(hf_pC) > 0:
        hf_wins_pC = int(np.sum(hf_pC > lm_pC))
        ties_pC = int(np.sum(hf_pC == lm_pC))
        lm_wins_pC = int(np.sum(hf_pC < lm_pC))
        mean_diff_pC = float(np.mean(hf_pC - lm_pC))
        delta_pC = hf_pC - lm_pC
        if np.all(delta_pC == 0):
            w_stat_pC, w_p_pC = float('nan'), float('nan')
        else:
            from scipy.stats import wilcoxon
            w_stat_pC, w_p_pC = wilcoxon(delta_pC)
        print(f"\n  HF vs LM-clean [Plan C] (n={len(hf_pC)} paired): "
              f"wins {hf_wins_pC}/{lm_wins_pC}/{ties_pC}, "
              f"mean diff {mean_diff_pC:+.4f}, Wilcoxon p={w_p_pC:.4g}")
        paired['HF_vs_LM_clean'] = {
            'n_paired': int(len(hf_pC)),
            'HF_wins': hf_wins_pC, 'LM_clean_wins': lm_wins_pC, 'ties': ties_pC,
            'mean_diff': mean_diff_pC,
            'wilcoxon_stat': None if np.isnan(w_stat_pC) else float(w_stat_pC),
            'wilcoxon_p': None if np.isnan(w_p_pC) else float(w_p_pC),
            'eval_times': [22, 24, 26, 30, 34],
        }

    diagnostics = {
        'event_rate_mean': float(np.mean(event_rates)) if event_rates else None,
        'pf_r_mean': float(np.mean(pf_rs)) if pf_rs else None,
        'median_ess_mean': float(np.mean(med_esses)) if med_esses else None,
        'p5_ess_mean': float(np.mean(p5_esses)) if p5_esses else None,
        'ess_warning_count': ess_warn_count,
        'cox_beta_eta_mean': float(np.mean(cox_betas)) if cox_betas else None,
        'lead_time_mean': float(np.mean(lead_times)) if lead_times else None,
        'lead_time_sd': (float(np.std(lead_times, ddof=1))
                         if len(lead_times) > 1 else None),
    }
    return summary, paired, diagnostics


def main():
    # --- CLI parsing ---
    n_mc = 80
    scenario = "baseline"
    run_all = False

    if '--n_mc' in sys.argv:
        idx = sys.argv.index('--n_mc')
        if idx + 1 < len(sys.argv):
            n_mc = int(sys.argv[idx + 1])
    if '--run_all' in sys.argv:
        run_all = True
    elif '--scenario' in sys.argv:
        idx = sys.argv.index('--scenario')
        if idx + 1 < len(sys.argv):
            scenario = sys.argv[idx + 1]
            if scenario not in SCENARIOS:
                print(f"ERROR: unknown scenario '{scenario}'")
                print(f"Valid: {list(SCENARIOS.keys())}")
                sys.exit(1)

    print("=" * 75)
    print(f"Monte Carlo Replication Study — Robustness Analysis")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run ID: {timestamp}")
    print(f"N_MC = {n_mc} independent replicates per scenario")
    if run_all:
        print(f"Scenarios: all {len(SCENARIOS)}")
    else:
        print(f"Scenario:  {scenario}")
    print("=" * 75)

    scenarios_to_run = list(SCENARIOS.keys()) if run_all else [scenario]
    cross_summary = {}

    for sc in scenarios_to_run:
        results, total_time = run_scenario(sc, n_mc, run_timestamp=timestamp)
        summary, paired, diagnostics = summarize_results(results, sc)

        # Per-scenario JSON
        output = {
            'scenario': sc,
            'n_mc': n_mc,
            'scenario_config': SCENARIOS[sc],
            'dgp_params': {
                'dgp_version': 'v4.0',
                'gamma': 0.8,
                'lambda_weibull_linear': 0.0002176,
                'lambda_weibull_nonlinear': 0.0001227,
                'beta': [0.005, 0.005, float(np.log(1.5)), float(np.log(3.0))],
                'covariates': ['age', 'bmi', 'age_ge33', 'prior_pe_history'],
                'prior_pe_prevalence': 0.10,  # working assumption, to be confirmed
                'hr_age_ge33': 1.5, 'hr_prior_pe': 3.0,  # working assumptions, tbc
                'N': 2000, 'T_weeks': 40,
                'target_event_rate': 0.04,
            },
            'summary': summary,
            'paired_comparisons': paired,
            'diagnostics': diagnostics,
            'total_time_seconds': float(total_time),
        }
        out_path = f'{RESULTS_DIR}/monte_carlo_{sc}_{timestamp}.json'
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)
        # Also save to standard name
        out_std = f'{RESULTS_DIR}/monte_carlo_{sc}.json'
        with open(out_std, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n  Saved: {out_path}")
        print(f"  Saved (std): {out_std}")

        cross_summary[sc] = {
            'hf_mean': summary.get('High-Freq PF+Cox', {}).get('mean'),
            'lm_mean': summary.get('Landmarking', {}).get('mean'),
            'hf_planC_mean': summary.get('HF (planC eval)', {}).get('mean'),
            'lm_clean_mean': summary.get('LM-clean (planC eval)', {}).get('mean'),
            'hf_vs_lm_diff': paired.get('HF_vs_LM', {}).get('mean_diff'),
            'hf_vs_lm_clean_diff': paired.get('HF_vs_LM_clean', {}).get('mean_diff'),
            'hf_vs_lm_clean_p': paired.get('HF_vs_LM_clean', {}).get('wilcoxon_p'),
            'pf_r': diagnostics.get('pf_r_mean'),
            'lead_time': diagnostics.get('lead_time_mean'),
            'event_rate': diagnostics.get('event_rate_mean'),
            'median_ess': diagnostics.get('median_ess_mean'),
            'ess_warnings': diagnostics.get('ess_warning_count'),
            'cox_beta_eta': diagnostics.get('cox_beta_eta_mean'),
        }

    # --- Cross-scenario comparison table ---
    if len(scenarios_to_run) > 1:
        print("\n" + "=" * 100)
        print("CROSS-SCENARIO COMPARISON")
        print("=" * 100)
        print(f"{'Scenario':<15} {'HF':>7} {'LM':>7} {'LMc':>7} "
              f"{'ΔHF-LM':>8} {'ΔHF-LMc':>8} {'p(planC)':>9} "
              f"{'PF r':>6} {'Evt%':>5} {'medESS':>7}")
        print("-" * 100)
        for sc in scenarios_to_run:
            s = cross_summary[sc]
            p_str = f"{s['hf_vs_lm_clean_p']:.2e}" if s.get('hf_vs_lm_clean_p') else 'N/A'
            print(f"{sc:<15} {s['hf_mean']:>7.4f} {s['lm_mean']:>7.4f} "
                  f"{s['lm_clean_mean']:>7.4f} "
                  f"{s['hf_vs_lm_diff']:>+8.4f} {s['hf_vs_lm_clean_diff']:>+8.4f} "
                  f"{p_str:>9} "
                  f"{s['pf_r']:>6.4f} "
                  f"{100*s['event_rate']:>4.1f}% "
                  f"{s['median_ess']:>7.1f}")
        print("=" * 100)
        print("\nInterpretation guide:")
        print("  · Plan C (HF vs LM-clean): same β, only η freshness differs.")
        print("    If ΔHF-LMc > 0 and significant → continuous monitoring buys real AUC.")
        print("    If ΔHF-LMc ≈ 0 → frequency doesn't matter (signal dominated).")
        print("  · Old LM (HF vs LM) is contaminated by in-sample optimism — ignore for frequency claims.")
        print("  · If medESS < 10 → particle degeneracy; consider increasing n_particles.")


if __name__ == "__main__":
    main()
