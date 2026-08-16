"""
run_monte_carlo.py -- Monte Carlo replication for Step 2
=========================================================
Runs N_MC independent simulation replicates to assess:
  1. Whether High-Freq PF+Cox consistently outperforms baselines
  2. Stable iAUC estimates with confidence intervals
  3. Paired comparison: method A vs method B across same datasets

This is the gold standard for simulation studies -- a single run
can be misleading due to seed-dependent noise (especially for
Landmarking with 4-6 events per window).

Usage:
    python run_monte_carlo.py               # default: 100 replicates
    python run_monte_carlo.py --n_mc 50     # faster: 50 replicates
"""

import numpy as np
import json
import time
import sys
import os
from datetime import datetime

os.makedirs('../results', exist_ok=True)

from simulate_data import simulate
from bootstrap_pf import BootstrapPF
from benchmarks import (time_dependent_auc, static_cox_benchmark,
                        downsample_observations, BootstrapPF as BPF2)
from time_varying_cox import prepare_counting_process_data, fit_cox_model


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


def run_one_iter(seed, interval=4):
    """Run one complete Monte Carlo iteration."""
    result = {}

    # 1. Generate independent dataset
    data = simulate(seed=seed)

    # 2. Particle filter
    pf = BootstrapPF(
        alpha=data.params.alpha,
        sigma2_omega=data.params.sigma2_omega,
        Sigma_eps=data.params.Sigma_eps,
        n_particles=100
    )
    eta_all, particles_all, weights_all = pf.filter_all(data.Y)

    from scipy.stats import pearsonr
    r, _ = pearsonr(data.theta.flatten(), eta_all.flatten())
    result['pf_r'] = r

    # 3. High-Freq PF + Cox: full pipeline — Cox risk score -> AUC
    landmark_times = [20, 28, 32]
    try:
        from time_varying_cox import fit_with_multiple_imputation
        mi_cox = fit_with_multiple_imputation(
            data.event_times, data.event_indicators,
            particles_all, weights_all, data.X, M=5)
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

    # 4. Static Cox (baseline only)
    try:
        res_sc = static_cox_benchmark(data, eval_times=landmark_times)
        result['iAUC_static'] = res_sc['iAUC']
    except Exception:
        result['iAUC_static'] = np.nan

    # 5. Landmarking (using eta at landmark times -- fair comparison)
    try:
        lm_aucs = run_landmarking_auc_only(data, eta_all)
        result['landmark_aucs'] = lm_aucs
        result['iAUC_landmark'] = np.mean(lm_aucs) if lm_aucs else np.nan
    except Exception:
        result['landmark_aucs'] = []
        result['iAUC_landmark'] = np.nan

    # 6. Low-Freq PF + Cox
    try:
        Y_down = downsample_observations(data.Y, interval=interval)
        pf_low = BootstrapPF(
            alpha=data.params.alpha,
            sigma2_omega=data.params.sigma2_omega,
            Sigma_eps=data.params.Sigma_eps,
            n_particles=100
        )
        eta_down, _, _ = pf_low.filter_all(Y_down)

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


def main():
    # Parse command line
    n_mc = 100
    if '--n_mc' in sys.argv:
        idx = sys.argv.index('--n_mc')
        if idx + 1 < len(sys.argv):
            n_mc = int(sys.argv[idx + 1])

    print("=" * 70)
    print(f"Monte Carlo Replication Study")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"N_MC = {n_mc} independent replicates")
    print(f"DGP: gamma={0.8}, lambda_weibull={0.0015}, N=500, T=40w")
    print("=" * 70)

    t0 = time.time()
    all_results = []

    for mc in range(n_mc):
        t_iter = time.time()
        try:
            res = run_one_iter(seed=42 + mc)
        except Exception as e:
            print(f"  [Iter {mc+1}] FATAL ERROR: {e}")
            res = {'iAUC_highfreq': np.nan, 'iAUC_static': np.nan,
                   'iAUC_landmark': np.nan, 'iAUC_lowfreq': np.nan,
                   'pf_r': np.nan, 'cox_beta_eta': np.nan,
                   'landmark_aucs': []}
        all_results.append(res)

        elapsed_iter = time.time() - t_iter
        eta_sec = elapsed_iter * (n_mc - mc - 1)

        if (mc + 1) % 10 == 0 or mc == 0:
            hf = res.get('iAUC_highfreq', float('nan'))
            lm = res.get('iAUC_landmark', float('nan'))
            print(f"  [{mc+1:3d}/{n_mc}] HighFreq={hf:.4f}  "
                  f"Landmark={lm:.4f}  "
                  f"({elapsed_iter:.1f}s/iter, ETA {eta_sec/60:.1f}min)")

    total_time = time.time() - t0
    print(f"\nAll iterations completed in {total_time/60:.1f} minutes")

    # =========================================================================
    # AGGREGATE RESULTS
    # =========================================================================
    methods = ['iAUC_highfreq', 'iAUC_static', 'iAUC_landmark', 'iAUC_lowfreq']
    labels = ['High-Freq PF+Cox', 'Static Cox', 'Landmarking', 'Low-Freq PF+Cox']

    print("\n" + "=" * 75)
    print("MONTE CARLO RESULTS")
    print("=" * 75)
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
            'mean': float(mean_v),
            'sd': float(sd_v),
            'ci_95': [float(ci_lo), float(ci_hi)],
            'median': float(med_v),
        }

    # =========================================================================
    # PAIRED COMPARISONS
    # =========================================================================
    print("\n" + "-" * 75)
    print("PAIRED COMPARISONS (across same datasets)")
    print("-" * 75)

    hf_vals = np.array([r['iAUC_highfreq'] for r in all_results
                        if (not np.isnan(r.get('iAUC_highfreq', np.nan)) and
                            not np.isnan(r.get('iAUC_landmark', np.nan)))])
    lm_vals = np.array([r['iAUC_landmark'] for r in all_results
                        if (not np.isnan(r.get('iAUC_highfreq', np.nan)) and
                            not np.isnan(r.get('iAUC_landmark', np.nan)))])

    if len(hf_vals) > 0:
        hf_wins = np.sum(hf_vals > lm_vals)
        ties = np.sum(hf_vals == lm_vals)
        lm_wins = np.sum(hf_vals < lm_vals)
        mean_diff = np.mean(hf_vals - lm_vals)
        se_diff = np.std(hf_vals - lm_vals, ddof=1) / np.sqrt(len(hf_vals))

        print(f"\n  High-Freq vs Landmarking (n={len(hf_vals)} paired):")
        print(f"    High-Freq wins: {hf_wins} ({hf_wins/len(hf_vals)*100:.1f}%)")
        print(f"    Landmarking wins: {lm_wins} ({lm_wins/len(hf_vals)*100:.1f}%)")
        print(f"    Ties: {ties}")
        print(f"    Mean diff (HF - LM): {mean_diff:+.4f} (SE = {se_diff:.4f})")

        # Simple sign test
        from scipy.stats import wilcoxon
        try:
            stat, pval = wilcoxon(hf_vals, lm_vals)
            print(f"    Wilcoxon signed-rank p-value: {pval:.4f}")
        except Exception:
            pass

    # HF vs Static
    hf_vals2 = np.array([r['iAUC_highfreq'] for r in all_results
                         if (not np.isnan(r.get('iAUC_highfreq', np.nan)) and
                             not np.isnan(r.get('iAUC_static', np.nan)))])
    st_vals2 = np.array([r['iAUC_static'] for r in all_results
                         if (not np.isnan(r.get('iAUC_highfreq', np.nan)) and
                             not np.isnan(r.get('iAUC_static', np.nan)))])
    if len(hf_vals2) > 0:
        mean_diff2 = np.mean(hf_vals2 - st_vals2)
        print(f"\n  High-Freq vs Static Cox (n={len(hf_vals2)}):")
        print(f"    Mean diff (HF - SC): {mean_diff2:+.4f}")
        print(f"    High-Freq wins: {np.sum(hf_vals2 > st_vals2)}/{len(hf_vals2)}")

    # HF vs Low-Freq
    hf_vals3 = np.array([r['iAUC_highfreq'] for r in all_results
                         if (not np.isnan(r.get('iAUC_highfreq', np.nan)) and
                             not np.isnan(r.get('iAUC_lowfreq', np.nan)))])
    lf_vals3 = np.array([r['iAUC_lowfreq'] for r in all_results
                         if (not np.isnan(r.get('iAUC_highfreq', np.nan)) and
                             not np.isnan(r.get('iAUC_lowfreq', np.nan)))])
    if len(hf_vals3) > 0:
        mean_diff3 = np.mean(hf_vals3 - lf_vals3)
        print(f"\n  High-Freq vs Low-Freq (n={len(hf_vals3)}):")
        print(f"    Mean diff (HF - LF): {mean_diff3:+.4f}")
        print(f"    High-Freq wins: {np.sum(hf_vals3 > lf_vals3)}/{len(hf_vals3)}")

    # =========================================================================
    # COX PARAMETER RECOVERY
    # =========================================================================
    cox_betas = [r['cox_beta_eta'] for r in all_results
                 if not np.isnan(r.get('cox_beta_eta', np.nan))]
    pf_rs = [r['pf_r'] for r in all_results if not np.isnan(r.get('pf_r', np.nan))]

    print(f"\n--- Parameter Recovery ---")
    if cox_betas:
        print(f"  Cox beta_eta: mean={np.mean(cox_betas):.4f}, "
              f"SD={np.std(cox_betas):.4f} (true gamma=0.8)")
    if pf_rs:
        print(f"  PF correlation: mean={np.mean(pf_rs):.4f}, "
              f"SD={np.std(pf_rs):.4f}")

    # =========================================================================
    # DETECTION LEAD TIME (continuous monitoring advantage)
    # =========================================================================
    lead_times_mc = [r['mean_lead_time'] for r in all_results
                     if not np.isnan(r.get('mean_lead_time', np.nan))]
    lead_n = [r['n_lead_subjects'] for r in all_results
              if r.get('n_lead_subjects', 0) > 0]
    if lead_times_mc:
        print(f"\n--- Detection Lead Time (weeks ahead of nearest landmark) ---")
        print(f"  Mean lead time: {np.mean(lead_times_mc):.1f} +/- "
              f"{np.std(lead_times_mc):.1f} weeks (n={len(lead_times_mc)} MC reps)")
        print(f"  Avg flagged subjects per replicate: "
              f"{np.mean(lead_n):.1f}")

    # =========================================================================
    # LANDMARK INDIVIDUAL AUCs
    # =========================================================================
    all_lm20 = [r['landmark_aucs'][0] for r in all_results
                if len(r.get('landmark_aucs', [])) >= 3]
    all_lm28 = [r['landmark_aucs'][1] for r in all_results
                if len(r.get('landmark_aucs', [])) >= 3]
    all_lm32 = [r['landmark_aucs'][2] for r in all_results
                if len(r.get('landmark_aucs', [])) >= 3]

    if all_lm20:
        print(f"\n--- Individual Landmark AUCs (mean +/- SD) ---")
        print(f"  t=20w: {np.mean(all_lm20):.4f} +/- {np.std(all_lm20):.4f} (n={len(all_lm20)})")
        print(f"  t=28w: {np.mean(all_lm28):.4f} +/- {np.std(all_lm28):.4f} (n={len(all_lm28)})")
        print(f"  t=32w: {np.mean(all_lm32):.4f} +/- {np.std(all_lm32):.4f} (n={len(all_lm32)})")

    # =========================================================================
    # SAVE
    # =========================================================================
    output = {
        'n_mc': n_mc,
        'dgf_params': {
            'gamma': 0.8,
            'lambda_weibull': 0.0015,
            'beta': [0.005, 0.005],
            'N': 500,
            'T_weeks': 40,
        },
        'summary': summary,
        'paired_comparisons': {
            'HF_vs_LM': {
                'n_paired': int(len(hf_vals)),
                'HF_wins': int(hf_wins) if len(hf_vals) > 0 else 0,
                'LM_wins': int(lm_wins) if len(hf_vals) > 0 else 0,
                'mean_diff': float(mean_diff) if len(hf_vals) > 0 else None,
            },
            'HF_vs_SC': {
                'n_paired': int(len(hf_vals2)),
                'mean_diff': float(mean_diff2) if len(hf_vals2) > 0 else None,
            },
            'HF_vs_LF': {
                'n_paired': int(len(hf_vals3)),
                'mean_diff': float(mean_diff3) if len(hf_vals3) > 0 else None,
            }
        },
        'cox_recovery': {
            'mean_beta_eta': float(np.mean(cox_betas)) if cox_betas else None,
            'sd_beta_eta': float(np.std(cox_betas)) if cox_betas else None,
            'true_gamma': 0.8
        },
        'pf_recovery': {
            'mean_r': float(np.mean(pf_rs)) if pf_rs else None,
            'sd_r': float(np.std(pf_rs)) if pf_rs else None,
        },
        'lead_time': {
            'mean_weeks': float(np.mean(lead_times_mc)) if lead_times_mc else None,
            'sd_weeks': float(np.std(lead_times_mc)) if lead_times_mc else None,
            'n_valid_reps': int(len(lead_times_mc)),
            'avg_flagged_subjects': float(np.mean(lead_n)) if lead_n else None,
        },
        'landmark_individual': {
            't20_mean': float(np.mean(all_lm20)) if all_lm20 else None,
            't20_sd': float(np.std(all_lm20)) if all_lm20 else None,
            't28_mean': float(np.mean(all_lm28)) if all_lm28 else None,
            't28_sd': float(np.std(all_lm28)) if all_lm28 else None,
            't32_mean': float(np.mean(all_lm32)) if all_lm32 else None,
            't32_sd': float(np.std(all_lm32)) if all_lm32 else None,
        },
        'total_time_seconds': float(total_time),
    }

    with open('../results/monte_carlo_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: ../results/monte_carlo_results.json")

    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print("\nNext: analyze results, then refine NISS abstract with stable numbers")


if __name__ == "__main__":
    main()
