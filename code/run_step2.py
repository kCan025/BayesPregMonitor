"""
run_step2.py -- Step 2: Benchmark Comparison
=============================================
Runs all benchmark methods and compares with the high-frequency pipeline.

Input: PF results and data from Step 1 (../results/pf_results.npz, data_summary.json)
       OR regenerates data + PF if results not found.

Output:
  ../results/benchmark_results.json  -- full comparison table
  ../figures/fig2_auc_comparison.png -- AUC comparison bar chart
  ../figures/fig3_roc_curves.png     -- time-dependent ROC curves

Usage:
    python run_step2.py
"""

import numpy as np
import json
import os
import time
from datetime import datetime

os.makedirs('../results', exist_ok=True)
os.makedirs('../figures', exist_ok=True)

from simulate_data import simulate, SimulatedData
from bootstrap_pf import BootstrapPF
from benchmarks import run_all_benchmarks


def main():
    print("=" * 70)
    print("BayesPregMonitor -- Step 2: Benchmark Comparison")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    t0 = time.time()

    # =========================================================================
    # 1. DATA + PF (regenerate or load from Step 1)
    # =========================================================================
    print("\n" + "=" * 50)
    print("Loading / regenerating data and PF results")
    print("=" * 50)

    # Always regenerate for reproducibility (same seed)
    data = simulate(seed=42)
    params = data.params

    pf = BootstrapPF(
        alpha=params.alpha,
        sigma2_omega=params.sigma2_omega,
        Sigma_eps=params.Sigma_eps,
        n_particles=100
    )

    t_pf = time.time()
    eta_all, particles_all, weights_all = pf.filter_all(data.Y)
    print(f"\nPF completed in {time.time() - t_pf:.1f}s")

    from scipy.stats import pearsonr
    r, _ = pearsonr(data.theta.flatten(), eta_all.flatten())
    print(f"Correlation(theta, eta): r = {r:.4f}")

    # =========================================================================
    # 2. RUN ALL BENCHMARKS
    # =========================================================================
    print("\n" + "=" * 50)
    print("Running benchmark comparisons")
    print("=" * 50)

    results = run_all_benchmarks(data, eta_all, particles_all, weights_all)

    # =========================================================================
    # 3. SAVE RESULTS
    # =========================================================================
    print("\n" + "=" * 50)
    print("Saving results")
    print("=" * 50)

    # Save benchmark results as JSON
    benchmark_json = {}
    for name, res in results.items():
        entry = {
            'method': res['method'],
            'iAUC': float(res['iAUC']) if not np.isnan(res['iAUC']) else None,
        }
        if 'ci_95' in res:
            entry['ci_95'] = [float(res['ci_95'][0]), float(res['ci_95'][1])]
        if 'auc_result' in res and res['auc_result']['eval_times']:
            entry['eval_times'] = res['auc_result']['eval_times']
            entry['auc_values'] = [float(a) for a in res['auc_result']['auc_values']]
        if 'correlation_r' in res:
            entry['correlation_r'] = float(res['correlation_r'])
        if 'cox_beta_eta' in res:
            entry['cox_beta_eta'] = float(res['cox_beta_eta'])
        if 'landmark_details' in res:
            entry['landmark_details'] = res['landmark_details']

        benchmark_json[name] = entry

    # Add DGP info
    benchmark_json['_metadata'] = {
        'N': int(params.N),
        'T_weeks': int(params.T_weeks),
        'event_rate': float(data.event_indicators.mean()),
        'gamma': float(params.gamma),
        'lambda_weibull': float(params.lambda_weibull),
        'beta': params.beta.tolist(),
        'delta_t': 4,
        'n_bootstrap': 200,
    }

    with open('../results/benchmark_results.json', 'w') as f:
        json.dump(benchmark_json, f, indent=2)
    print("  Saved: ../results/benchmark_results.json")

    # =========================================================================
    # 4. GENERATE FIGURES
    # =========================================================================
    print("\n" + "=" * 50)
    print("Generating figures")
    print("=" * 50)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        mpl.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 11,
            'legend.fontsize': 8,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.spines.top': False,
            'axes.spines.right': False,
        })

        # --- Fig 2: iAUC comparison bar chart ---
        fig, ax = plt.subplots(figsize=(8, 5))

        methods = list(results.keys())
        iAUCs = [results[m]['iAUC'] for m in methods]
        ci_lowers = [results[m]['ci_95'][0] if 'ci_95' in results[m] else np.nan for m in methods]
        ci_uppers = [results[m]['ci_95'][1] if 'ci_95' in results[m] else np.nan for m in methods]

        colors = ['#2563EB', '#6B7280', '#D97706', '#059669']
        x = np.arange(len(methods))
        bars = ax.bar(x, iAUCs, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)

        # Error bars for CI
        for i, (bar, cl, cu) in enumerate(zip(bars, ci_lowers, ci_uppers)):
            if not np.isnan(cl) and not np.isnan(cu):
                ax.plot([bar.get_x(), bar.get_x()], [cl, cu],
                       color='black', linewidth=1.5, marker='_', markersize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(' PF+Cox', '\nPF+Cox') for m in methods],
                          fontsize=9)
        ax.set_ylabel('Integrated AUC (iAUC)')
        ax.set_title('Time-Dependent AUC: Method Comparison')
        ax.set_ylim(0.4, 1.0)
        ax.axhline(0.5, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)

        # Add value labels
        for bar, iAUC in zip(bars, iAUCs):
            ax.text(bar.get_x(), bar.get_height() + 0.02,
                   f'{iAUC:.3f}', ha='center', va='bottom', fontsize=9,
                   fontweight='bold')

        fig.tight_layout()
        fig.savefig('../figures/fig2_auc_comparison.png')
        print("  Saved: ../figures/fig2_auc_comparison.png")

        # --- Fig 3: Time-dependent AUC curves ---
        fig, ax = plt.subplots(figsize=(8, 5))

        for name, res in results.items():
            if 'auc_result' in res and res['auc_result']['eval_times']:
                eval_t = res['auc_result']['eval_times']
                auc_v = res['auc_result']['auc_values']
                ax.plot(eval_t, auc_v, 'o-', label=name, markersize=4, linewidth=1.5)

        ax.set_xlabel('Evaluation Time (gestational week)')
        ax.set_ylabel('Incident/Dynamic AUC')
        ax.set_title('Time-Dependent AUC Across Gestation')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_ylim(0.3, 1.0)
        ax.axhline(0.5, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
        ax.axvline(20, color='gray', linewidth=0.3, linestyle='--', alpha=0.3)
        ax.axvline(28, color='gray', linewidth=0.3, linestyle='--', alpha=0.3)
        ax.axvline(32, color='gray', linewidth=0.3, linestyle='--', alpha=0.3)

        fig.tight_layout()
        fig.savefig('../figures/fig3_auc_over_time.png')
        print("  Saved: ../figures/fig3_auc_over_time.png")

        plt.close('all')

    except Exception as e:
        print(f"  Figure generation failed: {e}")
        print("  (Benchmark results still saved to JSON)")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"\nOutputs:")
    print(f"  ../results/benchmark_results.json")
    print(f"  ../figures/fig2_auc_comparison.png")
    print(f"  ../figures/fig3_auc_over_time.png")
    print(f"\nNext: run run_step3.py for cost-sensitive decision rule (Layer 3)")


if __name__ == "__main__":
    main()
