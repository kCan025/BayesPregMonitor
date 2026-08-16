"""
run_step3.py -- Step 3: Cost-Sensitive Decision Rule (Layer 3)
===============================================================
Completes the three-layer pipeline by adding the decision layer.

Input: Data + PF + Cox results from Steps 1-2
Output:
  ../results/decision_results.json     -- decision rule comparison
  ../figures/fig4_cost_benefit.png     -- cost-benefit frontier
  ../figures/fig5_decision_curves.png  -- decision curve analysis

Usage:
    python run_step3.py
"""

import numpy as np
import json
import os
import time
from datetime import datetime

os.makedirs('../results', exist_ok=True)
os.makedirs('../figures', exist_ok=True)

from simulate_data import simulate
from bootstrap_pf import BootstrapPF
from time_varying_cox import fit_naive_cox, fit_with_multiple_imputation
from adaptive_threshold import run_layer3


def main():
    print("=" * 70)
    print("BayesPregMonitor -- Step 3: Cost-Sensitive Decision Rule")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    t0 = time.time()

    # =========================================================================
    # 1. REGENERATE DATA + PF + COX (Steps 1-2 pipeline)
    # =========================================================================
    print("\n" + "=" * 50)
    print("Reproducing Steps 1-2 pipeline")
    print("=" * 50)

    data = simulate(seed=42)
    params = data.params

    pf = BootstrapPF(
        alpha=params.alpha,
        sigma2_omega=params.sigma2_omega,
        Sigma_eps=params.Sigma_eps,
        n_particles=100
    )

    eta_all, particles_all, weights_all = pf.filter_all(data.Y)

    from scipy.stats import pearsonr
    r, _ = pearsonr(data.theta.flatten(), eta_all.flatten())
    print(f"PF correlation: r = {r:.4f}")

    # Fit Cox model (use MI result)
    print("\nFitting Cox model for Layer 3...")
    print("--- MI Cox (M=5) ---")
    mi_result = fit_with_multiple_imputation(
        data.event_times, data.event_indicators,
        particles_all, weights_all, data.X, M=5)

    beta_eta = mi_result['beta_eta']
    beta_X = mi_result['beta_X']
    print(f"\nUsing: beta_eta = {beta_eta:.4f}, beta_X = {beta_X}")

    # =========================================================================
    # 2. LAYER 3: COST-SENSITIVE DECISION RULE (posterior mean probabilities)
    # =========================================================================
    # Probabilities use posterior mean eta; Cox coefficients (beta_eta, beta_X)
    # are already MI-fitted, so uncertainty is correctly propagated.
    layer3_results = run_layer3(data, eta_all, beta_eta, beta_X, delta_t=4)

    # =========================================================================
    # 3. SAVE RESULTS
    # =========================================================================
    print("\n" + "=" * 50)
    print("Saving results")
    print("=" * 50)

    decision_json = {
        'delta_t': 4,
        'beta_eta': float(beta_eta),
        'beta_X': beta_X.tolist(),
        'optimal_cost_ratio': float(layer3_results['sweep_result']['optimal_cost_ratio']),
        'optimal_d_star': float(layer3_results['sweep_result']['optimal_cost_ratio'] /
                                (1 + layer3_results['sweep_result']['optimal_cost_ratio'])),
        'max_net_benefit': float(layer3_results['sweep_result']['max_net_benefit']),
        'harm_reduction_pct': float(layer3_results['harm_reduction_pct']),
        'cost_benefit_frontier': {
            'cost_ratios': [float(cr) for cr in layer3_results['sweep_result']['cost_ratios']],
            'net_benefits': [float(nb) for nb in layer3_results['sweep_result']['net_benefits']],
        },
        'strategy_comparison': {}
    }

    for name, res in layer3_results['comparison']['strategies'].items():
        decision_json['strategy_comparison'][name] = {
            'threshold': float(res.get('threshold', 0)),
            'sensitivity': float(res.get('sensitivity', 0)),
            'specificity': float(res.get('specificity', 0)),
            'ppv': float(res.get('ppv', 0)) if not np.isnan(res.get('ppv', 0)) else None,
            'net_benefit': float(res.get('net_benefit', 0)),
        }

    with open('../results/decision_results.json', 'w') as f:
        json.dump(decision_json, f, indent=2)
    print("  Saved: ../results/decision_results.json")

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

        # --- Fig 4: Cost-benefit frontier ---
        fig, ax = plt.subplots(figsize=(8, 5))

        cost_ratios = layer3_results['sweep_result']['cost_ratios']
        net_benefits = layer3_results['sweep_result']['net_benefits']

        ax.semilogx(cost_ratios, net_benefits, 'b-', linewidth=2)
        ax.axvline(layer3_results['sweep_result']['optimal_cost_ratio'],
                  color='red', linestyle='--', linewidth=1, alpha=0.7,
                  label=f'Optimal CR={layer3_results["sweep_result"]["optimal_cost_ratio"]:.2f}')

        # Mark fixed thresholds
        for cr_fixed, label in [(0.1/0.9, 'd*=0.1'), (0.3/0.7, 'd*=0.3'),
                                 (1.0, 'd*=0.5'), (0.7/0.3, 'd*=0.7')]:
            idx = np.argmin(np.abs(np.array(cost_ratios) - cr_fixed))
            ax.plot(cost_ratios[idx], net_benefits[idx], 'ko', markersize=6)
            ax.annotate(label, (cost_ratios[idx], net_benefits[idx]),
                       textcoords="offset points", xytext=(5, 5), fontsize=7)

        ax.set_xlabel('Cost Ratio (C_FP / C_FN)')
        ax.set_ylabel('Net Benefit')
        ax.set_title('Cost-Benefit Frontier: Optimal Threshold Selection')
        ax.legend(loc='best', fontsize=8)

        fig.tight_layout()
        fig.savefig('../figures/fig4_cost_benefit.png')
        print("  Saved: ../figures/fig4_cost_benefit.png")

        # --- Fig 5: Decision curve analysis ---
        fig, ax = plt.subplots(figsize=(8, 5))

        strategies = layer3_results['comparison']['strategies']
        names = list(strategies.keys())
        nbs = [strategies[n].get('net_benefit', 0) for n in names]

        colors = ['#2563EB', '#059669', '#D97706', '#DC2626',
                  '#7C3AED', '#6B7280']
        bars = ax.barh(range(len(names)), nbs, color=colors[:len(names)],
                       alpha=0.85, edgecolor='white')

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Net Benefit')
        ax.set_title('Decision Curve Analysis: Strategy Comparison')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')

        for i, (bar, nb) in enumerate(zip(bars, nbs)):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{nb:.4f}', ha='left', va='center', fontsize=8)

        fig.tight_layout()
        fig.savefig('../figures/fig5_decision_curves.png')
        print("  Saved: ../figures/fig5_decision_curves.png")

        plt.close('all')

    except Exception as e:
        print(f"  Figure generation failed: {e}")
        print("  (Decision results still saved to JSON)")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("STEP 3 (LAYER 3) COMPLETE -- FULL PIPELINE FINISHED")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"\nAll outputs:")
    print(f"  ../results/data_summary.json")
    print(f"  ../results/pf_results.npz")
    print(f"  ../results/cox_results.json")
    print(f"  ../results/benchmark_results.json")
    print(f"  ../results/decision_results.json")
    print(f"  ../figures/fig1_eta_trajectories.png")
    print(f"  ../figures/fig2_auc_comparison.png")
    print(f"  ../figures/fig3_auc_over_time.png")
    print(f"  ../figures/fig4_cost_benefit.png")
    print(f"  ../figures/fig5_decision_curves.png")


if __name__ == "__main__":
    main()
