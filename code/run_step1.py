"""
run_step1.py -- Step 1: Data Generation + Bootstrap PF + Time-varying Cox
==========================================================================
Main pipeline script for the BayesPregMonitor project.

Runs the complete Layer 1-2 analysis:
  1. Generate N=500 synthetic subjects (DGP v3.3)
  2. Run Bootstrap Particle Filter (100 particles)
  3. Fit time-varying Cox model with marginal MI (M=5)
  4. Generate Fig 1 (eta trajectories)
  5. Save results to ../results/

Usage:
    python run_step1.py
"""

import numpy as np
import time
import os
import json
from datetime import datetime

# Create directories
os.makedirs('../results', exist_ok=True)
os.makedirs('../figures', exist_ok=True)

from simulate_data import simulate, DGPParams
from bootstrap_pf import BootstrapPF, marginal_multiple_imputation
from time_varying_cox import (
    prepare_counting_process_data,
    fit_cox_model,
    fit_with_multiple_imputation,
    fit_naive_cox
)
from plot_fig1 import plot_fig1


def main():
    print("=" * 70)
    print("BayesPregMonitor -- Step 1 Pipeline")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # =========================================================================
    # 1. DATA GENERATION
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 1: Data Generation")
    print("=" * 50)
    data = simulate(seed=42)

    # Save data summary
    data_summary = {
        'N': int(data.params.N),
        'T_weeks': int(data.params.T_weeks),
        'event_rate': float(data.event_indicators.mean()),
        'mean_event_time_events': float(
            data.event_times[data.event_indicators == 1].mean()),
        'mean_event_time_censored': float(
            data.event_times[data.event_indicators == 0].mean()),
        'dgp_params': {
            'sigma2_omega': data.params.sigma2_omega,
            'alpha': data.params.alpha.tolist(),
            'Sigma_eps_diag': np.diag(data.params.Sigma_eps).tolist(),
            'gamma': data.params.gamma,
            'k_weibull': data.params.k_weibull,
            'lambda_weibull': data.params.lambda_weibull,
            'beta': data.params.beta.tolist(),
        }
    }
    with open('../results/data_summary.json', 'w') as f:
        json.dump(data_summary, f, indent=2)

    # =========================================================================
    # 2. BOOTSTRAP PARTICLE FILTER
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 2: Bootstrap Particle Filter")
    print("=" * 50)

    pf = BootstrapPF(
        alpha=data.params.alpha,
        sigma2_omega=data.params.sigma2_omega,
        Sigma_eps=data.params.Sigma_eps,
        n_particles=100
    )

    t0 = time.time()
    eta_all, particles_all, weights_all = pf.filter_all(data.Y)
    pf_time = time.time() - t0

    print(f"\nPF completed in {pf_time:.1f} seconds")
    print(f"Output shapes: eta={eta_all.shape}, "
          f"particles={particles_all.shape}, weights={weights_all.shape}")

    # Quality check: correlation between true and estimated
    from scipy.stats import pearsonr
    r, p = pearsonr(data.theta.flatten(), eta_all.flatten())
    print(f"Correlation(theta, eta): r = {r:.4f}")

    # Save PF results
    np.savez('../results/pf_results.npz',
             eta=eta_all,
             particles=particles_all,
             weights=weights_all,
             pf_time=pf_time,
             correlation_r=r)

    # =========================================================================
    # 3. TIME-VARYING COX MODEL
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 3: Time-varying Cox Model")
    print("=" * 50)

    # 3a. Naive Cox (point estimate, no MI)
    print("\n--- 3a: Naive Cox (no MI) ---")
    naive_result = fit_naive_cox(
        data.event_times, data.event_indicators, eta_all, data.X)

    # 3b. MI Cox (M=5)
    print("\n--- 3b: Multiple Imputation Cox (M=5) ---")
    mi_result = fit_with_multiple_imputation(
        data.event_times, data.event_indicators,
        particles_all, weights_all, data.X, M=5)

    # Compare with truth
    print("\n" + "=" * 50)
    print("COMPARISON WITH GROUND TRUTH")
    print("=" * 50)
    print(f"True gamma (state -> hazard): {data.params.gamma:.4f}")
    print(f"True beta (baseline effects): {data.params.beta}")
    print(f"\nNaive Cox:  beta_eta = {naive_result['beta_eta']:.4f} "
          f"(SE={naive_result['se_eta']:.4f})")
    print(f"MI Cox:     beta_eta = {mi_result['beta_eta']:.4f} "
          f"(SE={mi_result['se_eta']:.4f})")
    print(f"\nMI SE / Naive SE = {mi_result['se_eta']/naive_result['se_eta']:.2f}")
    print(f"(>1 means MI correctly widens CIs)")

    # Save Cox results
    cox_results = {
        'true_gamma': data.params.gamma,
        'true_beta': data.params.beta.tolist(),
        'naive': {
            'beta_eta': naive_result['beta_eta'],
            'se_eta': naive_result['se_eta'],
            'beta_X': naive_result['beta_X'].tolist(),
            'log_likelihood': naive_result['log_likelihood'],
        },
        'mi_5': {
            'beta_eta': mi_result['beta_eta'],
            'se_eta': mi_result['se_eta'],
            'beta_X': mi_result['beta_X'].tolist(),
            'W_eta': mi_result['W_eta'],
            'B_eta': mi_result['B_eta'],
            'T_eta': mi_result['T_eta'],
            'fmi': mi_result['fmi_eta'],
        }
    }
    with open('../results/cox_results.json', 'w') as f:
        json.dump(cox_results, f, indent=2)

    # =========================================================================
    # 4. FIGURE 1
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 4: Generating Figure 1")
    print("=" * 50)

    fig = plot_fig1(data, eta_all, particles_all, weights_all,
                   save_path='../figures/fig1_eta_trajectories.png')

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1 PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total time: {(time.time() - t0):.1f} seconds")
    print(f"\nOutputs saved:")
    print(f"  ../results/data_summary.json")
    print(f"  ../results/pf_results.npz")
    print(f"  ../results/cox_results.json")
    print(f"  ../figures/fig1_eta_trajectories.png")
    print(f"\nNext: run run_step2.py for benchmarks (Static Cox, Landmarking, JMbayes2)")


if __name__ == "__main__":
    main()
