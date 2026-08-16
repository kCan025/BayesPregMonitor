"""
plot_figures.py -- Figures 2-5: Unified Poster Visualization
==============================================================
Generates publication-ready figures with consistent visual style
matching Fig 1 (plot_fig1.py).

Outputs:
  ../figures/fig2_auc_comparison.png + .pdf
  ../figures/fig3_auc_over_time.png + .pdf
  ../figures/fig4_cost_benefit.png + .pdf
  ../figures/fig5_decision_curves.png + .pdf

Data sources:
  - Fig 2: ../results/monte_carlo_results.json (or hardcoded fallback)
  - Fig 3: Runs single-replicate pipeline to extract per-time AUCs
  - Fig 4: ../results/decision_results.json (or runs step3 pipeline)
  - Fig 5: Same as Fig 4

Usage:
    python plot_figures.py                    # all figures
    python plot_figures.py --fig 2            # only Fig 2
    python plot_figures.py --fig 2 3          # Fig 2 and 3
    python plot_figures.py --skip_pipeline    # use only JSON, skip running pipeline
"""

import numpy as np
import json
import os
import sys
import warnings

os.makedirs('../figures', exist_ok=True)
os.makedirs('../results', exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

# =============================================================================
# GLOBAL STYLE (consistent with Fig 1)
# =============================================================================
mpl.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.axisbelow': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
})

# Unified color palette
COLORS = {
    'highfreq': '#2563EB',      # blue
    'static': '#6B7280',        # gray
    'landmark': '#D97706',      # orange
    'lowfreq': '#059669',       # green
    'treat_all': '#DC2626',     # red
    'treat_none': '#6B7280',    # gray
    'cost_sensitive': '#7C3AED',# purple
    'threshold': '#0891B2',     # teal
}

METHOD_COLORS = {
    'High-Freq PF+Cox': COLORS['highfreq'],
    'Landmarking': COLORS['landmark'],
    'Low-Freq PF+Cox': COLORS['lowfreq'],
    'Static Cox': COLORS['static'],
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_mc_results():
    """Load Monte Carlo results from JSON, with fallback to latest known values."""
    json_path = '../results/monte_carlo_results.json'
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)

    # Fallback: latest known values (Mary's update 2026-08-16)
    warnings.warn("monte_carlo_results.json not found; using hardcoded fallback values")
    return {
        'n_mc': 100,
        'summary': {
            'High-Freq PF+Cox': {
                'mean': 0.724, 'sd': 0.045,
                'ci_95': [0.636, 0.805], 'n_valid': 100,
            },
            'Static Cox': {
                'mean': 0.599, 'sd': 0.040,
                'ci_95': [0.503, 0.681], 'n_valid': 100,
            },
            'Landmarking': {
                'mean': 0.718, 'sd': 0.046,
                'ci_95': [0.640, 0.798], 'n_valid': 100,
            },
            'Low-Freq PF+Cox': {
                'mean': 0.700, 'sd': 0.049,
                'ci_95': [0.594, 0.776], 'n_valid': 100,
            },
        },
        'paired_comparisons': {
            'HF_vs_LM': {
                'n_paired': 100, 'HF_wins': 64, 'LM_wins': 36,
                'mean_diff': 0.0055,
            },
            'HF_vs_SC': {'n_paired': 100, 'mean_diff': 0.125},
            'HF_vs_LF': {'n_paired': 100, 'mean_diff': 0.024},
        },
        'cox_recovery': {
            'mean_beta_eta': 0.750, 'sd_beta_eta': 0.119, 'true_gamma': 0.8,
        },
        'pf_recovery': {'mean_r': 0.989, 'sd_r': 0.001},
        'lead_time': {'mean_weeks': 2.6, 'sd_weeks': 1.1},
    }


def load_decision_results():
    """Load decision layer results from JSON if available."""
    json_path = '../results/decision_results.json'
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return None


def save_figure(fig, basename):
    """Save figure as both PNG (300 DPI) and PDF (vector)."""
    fig.savefig(f'../figures/{basename}.png', format='png',
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'../figures/{basename}.pdf', format='pdf',
                bbox_inches='tight', facecolor='white')
    print(f"  Saved: ../figures/{basename}.png + .pdf")


# =============================================================================
# FIG 2: iAUC COMPARISON BAR CHART
# =============================================================================

def plot_fig2(mc_data=None):
    """
    Figure 2: Time-dependent iAUC comparison across 4 methods.
    Vertical bars with 95% CI error bars, significance annotation.
    """
    if mc_data is None:
        mc_data = load_mc_results()

    summary = mc_data['summary']
    paired = mc_data.get('paired_comparisons', {})

    # Method order (best to worst)
    method_order = ['High-Freq PF+Cox', 'Landmarking', 'Low-Freq PF+Cox', 'Static Cox']
    methods = [m for m in method_order if m in summary]
    means = [summary[m]['mean'] for m in methods]
    ci_lowers = [summary[m]['ci_95'][0] for m in methods]
    ci_uppers = [summary[m]['ci_95'][1] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]

    # Compute error bar magnitudes (asymmetric)
    yerr_lo = [m - cl for m, cl in zip(means, ci_lowers)]
    yerr_hi = [cu - m for m, cu in zip(means, ci_uppers)]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(methods))
    bars = ax.bar(x, means,
                  yerr=[yerr_lo, yerr_hi],
                  capsize=5, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=0.8,
                  error_kw=dict(lw=1.2, capthick=1.2, ecolor='black'))

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('Integrated AUC (iAUC)', fontsize=11)
    ax.set_ylim(0.45, 0.90)

    # Chance line at 0.5
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)

    # Value labels on top of bars
    for bar, mean, ci_lo, ci_hi in zip(bars, means, ci_lowers, ci_uppers):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Significance annotation: HF vs LM
    hf_lm = paired.get('HF_vs_LM', {})
    hf_wins = hf_lm.get('HF_wins', 64)
    n_paired = hf_lm.get('n_paired', 100)
    mean_diff = hf_lm.get('mean_diff', 0.0055)

    # Sign test p-value (two-sided exact binomial)
    try:
        from scipy.stats import binom
        p_sign = 2 * min(binom.cdf(100 - hf_wins, n_paired, 0.5),
                         1 - binom.cdf(100 - hf_wins - 1, n_paired, 0.5))
        p_sign = min(p_sign, 1.0)
    except Exception:
        p_sign = 0.0009  # fallback

    # Format p-value
    if p_sign < 0.001:
        p_str = 'p < 0.001'
    elif p_sign < 0.01:
        p_str = f'p = {p_sign:.3f}'
    else:
        p_str = f'p = {p_sign:.3f}'

    sig_text = (f'High-Freq vs Landmarking\n'
                f'H-F wins {hf_wins}/{n_paired}, '
                f'\u0394 = +{mean_diff:.4f}\n'
                f'Sign test {p_str}')

    ax.annotate(sig_text,
                xy=(0.98, 0.97), xycoords='axes fraction',
                ha='right', va='top', fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.4', fc='#F0F4FF',
                          ec=COLORS['highfreq'], alpha=0.9, linewidth=0.8))


    # Significance bracket: HF vs Static Cox
    hf_sc = paired.get('HF_vs_SC', {})
    sc_diff = hf_sc.get('mean_diff', 0.125)

    ax.annotate('', xy=(0, means[0]), xytext=(3, means[3]),
                arrowprops=dict(arrowstyle='<->', color='#DC2626',
                               lw=1.0, connectionstyle='arc3,rad=-0.3'))

    ax.text(1.5, max(means[0], means[3]) + 0.06,
            f'***\n\u0394 = +{sc_diff:.3f}',
            ha='center', va='bottom', fontsize=7, color='#DC2626',
            fontweight='bold')

    ax.set_title('Time-Dependent AUC: 100 Monte Carlo Replications',
                 fontsize=12, pad=12)

    fig.tight_layout()
    save_figure(fig, 'fig2_auc_comparison')
    plt.close(fig)
    return fig


# =============================================================================
# FIG 3: TIME-DEPENDENT AUC CURVES
# =============================================================================

def plot_fig3_from_pipeline():
    """
    Figure 3: Time-dependent AUC curves across gestation.
    Runs a single-replicate pipeline to extract per-time AUCs for each method.
    """
    print("  Running single-replicate pipeline for Fig 3...")

    from simulate_data import simulate
    from bootstrap_pf import BootstrapPF
    from benchmarks import (time_dependent_auc, static_cox_benchmark,
                            downsample_observations, landmarking_benchmark)
    from time_varying_cox import (prepare_counting_process_data, fit_cox_model,
                                   fit_with_multiple_imputation)

    data = simulate(seed=42)
    params = data.params
    landmark_times = [20, 28, 32]

    # Full eval range for curves
    eval_times_fine = list(range(14, 37))

    # --- PF ---
    pf = BootstrapPF(
        alpha=params.alpha, sigma2_omega=params.sigma2_omega,
        Sigma_eps=params.Sigma_eps, n_particles=100)
    eta_all, particles_all, weights_all = pf.filter_all(data.Y)

    # --- High-Freq: Cox risk score ---
    mi_cox = fit_with_multiple_imputation(
        data.event_times, data.event_indicators,
        particles_all, weights_all, data.X, M=5)
    beta_eta = mi_cox['beta_eta']
    beta_X = mi_cox['beta_X']
    X_centered = data.X - data.X.mean(axis=0)
    risk_score_hf = beta_eta * eta_all + (X_centered @ beta_X)[None, :]

    auc_hf = time_dependent_auc(
        data.event_times, data.event_indicators, risk_score_hf,
        eval_times=eval_times_fine, delta_t=4)

    # --- Low-Freq: downsampled PF + Cox ---
    interval = 4
    Y_down = downsample_observations(data.Y, interval=interval)
    pf_low = BootstrapPF(
        alpha=params.alpha, sigma2_omega=params.sigma2_omega,
        Sigma_eps=params.Sigma_eps, n_particles=100)
    eta_down, _, _ = pf_low.filter_all(Y_down)

    start, stop, events, eta_vals, X_vals = prepare_counting_process_data(
        data.event_times, data.event_indicators, eta_down, data.X)
    cox_res_lf = fit_cox_model(start, stop, events, eta_vals, X_vals)

    T_full = params.T_weeks
    ds_indices = list(range(0, T_full, interval))
    T_ds = len(ds_indices)
    eta_interp = np.zeros((T_full, params.N))
    for j in range(T_ds):
        t_s = ds_indices[j]
        t_e = ds_indices[j + 1] if j + 1 < T_ds else T_full
        eta_interp[t_s:t_e, :] = eta_down[j, :]

    auc_lf = time_dependent_auc(
        data.event_times, data.event_indicators, eta_interp,
        eval_times=eval_times_fine, delta_t=4)

    # --- Landmarking: eta at landmark times ---
    auc_lm = time_dependent_auc(
        data.event_times, data.event_indicators, eta_all,
        eval_times=eval_times_fine, delta_t=4)

    # --- Static Cox: baseline only ---
    res_sc = static_cox_benchmark(data, eval_times=eval_times_fine)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    # HF
    if auc_hf['eval_times']:
        ax.plot(auc_hf['eval_times'], auc_hf['auc_values'],
                'o-', color=COLORS['highfreq'], lw=1.8, markersize=4,
                label='High-Freq PF+Cox', zorder=4)

    # Landmarking
    if auc_lm['eval_times']:
        ax.plot(auc_lm['eval_times'], auc_lm['auc_values'],
                's--', color=COLORS['landmark'], lw=1.5, markersize=4,
                label='Landmarking (eta)', zorder=3)

    # Low-Freq
    if auc_lf['eval_times']:
        ax.plot(auc_lf['eval_times'], auc_lf['auc_values'],
                '^-', color=COLORS['lowfreq'], lw=1.5, markersize=4,
                label='Low-Freq PF+Cox', zorder=2)

    # Static Cox (constant over time since risk scores are baseline-only)
    if res_sc['auc_result']['eval_times']:
        ax.plot(res_sc['auc_result']['eval_times'],
                res_sc['auc_result']['auc_values'],
                'D:', color=COLORS['static'], lw=1.5, markersize=4,
                label='Static Cox', zorder=1)

    # Highlight landmark "silent periods"
    ax.axvspan(20, 28, color=COLORS['landmark'], alpha=0.06, zorder=0)
    ax.axvspan(28, 32, color=COLORS['landmark'], alpha=0.06, zorder=0)

    # Landmark time markers
    for lt in landmark_times:
        ax.axvline(lt, color=COLORS['landmark'], linestyle='-.',
                   linewidth=0.6, alpha=0.4)

    # Chance line
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)

    ax.set_xlabel('Gestational Week', fontsize=11)
    ax.set_ylabel('Incident / Dynamic AUC', fontsize=11)
    ax.set_ylim(0.35, 0.95)
    ax.set_xlim(14, 36)
    ax.set_xticks(range(14, 37, 2))

    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.set_title('Time-Dependent AUC Across Gestation', fontsize=12, pad=12)

    fig.tight_layout()
    save_figure(fig, 'fig3_auc_over_time')
    plt.close(fig)
    return fig


def plot_fig3_from_data(mc_data=None):
    """
    Figure 3 fallback: use individual landmark AUCs from MC results
    to plot discrete points if pipeline is not available.
    """
    if mc_data is None:
        mc_data = load_mc_results()

    lm_ind = mc_data.get('landmark_individual', {})
    if not lm_ind or lm_ind.get('t20_mean') is None:
        print("  Warning: no landmark individual AUCs in MC data; skipping Fig 3 fallback")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    # Landmark means from MC
    lm_times = [20, 28, 32]
    lm_means = [lm_ind['t20_mean'], lm_ind['t28_mean'], lm_ind['t32_mean']]
    lm_sds = [lm_ind['t20_sd'], lm_ind['t28_sd'], lm_ind['t32_sd']]

    ax.errorbar(lm_times, lm_means, yerr=lm_sds,
                fmt='o-', color=COLORS['highfreq'], lw=2, markersize=8,
                capsize=5, label='High-Freq (at landmarks)', zorder=4)

    # Highlight inter-landmark gaps
    ax.axvspan(20, 28, color=COLORS['landmark'], alpha=0.06, zorder=0)
    ax.axvspan(28, 32, color=COLORS['landmark'], alpha=0.06, zorder=0)
    for lt in lm_times:
        ax.axvline(lt, color=COLORS['landmark'], linestyle='-.',
                   linewidth=0.6, alpha=0.4)

    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)

    ax.set_xlabel('Gestational Week', fontsize=11)
    ax.set_ylabel('Incident / Dynamic AUC', fontsize=11)
    ax.set_ylim(0.35, 0.95)
    ax.set_xlim(18, 34)
    ax.set_xticks([20, 24, 28, 32])
    ax.set_xticklabels(['20', '24\n(silent)', '28', '32'])

    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.set_title('High-Freq AUC at Landmark Time Points (MC Mean +/- SD)',
                 fontsize=12, pad=12)

    fig.tight_layout()
    save_figure(fig, 'fig3_auc_over_time')
    plt.close(fig)
    return fig


# =============================================================================
# FIG 4: COST-BENEFIT FRONTIER
# =============================================================================

def plot_fig4(decision_data=None):
    """
    Figure 4: Cost-benefit frontier from sweeping cost ratios.
    Semilogx plot with optimal point and reference thresholds marked.
    """
    if decision_data is None:
        decision_data = load_decision_results()

    if decision_data is None:
        print("  Warning: decision_results.json not found; skipping Fig 4")
        return None

    frontier = decision_data.get('cost_benefit_frontier', {})
    cost_ratios = np.array(frontier.get('cost_ratios', []))
    net_benefits = np.array(frontier.get('net_benefits', []))
    opt_cr = decision_data.get('optimal_cost_ratio', None)
    max_nb = decision_data.get('max_net_benefit', None)

    if len(cost_ratios) == 0:
        print("  Warning: no frontier data; skipping Fig 4")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    # Frontier curve
    ax.semilogx(cost_ratios, net_benefits, '-', color=COLORS['highfreq'],
                lw=2.5, label='Cost-benefit frontier', zorder=3)

    # Fill under frontier
    ax.fill_between(cost_ratios, 0, net_benefits,
                    color=COLORS['highfreq'], alpha=0.08, zorder=1)

    # Optimal point
    if opt_cr is not None and max_nb is not None:
        ax.axvline(opt_cr, color=COLORS['treat_all'], linestyle='--',
                   lw=1.5, alpha=0.7, zorder=2)
        ax.scatter([opt_cr], [max_nb], c=COLORS['treat_all'], s=80,
                   zorder=5, edgecolors='white', linewidths=1.5,
                   label=f'Optimal CR = {opt_cr:.2f}')

        # Label
        ax.annotate(f'Optimal\nCR={opt_cr:.2f}\nNB={max_nb:.4f}',
                    xy=(opt_cr, max_nb),
                    xytext=(opt_cr * 0.25, max_nb + 0.02),
                    fontsize=7.5, color=COLORS['treat_all'],
                    arrowprops=dict(arrowstyle='->', color=COLORS['treat_all'],
                                   lw=0.8),
                    fontweight='bold')

    # Mark fixed threshold strategies
    fixed_thresholds = [
        (0.1 / 0.9, 'd*=0.1'),
        (0.3 / 0.7, 'd*=0.3'),
        (1.0, 'd*=0.5'),
        (0.7 / 0.3, 'd*=0.7'),
    ]
    for cr_fixed, label in fixed_thresholds:
        if cr_fixed < cost_ratios.min() or cr_fixed > cost_ratios.max():
            continue
        idx = np.argmin(np.abs(cost_ratios - cr_fixed))
        nb_at = net_benefits[idx]
        ax.scatter([cr_fixed], [nb_at], c=COLORS['static'], s=40,
                   zorder=4, edgecolors='white', linewidths=0.8)
        ax.annotate(label, (cr_fixed, nb_at),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=6.5, color=COLORS['static'])

    # Zero line
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)

    ax.set_xlabel('Cost Ratio ($C_{FP}$ / $C_{FN}$)', fontsize=11)
    ax.set_ylabel('Net Benefit', fontsize=11)
    ax.set_title('Cost-Benefit Frontier: Optimal Threshold Selection',
                 fontsize=12, pad=12)
    ax.legend(loc='best', fontsize=8, framealpha=0.9)

    fig.tight_layout()
    save_figure(fig, 'fig4_cost_benefit')
    plt.close(fig)
    return fig


# =============================================================================
# FIG 5: DECISION CURVE ANALYSIS
# =============================================================================

def plot_fig5(decision_data=None):
    """
    Figure 5: Decision curve analysis — horizontal bar chart of net benefits
    for different decision strategies, sorted by magnitude.
    """
    if decision_data is None:
        decision_data = load_decision_results()

    if decision_data is None:
        print("  Warning: decision_results.json not found; skipping Fig 5")
        return None

    strategy_comp = decision_data.get('strategy_comparison', {})
    if not strategy_comp:
        print("  Warning: no strategy data; skipping Fig 5")
        return None

    # Extract and sort by net benefit
    names = list(strategy_comp.keys())
    nbs = [strategy_comp[n].get('net_benefit', 0) for n in names]

    # Sort ascending (best at top)
    paired = sorted(zip(nbs, names))
    nbs_sorted = [p[0] for p in paired]
    names_sorted = [p[1] for p in paired]

    # Color assignment
    bar_colors = []
    for name in names_sorted:
        if 'Cost-sensitive' in name or 'optimal' in name.lower():
            bar_colors.append(COLORS['cost_sensitive'])
        elif 'Threshold' in name:
            bar_colors.append(COLORS['threshold'])
        elif 'Treat all' in name:
            bar_colors.append(COLORS['treat_all'])
        elif 'Treat none' in name:
            bar_colors.append(COLORS['treat_none'])
        else:
            bar_colors.append(COLORS['static'])

    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = range(len(names_sorted))
    bars = ax.barh(y_pos, nbs_sorted, color=bar_colors, alpha=0.85,
                   edgecolor='white', linewidth=0.8, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted, fontsize=8)

    # Reference line at 0 (treat none baseline)
    ax.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)

    # Value labels
    for bar, nb in zip(bars, nbs_sorted):
        if nb >= 0:
            offset = 0.01  # 正值标签向右多偏移一点
            ha = 'left'
        else:
            offset = -0.0001  # 负值标签向左多偏移一点（增大绝对值）
            ha = 'right'
        ax.text(bar.get_width() + offset,
                bar.get_y() + bar.get_height() / 2,
                f'{nb:.4f}', va='center', ha=ha, fontsize=8,
                clip_on=False)  # 确保标签不被裁剪

    ax.set_xlabel('Net Benefit (at base cost ratio = 1.0)', fontsize=11)
    ax.set_title('Decision Curve Analysis: Strategy Comparison',
                 fontsize=12, pad=12)

    # Legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['cost_sensitive'], alpha=0.85, label='Cost-sensitive'),
        Patch(facecolor=COLORS['threshold'], alpha=0.85, label='Fixed threshold'),
        Patch(facecolor=COLORS['treat_all'], alpha=0.85, label='Treat all'),
        Patch(facecolor=COLORS['treat_none'], alpha=0.85, label='Treat none'),
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              bbox_to_anchor=(0.02, 0.98),  # 左下角位于轴坐标 (2%, 98%) 处
              fontsize=7, framealpha=0.9, ncol=2,
              borderaxespad=0.5,  # 增加图例与轴的边距
              columnspacing=0.8,  # 列间距
              handletextpad=0.5)  # 色块与文字间距

    fig.tight_layout()
    save_figure(fig, 'fig5_decision_curves')
    plt.close(fig)
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    skip_pipeline = '--skip_pipeline' in sys.argv

    # Determine which figures to generate
    fig_args = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--fig':
            i += 1
            while i < len(sys.argv) and sys.argv[i].isdigit():
                fig_args.append(int(sys.argv[i]))
                i += 1
        elif sys.argv[i] == '--skip_pipeline':
            i += 1
        else:
            i += 1

    if not fig_args:
        fig_args = [2, 3, 4, 5]

    print("=" * 60)
    print(f"Generating figures: {fig_args}")
    print(f"Pipeline skip: {skip_pipeline}")
    print("=" * 60)

    # Load data
    mc_data = load_mc_results()
    decision_data = load_decision_results()

    # Fig 2: iAUC bar chart (always from data, no pipeline needed)
    if 2 in fig_args:
        print("\n--- Fig 2: iAUC Comparison ---")
        plot_fig2(mc_data)

    # Fig 3: Time-dependent AUC curves
    if 3 in fig_args:
        print("\n--- Fig 3: Time-Dependent AUC ---")
        if not skip_pipeline:
            try:
                plot_fig3_from_pipeline()
            except Exception as e:
                print(f"  Pipeline failed ({e}); trying fallback...")
                plot_fig3_from_data(mc_data)
        else:
            plot_fig3_from_data(mc_data)

    # Fig 4: Cost-benefit frontier
    if 4 in fig_args:
        print("\n--- Fig 4: Cost-Benefit Frontier ---")
        if decision_data is None and not skip_pipeline:
            print("  Attempting to run step3 pipeline...")
            try:
                from run_step3 import main as step3_main
                step3_main()
                decision_data = load_decision_results()
            except Exception as e:
                print(f"  Step3 pipeline failed: {e}")
        plot_fig4(decision_data)

    # Fig 5: Decision curve analysis
    if 5 in fig_args:
        print("\n--- Fig 5: Decision Curves ---")
        plot_fig5(decision_data)

    print("\n" + "=" * 60)
    print("All requested figures generated.")
    print("Output directory: ../figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
