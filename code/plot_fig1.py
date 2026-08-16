"""
plot_fig1.py -- Figure 1: Dynamic Risk Score Trajectories eta_i(t)
==================================================================
Generates a multi-panel figure showing:
  (a) True latent state trajectories theta_i(t) for selected subjects
  (b) Estimated risk scores eta_i(t) from Bootstrap PF
  (c) Observed wearable signals (4-dim) for one representative subject
  (d) PF posterior uncertainty: eta_i(t) +/- 2*std from particle ensemble

Style: publication-ready, clean, suitable for SCD/NISS poster.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Style settings
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

# Colors
COLORS = {
    'truth': '#2563EB',       # Blue
    'estimate': '#DC2626',    # Red
    'uncertainty': '#93C5FD', # Light blue
    'obs': ['#059669', '#D97706', '#7C3AED', '#DB2777'],  # 4 wearable signals
    'event': '#DC2626',
    'censor': '#6B7280',
}


def plot_fig1(data, eta_all, particles_all=None, weights_all=None,
              save_path: str = None):
    """
    Generate Fig 1 with 4 panels.

    Args:
        data: SimulatedData object
        eta_all: (T, N) estimated risk scores
        particles_all: (T, N, n_particles) optional, for uncertainty bands
        weights_all: (T, N, n_particles) optional
        save_path: path to save figure (None = display only)
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    T = data.params.T_weeks
    weeks = np.arange(1, T + 1)

    # Select subjects for visualization: 1 event, 1 censored, 1 high-risk
    event_idx = np.where(data.event_indicators == 1)[0]
    censor_idx = np.where(data.event_indicators == 0)[0]

    np.random.seed(123)

    # 处理无删失样本的情况
    if len(censor_idx) > 0:
        subj_censor = censor_idx[np.argmax(data.theta[:, censor_idx].mean(axis=0))]
        label_censor = 'Censored (high risk)'
    else:
        # 若没有删失样本，选最晚事件的个体作为替代
        subj_censor = event_idx[np.argmax(data.event_times[event_idx])]
        label_censor = 'Latest event (no censoring)'

    # 事件个体仍正常选择
    subj_event = event_idx[np.argmin(np.abs(
        data.event_times[event_idx] - np.median(data.event_times[event_idx])))]
    subj_highrisk = event_idx[np.argmin(data.event_times[event_idx])]

    selected = [subj_event, subj_censor, subj_highrisk]
    labels = ['Typical event', label_censor, 'Early event']
    # --- Panel (a): True latent state trajectories ---
    ax = axes[0, 0]
    for idx, (s, lbl) in enumerate(zip(selected, labels)):
        lw = 1.5 if idx == 0 else 1.2
        ls = '-' if idx < 2 else '--'
        ax.plot(weeks, data.theta[:, s], color=COLORS['truth'],
                linewidth=lw, linestyle=ls, label=lbl, alpha=0.8)
        # Mark event/censor time
        marker = 'v' if data.event_indicators[s] == 1 else '|'
        t_mark = data.event_times[s]
        ax.plot(t_mark, data.theta[t_mark - 1, s], marker=marker,
                color=COLORS['event'] if data.event_indicators[s] == 1
                else COLORS['censor'], markersize=6, zorder=5)

    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.set_xlabel('Gestational Week')
    ax.set_ylabel(r'Latent Risk State $\theta_i(t)$')
    ax.set_title('(a) True Latent Trajectories')
    ax.legend(loc='upper left', framealpha=0.9)

    # --- Panel (b): Estimated eta_i(t) vs true theta_i(t) ---
    ax = axes[0, 1]
    for idx, (s, lbl) in enumerate(zip(selected, labels)):
        # True trajectory
        ax.plot(weeks, data.theta[:, s], color=COLORS['truth'],
                linewidth=1, alpha=0.4, linestyle='--')
        # Estimated
        ax.plot(weeks, eta_all[:, s], color=COLORS['estimate'],
                linewidth=1.5, label=lbl)

    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.set_xlabel('Gestational Week')
    ax.set_ylabel(r'$\theta_i(t)$ / $\hat{\eta}_i(t)$')
    ax.set_title('(b) PF Estimate vs Truth (dashed=true)')
    ax.legend(loc='upper left', framealpha=0.9)

    # --- Panel (c): Wearable observations (one subject) ---
    ax = axes[1, 0]
    subj_vis = subj_highrisk
    var_names = ['HRV', 'Sleep frag.', 'Activity', 'Mood']
    for d in range(4):
        # Normalize for visualization
        y_d = data.Y[:, subj_vis, d]
        y_norm = (y_d - y_d.mean()) / (y_d.std() + 1e-8)
        ax.plot(weeks, y_norm, color=COLORS['obs'][d],
                linewidth=0.8, alpha=0.7, label=var_names[d])
    ax.set_xlabel('Gestational Week')
    ax.set_ylabel('Normalized Signal')
    ax.set_title('(c) Wearable Signals (Early Event Subject)')
    ax.legend(loc='upper right', framealpha=0.9, ncol=2)

    # --- Panel (d): PF uncertainty (if particles available) ---
    ax = axes[1, 1]
    if particles_all is not None and weights_all is not None:
        subj_unc = subj_event
        # Compute weighted std at each time
        eta_mean = eta_all[:, subj_unc]
        eta_var = np.zeros(T)
        for t in range(T):
            p = particles_all[t, subj_unc, :]
            w = weights_all[t, subj_unc, :]
            eta_var[t] = np.sum(w * (p - eta_mean[t]) ** 2)
        eta_std = np.sqrt(eta_var)

        # Plot with uncertainty band
        ax.fill_between(weeks, eta_mean - 2 * eta_std, eta_mean + 2 * eta_std,
                       color=COLORS['uncertainty'], alpha=0.4,
                       label=r'$\hat{\eta}_i(t) \pm 2$ SD')
        ax.plot(weeks, eta_mean, color=COLORS['estimate'],
               linewidth=1.5, label=r'$\hat{\eta}_i(t)$')
        ax.plot(weeks, data.theta[:, subj_unc], color=COLORS['truth'],
               linewidth=1, linestyle='--', alpha=0.6, label=r'True $\theta_i(t)$')
        ax.legend(loc='upper left', framealpha=0.9)
    else:
        # Fallback: show all subjects' eta
        for i in range(min(20, data.params.N)):
            ax.plot(weeks, eta_all[:, i], color=COLORS['estimate'],
                   linewidth=0.5, alpha=0.3)
        ax.set_ylabel(r'$\hat{\eta}_i(t)$')
        ax.set_title('(d) All PF Estimates (first 20 subjects)')

    ax.set_xlabel('Gestational Week')
    ax.set_ylabel(r'$\hat{\eta}_i(t)$')
    if particles_all is not None:
        ax.set_title('(d) PF Posterior Uncertainty')

    fig.suptitle('Figure 1: Dynamic Risk Score Estimation via Bootstrap PF',
                fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    from simulate_data import simulate
    from bootstrap_pf import BootstrapPF

    # Run simulation and PF
    data = simulate()
    params = data.params

    pf = BootstrapPF(
        alpha=params.alpha,
        sigma2_omega=params.sigma2_omega,
        Sigma_eps=params.Sigma_eps,
        n_particles=100
    )

    print("\nRunning Bootstrap PF...")
    eta_all, particles_all, weights_all = pf.filter_all(data.Y)

    # Generate Fig 1
    plot_fig1(data, eta_all, particles_all, weights_all,
             save_path='../figures/fig1_eta_trajectories.png')
    print("Done.")
