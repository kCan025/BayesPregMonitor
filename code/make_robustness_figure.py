import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

os.makedirs('../figures', exist_ok=True)

RESULTS_DIR = '../results'

SCENARIO_PAIRS = [
    ('baseline', 'heavy_tail', 'Linear θ\n(gauss → t(3))'),
    ('nonlinear', 'combined',   'Nonlinear θ\n(gauss → t(3))'),
]

METHOD_PAIRS = [
    ('Δ HF − LM',   'iAUC_highfreq',  'iAUC_landmark', '#2c5282'),
    ('Δ HF − LF',   'iAUC_highfreq',  'iAUC_lowfreq',  '#c05621'),
    ('Δ LF − LM',   'iAUC_lowfreq',   'iAUC_landmark', '#2f855a'),
]


def load_iters(scenario):
    with open(f"{RESULTS_DIR}/monte_carlo_{scenario}_iters.json") as f:
        return json.load(f)


def paired_stats(d1, d2):
    """Compute mean diff, 95% CI (t-based), Wilcoxon p, d_z."""
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)
    mask = ~(np.isnan(d1) | np.isnan(d2))
    d1, d2 = d1[mask], d2[mask]
    n = len(d1)
    if n < 3:
        return None
    diff = d2 - d1
    mean_d = np.mean(diff)
    se = stats.sem(diff)
    ci_lo, ci_hi = stats.t.interval(0.95, n - 1, loc=mean_d, scale=se)
    if np.all(diff == 0):
        w_p = float('nan')
    else:
        _, w_p = stats.wilcoxon(diff)
    dz = mean_d / (np.std(diff, ddof=1) + 1e-12)
    return mean_d, ci_lo, ci_hi, w_p, dz, n


def main():
    # Load data
    all_data = {}
    for base, mis, _ in SCENARIO_PAIRS:
        all_data[base] = load_iters(base)
        all_data[mis] = load_iters(mis)

    # ── Compute all results ─────────────────────────────────────────
    results = []  # list of (pair_idx, pair_label, method_label, color, stats)
    for pi, (base, mis, pair_label) in enumerate(SCENARIO_PAIRS):
        d_base = all_data[base]
        d_mis = all_data[mis]
        for ml, ka, kb, color in METHOD_PAIRS:
            delta_base = np.array([
                (r.get(ka) - r.get(kb))
                if (r.get(ka) is not None and r.get(kb) is not None)
                else np.nan
                for r in d_base
            ])
            delta_mis = np.array([
                (r.get(ka) - r.get(kb))
                if (r.get(ka) is not None and r.get(kb) is not None)
                else np.nan
                for r in d_mis
            ])
            s = paired_stats(delta_base, delta_mis)
            results.append((pi, pair_label, ml, color, s))

    # Diagnostics
    diag_fields = [
        ('PF r (θ ↔ η)',   'pf_r'),
        ('Median ESS',     'median_ess'),
        ('Static iAUC',    'iAUC_static'),
    ]
    diag_results = []
    for pi, (base, mis, pair_label) in enumerate(SCENARIO_PAIRS):
        d_base = all_data[base]
        d_mis = all_data[mis]
        for fl, fk in diag_fields:
            vb = np.array([r.get(fk, np.nan) for r in d_base], dtype=float)
            vh = np.array([r.get(fk, np.nan) for r in d_mis], dtype=float)
            s = paired_stats(vb, vh)
            diag_results.append((pi, fl, s))

    # ── Figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 5.2))
    gs = GridSpec(1, 2, width_ratios=[2.2, 1.2], wspace=0.35)

    # --- Left panel: forest plot ---
    ax = fig.add_subplot(gs[0])

    # y positions: for each pair, 2 rows (HF-LM, HF-LF)
    n_pairs = len(SCENARIO_PAIRS)
    n_methods = len(METHOD_PAIRS)
    y_positions = []
    for pi in range(n_pairs):
        for mi in range(n_methods):
            y_positions.append(pi * (n_methods + 1) + mi)
    # Reverse for top-to-bottom
    y_positions = y_positions[::-1]

    y_idx = 0
    for pi, pair_label, ml, color, s in results:
        if s is None:
            y_idx += 1
            continue
        mean_d, ci_lo, ci_hi, w_p, dz, n = s
        y = y_positions[y_idx]

        # Error bar + point
        ax.errorbar(mean_d, y,
                    xerr=[[mean_d - ci_lo], [ci_hi - mean_d]],
                    fmt='o', color=color, capsize=4, capthick=1.5,
                    markersize=7, markeredgecolor='white',
                    markeredgewidth=1.2, linewidth=2)

        # p-value text on right
        if np.isnan(w_p):
            p_text = "p = n/a"
        elif w_p < 0.001:
            p_text = f"p = {w_p:.2e}"
        else:
            p_text = f"p = {w_p:.3f}"
        ax.text(ci_hi + 0.0003, y, p_text,
                va='center', ha='left', fontsize=8.5, color=color)

        y_idx += 1

    # Zero line
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    # Y-axis labels (pair names on the left, method names inline)
    y_ticks = []
    y_ticklabels = []
    for pi, (base, mis, pair_label) in enumerate(SCENARIO_PAIRS):
        # Put pair label in the middle of the two method rows
        y_mid = pi * (n_methods + 1) + (n_methods - 1) / 2
        y_ticks.append(y_mid)
        y_ticklabels.append(pair_label)
    y_ticks = y_ticks[::-1]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels, fontsize=9.5)

    # Add method labels on the right side of each row
    for yi, (pi, pair_label, ml, color, s) in enumerate(results[::-1]):
        if s is None:
            continue
        y = y_positions[::-1][yi]
        ax.text(-0.0005, y, ml, va='center', ha='right',
                fontsize=8.5, color=color, transform=ax.get_yaxis_transform(),
                clip_on=False)

    ax.set_xlabel("Change in Δ  (heavy-tail − gaussian)", fontsize=10)
    ax.set_title("Does heavy-tail noise change the advantage?",
                 fontsize=11, fontweight='bold', pad=10)

    # Legend
    patches = [
        mpatches.Patch(color=mc[3], label=mc[0])
        for mc in METHOD_PAIRS
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=8.5,
              frameon=False)

    ax.set_xlim(-0.004, 0.006)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Right panel: diagnostic table ---
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    ax2.set_title("Diagnostics (paired)", fontsize=11, fontweight='bold', pad=10)

    # Build table: rows = diagnostic fields, cols = pairs
    cell_text = []
    row_labels = [fl for fl, _ in diag_fields]
    col_labels = [pair_label for _, _, pair_label in SCENARIO_PAIRS]

    for fi, (fl, fk) in enumerate(diag_fields):
        row = []
        for pi, (base, mis, _) in enumerate(SCENARIO_PAIRS):
            # Find the diag result
            s = None
            for rpi, rfl, rs in diag_results:
                if rpi == pi and rfl == fl:
                    s = rs
                    break
            if s is None:
                row.append("n/a")
            else:
                mean_d, _, _, w_p, dz, n = s
                if np.isnan(w_p):
                    p_str = "n/a"
                elif w_p < 0.001:
                    p_str = f"p<0.001"
                else:
                    p_str = f"p={w_p:.3f}"
                direction = "↑" if mean_d > 0 else "↓"
                row.append(f"{mean_d:+.4f}\n({p_str})")
        cell_text.append(row)

    table = ax2.table(cellText=cell_text,
                      rowLabels=row_labels,
                      colLabels=col_labels,
                      loc='center',
                      cellLoc='center',
                      colWidths=[0.38, 0.38])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 2.0)

    # Style header
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # header row
            cell.set_text_props(weight='bold', fontsize=9)
            cell.set_facecolor('#e2e8f0')
        if key[1] == -1:  # row labels
            cell.set_text_props(weight='bold', fontsize=9)
            cell.set_facecolor('#f1f5f9')

    # ── Save ────────────────────────────────────────────────────────
    fig.suptitle("Advantage of High-Freq PF+Cox Remains Stable Under Heavy-Tail Observation Noise",
                 fontsize=12.5, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    png_path = '../figures/robustness_forest.png'
    pdf_path = '../figures/robustness_forest.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
