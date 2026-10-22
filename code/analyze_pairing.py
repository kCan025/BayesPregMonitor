import json
import numpy as np
from scipy import stats

RESULTS_DIR = '../results'

SCENARIO_PAIRS = [
    ('baseline', 'heavy_tail'),
    ('nonlinear', 'combined'),
]

# Method pairs: (label, method_a_key, method_b_key)
# Δ is computed as method_a - method_b.
METHOD_PAIRS = [
    ('HF - LM',   'iAUC_highfreq',  'iAUC_landmark'),
    ('HF - LF',   'iAUC_highfreq',  'iAUC_lowfreq'),
    ('LF - LM',   'iAUC_lowfreq',   'iAUC_landmark'),
]

# Diagnostic fields to compare across pairs
DIAG_FIELDS = [
    ('PF r (θ↔η)',     'pf_r'),
    ('median ESS',     'median_ess'),
    ('p5 ESS',         'p5_ess'),
    ('Cox β_η (HF)',   'cox_beta_eta'),
    ('event rate',     'event_rate'),
    ('Static iAUC',    'iAUC_static'),
]


def load_iters(scenario):
    path = f"{RESULTS_DIR}/monte_carlo_{scenario}_iters.json"
    with open(path) as f:
        return json.load(f)


def paired_test(d1, d2, label=""):
    """Run paired t + Wilcoxon + Cohen's d_z on d2 - d1.
    Returns dict with stats, or None if insufficient paired data.
    """
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)
    mask = ~(np.isnan(d1) | np.isnan(d2))
    d1, d2 = d1[mask], d2[mask]
    n = len(d1)
    if n < 3:
        return None

    diff = d2 - d1   # positive = mis > base
    mean_d = np.mean(diff)
    sd_d = np.std(diff, ddof=1)
    se_d = stats.sem(diff)
    ci_lo, ci_hi = stats.t.interval(0.95, n - 1, loc=mean_d, scale=se_d)

    t_stat, t_p = stats.ttest_rel(d2, d1)

    if np.all(diff == 0):
        w_stat, w_p = float('nan'), float('nan')
    else:
        w_stat, w_p = stats.wilcoxon(diff)

    dz = mean_d / (sd_d + 1e-12)

    return {
        'n': n,
        'mean_diff': mean_d,
        'sd_diff': sd_d,
        'se_diff': se_d,
        'ci95': (ci_lo, ci_hi),
        't_stat': t_stat,
        't_p': t_p,
        'w_stat': w_stat,
        'w_p': w_p,
        'd_z': dz,
        'mean_base': np.mean(d1),
        'mean_mis': np.mean(d2),
        'label': label,
    }


def print_result(r, indent="  "):
    """Pretty-print a paired_test result dict."""
    if r is None:
        print(f"{indent}(insufficient paired data)")
        return
    print(f"{indent}n = {r['n']}  paired")
    print(f"{indent}mean(base)  = {r['mean_base']:+.4f}")
    print(f"{indent}mean(mis)   = {r['mean_mis']:+.4f}")
    print(f"{indent}Δ_mis - Δ_base = {r['mean_diff']:+.4f}  "
          f"95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]")
    print(f"{indent}paired t:   t = {r['t_stat']:+.3f},  p = {r['t_p']:.4g}")
    if np.isnan(r['w_p']):
        print(f"{indent}Wilcoxon:   all zeros, p = n/a")
    else:
        print(f"{indent}Wilcoxon:   W = {r['w_stat']:.1f},  p = {r['w_p']:.4g}")
    print(f"{indent}d_z = {r['d_z']:+.3f}")


def main():
    # ── Load all data ────────────────────────────────────────────────
    all_data = {}
    for base, mis in SCENARIO_PAIRS:
        all_data[base] = load_iters(base)
        all_data[mis] = load_iters(mis)

    # ── For each scenario pair ──────────────────────────────────────
    for base, mis in SCENARIO_PAIRS:
        d_base = all_data[base]
        d_mis = all_data[mis]
        n_iter = min(len(d_base), len(d_mis))
        # truncate to common length (safety — should be equal)
        d_base = d_base[:n_iter]
        d_mis = d_mis[:n_iter]

        print("\n" + "=" * 72)
        print(f"  PAIR:  {base}  vs  {mis}   (n = {n_iter} matched iterations)")
        print("=" * 72)

        # ── Layer 2A/B: iAUC method pairs ────────────────────────────
        for label, key_a, key_b in METHOD_PAIRS:
            # compute Δ per iteration in each scenario
            delta_base = np.array([
                (r.get(key_a) - r.get(key_b))
                if (r.get(key_a) is not None and r.get(key_b) is not None)
                else np.nan
                for r in d_base
            ])
            delta_mis = np.array([
                (r.get(key_a) - r.get(key_b))
                if (r.get(key_a) is not None and r.get(key_b) is not None)
                else np.nan
                for r in d_mis
            ])

            r = paired_test(delta_base, delta_mis, label=label)

            print(f"\n  ──  Δ = {label}  ──")
            print_result(r, indent="     ")

        # ── Diagnostics layer: raw field comparisons ──────────────────
        print(f"\n  ──  Diagnostic fields (mis - base)  ──")
        print(f"     {'Field':<18} {'base mean':>12} {'mis mean':>12} "
              f"{'diff':>10} {'t p':>10} {'d_z':>8}")
        print(f"     {'-'*18} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*8}")
        for field_label, field_key in DIAG_FIELDS:
            vals_base = np.array([r.get(field_key, np.nan) for r in d_base],
                                 dtype=float)
            vals_mis = np.array([r.get(field_key, np.nan) for r in d_mis],
                                dtype=float)
            r = paired_test(vals_base, vals_mis, label=field_label)
            if r is None:
                print(f"     {field_label:<18} {'n/a':>12}")
                continue
            p_str = f"{r['t_p']:.4g}" if not np.isnan(r['t_p']) else "n/a"
            dz_str = f"{r['d_z']:+.3f}" if not np.isnan(r['d_z']) else "n/a"
            print(f"     {field_label:<18} {r['mean_base']:>12.4f} "
                  f"{r['mean_mis']:>12.4f} {r['mean_diff']:>+10.4f} "
                  f"{p_str:>10} {dz_str:>8}")

        print()

    # ── Cross-pair summary: replication evidence ─────────────────────
    print("\n" + "=" * 72)
    print("  REPLICATION SUMMARY — do null results replicate?")
    print("=" * 72)
    print()
    print("  If both pairs give a non-significant result with the same")
    print("  direction and similar effect size, the null is supported by")
    print("  two independent replicates under different theta backgrounds.")
    print()

    # Collect HF-LM d_z and p from both pairs
    hflm_results = []
    for base, mis in SCENARIO_PAIRS:
        d_base = all_data[base]
        d_mis = all_data[mis]
        delta_base = np.array([
            (r.get('iAUC_highfreq') - r.get('iAUC_landmark'))
            if (r.get('iAUC_highfreq') is not None
                and r.get('iAUC_landmark') is not None)
            else np.nan
            for r in d_base
        ])
        delta_mis = np.array([
            (r.get('iAUC_highfreq') - r.get('iAUC_landmark'))
            if (r.get('iAUC_highfreq') is not None
                and r.get('iAUC_landmark') is not None)
            else np.nan
            for r in d_mis
        ])
        r = paired_test(delta_base, delta_mis)
        hflm_results.append((base, mis, r))

    print(f"  {'Pair':<30} {'Δ_diff':>8} {'Wilcoxon p':>12} {'d_z':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*12} {'-'*8}")
    directions = []
    for base, mis, r in hflm_results:
        if r is None:
            continue
        print(f"  {base + ' ↔ ' + mis:<30} {r['mean_diff']:>+8.4f} "
              f"{r['w_p']:>12.4g} {r['d_z']:>+8.3f}")
        directions.append(np.sign(r['mean_diff']))

    if len(directions) == 2 and directions[0] == directions[1]:
        print(f"\n  ✓ Effect direction consistent across both pairs")
        print(f"    (both d_z have same sign: {directions[0]:+.0f})")
    elif len(directions) == 2:
        print(f"\n  ✗ Effect direction differs between pairs")

    print()
    print("  Interpretation: two independent null results, same direction,")
    print("  small and similar effect sizes → stronger evidence that the")
    print("  advantage is truly robust (the null is meaningful, not just")
    print("  low power in one test).")


if __name__ == '__main__':
    main()
