import numpy as np
import json
import sys
import time

from simulate_data import simulate
from bootstrap_pf import BootstrapPF

SCENARIOS = {
    "baseline":   {"theta_mode": "linear",    "obs_noise": "gaussian"},
    "nonlinear":  {"theta_mode": "nonlinear", "obs_noise": "gaussian"},
    "heavy_tail": {"theta_mode": "linear",    "obs_noise": "heavy_tail"},
    "combined":   {"theta_mode": "nonlinear", "obs_noise": "heavy_tail"},
}


def landmark_auc_in_out(data, eta_all, landmark_times=(20, 28, 32),
                        delta_t=4, rng=None):
    """Return (in_sample_aucs, cross_fitted_aucs, n_events) per landmark."""
    from lifelines import CoxPHFitter
    from sklearn.metrics import roc_auc_score
    import pandas as pd

    T, N = eta_all.shape
    in_aucs, out_aucs, n_events = [], [], []

    for t_L in landmark_times:
        t_idx = min(t_L - 1, T - 1)
        t_end = t_L + delta_t
        at_risk = data.event_times > t_L
        idx = np.where(at_risk)[0]
        if len(idx) < 20:
            in_aucs.append(np.nan); out_aucs.append(np.nan); n_events.append(0)
            continue

        et = data.event_times[idx]
        ev = ((et > t_L) & (et <= t_end) &
              (data.event_indicators[idx] == 1)).astype(int)
        dur = np.maximum(np.minimum(et, t_end) - t_L, 0.01)
        df = pd.DataFrame({
            'eta': eta_all[t_idx, idx], 'age': data.X[idx, 0],
            'bmi': data.X[idx, 1], 'duration': dur, 'event': ev})
        n_events.append(int(ev.sum()))

        if ev.sum() < 5:
            in_aucs.append(np.nan); out_aucs.append(np.nan)
            continue

        # --- in-sample (current behaviour) ---
        try:
            cph = CoxPHFitter()
            cph.fit(df, duration_col='duration', event_col='event',
                    formula='eta + age + bmi')
            p_in = cph.predict_partial_hazard(df).values
            keep = (ev == 1) | (et > t_end)
            a_in = roc_auc_score(ev[keep], p_in[keep])
        except Exception:
            a_in = np.nan
        in_aucs.append(a_in)

        # --- 2-fold cross-fitted ---
        perm = rng.permutation(len(idx))
        h = len(idx) // 2
        halves = [perm[:h], perm[h:]]
        p_cv = np.full(len(idx), np.nan)
        for k in range(2):
            tr, te = halves[1 - k], halves[k]
            if ev[tr].sum() < 3 or ev[tr].sum() == len(tr):
                continue
            try:
                c2 = CoxPHFitter()
                c2.fit(df.iloc[tr], duration_col='duration',
                       event_col='event', formula='eta + age + bmi')
                p_cv[te] = c2.predict_partial_hazard(df.iloc[te]).values
            except Exception:
                continue
        ok = (~np.isnan(p_cv)) & ((ev == 1) | (et > t_end))
        try:
            a_out = roc_auc_score(ev[ok], p_cv[ok])
        except Exception:
            a_out = np.nan
        out_aucs.append(a_out)

    return in_aucs, out_aucs, n_events


def main():
    n_mc = 80
    scenario = 'baseline'
    if '--n_mc' in sys.argv:
        n_mc = int(sys.argv[sys.argv.index('--n_mc') + 1])
    if '--scenario' in sys.argv:
        scenario = sys.argv[sys.argv.index('--scenario') + 1]

    cfg = SCENARIOS[scenario]
    lm_times = [20, 28, 32]
    rows = []
    t0 = time.time()
    for mc in range(n_mc):
        seed = 42 + mc
        data = simulate(seed=seed, theta_mode=cfg['theta_mode'],
                        obs_noise=cfg['obs_noise'], verbose=False)
        pf = BootstrapPF(alpha=data.params.alpha,
                         sigma2_omega=data.params.sigma2_omega,
                         Sigma_eps=data.params.Sigma_eps, n_particles=100)
        eta_all, _, _ = pf.filter_all(data.Y, seed=seed + 1000)
        a_in, a_out, nev = landmark_auc_in_out(
            data, eta_all, lm_times, rng=np.random.RandomState(seed + 7000))
        rows.append({'seed': seed, 'in': a_in, 'out': a_out, 'n_events': nev})
        if (mc + 1) % 20 == 0:
            print(f'  [{mc+1}/{n_mc}] {(time.time()-t0)/(mc+1):.2f}s/iter')

    def col(key, j):
        v = [r[key][j] for r in rows
             if r[key][j] is not None and not np.isnan(r[key][j])]
        return float(np.mean(v)) if v else float('nan')

    print(f'\n=== LM in-sample vs 2-fold cross-fitted AUC -- {scenario} (n={n_mc}) ===')
    print(f"{'landmark':<10}{'n_events':>10}{'in-sample':>12}{'cross-fit':>12}{'optimism':>12}")
    for j, t_L in enumerate(lm_times):
        nev = float(np.mean([r['n_events'][j] for r in rows]))
        ai, ao = col('in', j), col('out', j)
        print(f'{t_L:<10}{nev:>10.1f}{ai:>12.4f}{ao:>12.4f}{ai-ao:>+12.4f}')
    ai_all = float(np.nanmean([[r['in'][j] for j in range(3)] for r in rows]))
    ao_all = float(np.nanmean([[r['out'][j] for j in range(3)] for r in rows]))
    print(f'{"mean":<10}{"":>10}{ai_all:>12.4f}{ao_all:>12.4f}{ai_all-ao_all:>+12.4f}')

    out_path = f'../results/lm_optimism_{scenario}.json'
    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'saved: {out_path}  ({(time.time()-t0)/60:.1f} min)')


if __name__ == '__main__':
    main()
