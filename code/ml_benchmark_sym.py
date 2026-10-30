#!/usr/bin/env python3
"""
ml_benchmark_sym.py -- symmetric PF+Cox vs GBM comparison (new file; no committed code is changed).

Per replicate (scenario baseline, seed = 42 + i), on ONE dataset:
  PF_in : v4.0.1 pipeline exactly (PF seed+1000, MI-Cox seed+3000, fitted on all N, scored on all N)
          -> reproduces the published v4.0.1 per-replicate iAUC.
  PF_cv : same particles, MI-Cox refitted on the training folds only, scored on the held-out fold.
  GBM   : pooled person-week GBM (4-week feature window + 4 baseline covariates), same held-out folds.
All three are evaluated at weeks [22,24,26,30,34], delta=4, min_pairs=5, HF and LM-clean (landmarks 20/28/32).

Usage (from code/):
  python ml_benchmark_sym.py run 0 80 4     # replicates 0..79, 4 worker processes
  python ml_benchmark_sym.py summary        # reads ../results/ml/*.json
Output: ../results/ml/rep_<seed>.json (one file per replicate; reruns skip finished seeds).
"""
import os, sys, io, json, glob, contextlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from simulate_data import simulate
from benchmarks import time_dependent_auc
from bootstrap_pf import BootstrapPF
from time_varying_cox import fit_with_multiple_imputation

EVAL = [22, 24, 26, 30, 34]
LANDMARKS = [20, 28, 32]
K = 4                       # feature window (weeks)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "ml")


def lm_clean(score):
    out = np.zeros_like(score)
    for t in range(1, 41):
        lb = [lm for lm in LANDMARKS if lm <= t]
        out[t - 1] = score[(lb[-1] if lb else LANDMARKS[0]) - 1]
    return out


def build_rows(d, max_t=36):
    N = d.Y.shape[1]
    F, y, tv, sv = [], [], [], []
    for t in range(K, max_t + 1):
        lab = ((d.event_times > t) & (d.event_times <= t + 4) & (d.event_indicators == 1)).astype(int)
        idx = np.where(d.event_times > t)[0]                      # at-risk only
        W = d.Y[t - K:t].transpose(1, 0, 2).reshape(N, -1)        # weeks t-3..t, never theta
        F.append(np.hstack([W[idx], d.X[idx]])); y.append(lab[idx])
        tv.append(np.full(len(idx), t)); sv.append(idx)
    return np.vstack(F), np.concatenate(y), np.concatenate(tv), np.concatenate(sv)


def iauc(d, score):
    r = time_dependent_auc(d.event_times, d.event_indicators, score,
                           eval_times=EVAL, delta_t=4, min_pairs=5)
    return float(r["iAUC"]), r.get("per_window", None), r.get("eval_times", None)


def run_rep(i):
    seed = 42 + i
    path = os.path.join(OUT, f"rep_{seed}.json")
    if os.path.exists(path):
        return json.load(open(path))
    d = simulate(seed=seed, theta_mode="linear", obs_noise="gaussian", verbose=False)
    N = d.params.N
    assert d.Y.shape == (40, N, 4) and d.X.shape == (N, 4) and d.event_times.shape == (N,)

    with contextlib.redirect_stdout(io.StringIO()):
        pf = BootstrapPF(alpha=d.params.alpha, sigma2_omega=d.params.sigma2_omega,
                         Sigma_eps=d.params.Sigma_eps, n_particles=100)
        eta, P, Wt = pf.filter_all(d.Y, seed=seed + 1000)
        mi = fit_with_multiple_imputation(d.event_times, d.event_indicators, P, Wt, d.X,
                                          M=5, seed=seed + 3000)
    s_in = mi["beta_eta"] * eta + ((d.X - d.X.mean(0)) @ mi["beta_X"])[None, :]

    Xm, y, tv, sv = build_rows(d)
    assert not np.any(d.event_times[sv] <= tv), "at-risk violation"
    us = np.unique(sv)
    kf = KFold(5, shuffle=True, random_state=seed)
    folds = list(kf.split(us))                                    # ONE split used by both arms
    cover = np.concatenate([us[te] for _, te in folds])
    assert len(cover) == len(np.unique(cover)) == len(us), "folds must partition subjects"

    g = np.zeros((40, N)); s_cv = np.zeros((40, N)); written = np.zeros((40, N), dtype=int)
    for tr, te in folds:
        tes = us[te]; trm = np.isin(sv, us[tr]); tem = np.isin(sv, tes)
        clf = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.05, max_leaf_nodes=3, min_samples_leaf=20,
            l2_regularization=1.0, early_stopping=False, class_weight="balanced", random_state=seed)
        clf.fit(Xm[trm], y[trm])
        g[tv[tem] - 1, sv[tem]] = clf.predict_proba(Xm[tem])[:, 1]
        written[tv[tem] - 1, sv[tem]] += 1
        trall = np.setdiff1d(np.arange(N), tes)                   # PF arm: refit without test fold
        with contextlib.redirect_stdout(io.StringIO()):
            m2 = fit_with_multiple_imputation(d.event_times[trall], d.event_indicators[trall],
                                              P[:, trall], Wt[:, trall], d.X[trall], M=5, seed=seed + 3000)
        full = m2["beta_eta"] * eta + ((d.X - d.X[trall].mean(0)) @ m2["beta_X"])[None, :]
        s_cv[:, tes] = full[:, tes]
    assert written.max() <= 1, "a (week, subject) cell was predicted twice"
    for t in EVAL:                                                # every evaluated cell has a GBM score
        cases = (d.event_times > t) & (d.event_times <= t + 4) & (d.event_indicators == 1)
        ev = cases | (d.event_times > t + 4)
        assert np.all(written[t - 1, ev] == 1), f"unpredicted evaluation cell at week {t}"

    res = {"seed": seed, "rows": int(len(y)), "pos_rows": int(y.sum()), "n_events": int(d.event_indicators.sum())}
    for name, sc in (("pf_in", s_in), ("pf_cv", s_cv), ("gbm", g)):
        res[name + "_hf"], w_hf, ev_hf = iauc(d, sc)
        res[name + "_lm"], w_lm, ev_lm = iauc(d, lm_clean(sc))
        res[name + "_eval_times"] = ev_hf
    assert res["pf_in_eval_times"] == res["pf_cv_eval_times"] == res["gbm_eval_times"], "arms evaluated on different windows"
    if seed == 42:                                                # reference values (seed 42)
        assert (res["rows"], res["pos_rows"]) == (64795, 296)
        assert abs(res["pf_in_hf"] - 0.7952) < 5e-4, res["pf_in_hf"]
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(path, "w"))
    return res


def summary():
    from scipy.stats import wilcoxon, t as tdist
    R = sorted((json.load(open(f)) for f in glob.glob(os.path.join(OUT, "rep_*.json"))), key=lambda r: r["seed"])
    n = len(R); print(f"replicates: {n} (seeds {R[0]['seed']}..{R[-1]['seed']})")
    g = lambda k: np.array([r[k] for r in R])
    D = {}
    for nm in ("pf_in", "pf_cv", "gbm"):
        hf, lm = g(nm + "_hf"), g(nm + "_lm"); D[nm] = hf - lm
        print(f"{nm:6s} HF {hf.mean():.4f}  LM {lm.mean():.4f}  mean d {D[nm].mean():+.4f}  wins {(D[nm] > 0).sum()}/{n}")
    print(f"PF optimism (in - cv), HF level: {(g('pf_in_hf') - g('pf_cv_hf')).mean():+.4f}")
    for nm in ("pf_in", "pf_cv"):
        x = D["gbm"] - D[nm]; se = x.std(ddof=1) / np.sqrt(n); h = tdist.ppf(.975, n - 1) * se
        print(f"GBM d - {nm} d: {x.mean():+.4f}  95% CI [{x.mean()-h:+.4f}, {x.mean()+h:+.4f}]  "
              f"GBM larger in {(x > 0).sum()}/{n}  Wilcoxon p = {wilcoxon(x).pvalue:.3g}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "summary":
        summary()
    elif len(sys.argv) >= 5 and sys.argv[1] == "run":
        a, b, w = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        os.makedirs(OUT, exist_ok=True)
        from multiprocessing import Pool
        with Pool(w) as p:
            for r in p.imap_unordered(run_rep, range(a, b)):
                print(f"seed {r['seed']}: PF_in {r['pf_in_hf']:.4f}/{r['pf_in_lm']:.4f}  "
                      f"PF_cv {r['pf_cv_hf']:.4f}/{r['pf_cv_lm']:.4f}  GBM {r['gbm_hf']:.4f}/{r['gbm_lm']:.4f}", flush=True)
        summary()
    else:
        print(__doc__)
