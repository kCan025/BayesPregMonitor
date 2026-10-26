#!/usr/bin/env python3
"""regression_v34.py -- the regression gate.

With the OLD parameters (N=500, lambda=0.0015, beta=[0.005,0.005], first two
covariate columns only) the patched simulate_data.py must reproduce the event
rate of EVERY committed v3.4 replicate (80 per latent mode), not just seeds
42-46.

    python3 regression_v34.py CODE_DIR COMMITTED_RESULTS_DIR
    (e.g.  python3 regression_v34.py . ../results)

Reads monte_carlo_baseline_iters.json (linear) and
monte_carlo_nonlinear_iters.json (nonlinear) from the COMMITTED v3.4 results
(git show v3.4-archive:results/... if you moved them). Exit 1 on any mismatch.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, sys.argv[1])
import simulate_data as S

res = sys.argv[2]
worst, n_bad, n = 0.0, 0, 0
for mode, name in (("linear", "baseline"), ("nonlinear", "nonlinear")):
    committed = json.load(open(os.path.join(res, f"monte_carlo_{name}_iters.json")))
    for mc, row in enumerate(committed):
        seed = 42 + mc
        p = S.DGPParams()
        p.N, p.lambda_weibull, p.beta = 500, 0.0015, np.array([0.005, 0.005])
        X = S.generate_baseline_covariates(500, seed=seed)[:, :2]
        th = S.generate_latent_trajectories(500, 40, p.sigma2_omega,
                                            seed=seed + 1, mode=mode)
        _, ev = S.generate_event_times(th, X, p, seed=seed + 3)
        diff = abs(float(ev.mean()) - row["event_rate"])
        worst, n = max(worst, diff), n + 1
        if diff > 1e-12:
            n_bad += 1
            print(f" MISMATCH {name} seed {seed}: got {ev.mean():.6f} "
                  f"committed {row['event_rate']:.6f}")

print(f"checked {n} replicates, max |diff| = {worst:.2e}, mismatches = {n_bad}")
print("REGRESSION GATE: PASS" if n_bad == 0 else
      "REGRESSION GATE: FAIL -- do not proceed")
sys.exit(1 if n_bad else 0)
