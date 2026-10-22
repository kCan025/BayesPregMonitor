import json
import math
import os
import sys

import numpy as np
from scipy.stats import rankdata, t, wilcoxon

SCENARIOS = ["baseline", "nonlinear", "heavy_tail", "combined"]
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def load(path):
    with open(path) as f:
        rows = json.load(f)
    hf = np.array([r["iAUC_hf_planC"] for r in rows], dtype=float)
    lm = np.array([r["iAUC_lm_clean"] for r in rows], dtype=float)
    return hf, lm


def manual_signed_rank_p(d):
    """Two-sided Wilcoxon signed-rank p from scratch (normal approx, tie-corrected)."""
    d = d[d != 0]
    n = len(d)
    ranks = rankdata(np.abs(d), method="average")
    w_plus = float(np.sum(ranks[d > 0]))
    mu = n * (n + 1) / 4.0
    # tie correction on the ranks of |d|
    _, tie_counts = np.unique(ranks, return_counts=True)
    tie_term = float(np.sum(tie_counts ** 3 - tie_counts))
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    z = (w_plus - mu - 0.5 * np.sign(w_plus - mu)) / np.sqrt(var)
    # two-sided normal tail via erfc
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return w_plus, z, p


def verify(path):
    hf, lm = load(path)
    n = len(hf)
    d = hf - lm
    mean_d, sd_d = float(d.mean()), float(d.std(ddof=1))
    se = sd_d / np.sqrt(n)
    ci = t.ppf(0.975, n - 1) * se
    dz = mean_d / sd_d
    p_scipy = wilcoxon(hf, lm, alternative="two-sided",
                       zero_method="wilcox").pvalue
    w_plus, z, p_manual = manual_signed_rank_p(d)
    wins, ties, losses = int(np.sum(d > 0)), int(np.sum(d == 0)), int(np.sum(d < 0))

    print(f"  file     : {os.path.basename(path)}")
    print(f"  n        : {n} valid iterations")
    print(f"  HF       : mean {hf.mean():.4f}  sd {hf.std(ddof=1):.4f}  "
          f"median {np.median(hf):.4f}")
    print(f"  LM-clean : mean {lm.mean():.4f}  sd {lm.std(ddof=1):.4f}  "
          f"median {np.median(lm):.4f}")
    print(f"  paired d : mean {mean_d:+.4f}  95% CI [{mean_d-ci:+.4f}, "
          f"{mean_d+ci:+.4f}]  dz {dz:.3f}")
    print(f"  wins/tie/loss (HF perspective): {wins}/{ties}/{losses}")
    print(f"  Wilcoxon p (scipy)           : {p_scipy:.3e}")
    print(f"  signed-rank cross-check (W+={w_plus:.0f}, z={z:.2f}, "
          f"normal p={p_manual:.3e})")
    ok = mean_d > 0 and p_scipy < 0.05 and abs(np.log10(p_scipy / p_manual)) < 0.15
    print(f"  verdict  : {'HF significantly better; cross-check agrees' if ok else 'CHECK'}")
    print()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else None
    if args:
        files = args
    else:
        suffix = f"_{tag}" if tag else ""
        files = [os.path.join(RESULTS, f"monte_carlo_{s}_iters{suffix}.json")
                 for s in SCENARIOS]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        sys.exit("No matching _iters.json files found in " + RESULTS)
    print("=" * 66)
    print("Plan C independent verification (paired HF vs LM-clean)")
    print("=" * 66)
    for f in files:
        verify(f)


if __name__ == "__main__":
    main()
