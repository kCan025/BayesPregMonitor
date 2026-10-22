#!/usr/bin/env python3
"""
Independent Plan C verification.

Reads ONLY the per-iteration JSON records (no model code, no re-running).
Recomputes, from scratch, the paired comparison:
  mean / SD / median of HF and LM-clean iAUC,
  mean paired difference, win/tie/loss counts,
  two-sided paired Wilcoxon signed-rank p-value.

Usage:
  python verify_planC.py                       # all 4 scenarios, std files
  python verify_planC.py path/to/file.json     # one file
  python verify_planC.py --tag 20261022_1157   # timestamped files

Reference (cloud run, 2026-10-22 80 reps):
  baseline   HF .7205  LMc .7070  diff +.0135  62:18  p 5.16e-09
  nonlinear  HF .8394  LMc .8326  diff +.0067  69:11  p 8.79e-12
  heavy_tail HF .7212  LMc .7055  diff +.0157  65:15  p 2.84e-10
  combined   HF .8385  LMc .8331  diff +.0054  65:15  p 8.55e-10
(combined independently reproduced on WSL/Python 3.13, 2026-10-22 23:39.)
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

SCENARIOS = ["baseline", "nonlinear", "heavy_tail", "combined"]
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def load(path):
    with open(path) as f:
        rows = json.load(f)
    hf = np.array([r["iAUC_hf_planC"] for r in rows], dtype=float)
    lm = np.array([r["iAUC_lm_clean"] for r in rows], dtype=float)
    return hf, lm


def verify(path):
    hf, lm = load(path)
    d = hf - lm
    nonzero = d[d != 0]
    try:
        p = wilcoxon(hf, lm, alternative="two-sided", zero_method="wilcox").pvalue
    except ValueError as e:
        p = float("nan")
    wins = int(np.sum(d > 0))
    ties = int(np.sum(d == 0))
    losses = int(np.sum(d < 0))
    print(f"  file     : {os.path.basename(path)}")
    print(f"  n        : {len(d)} valid iterations")
    print(f"  HF       : mean {hf.mean():.4f}  sd {hf.std(ddof=1):.4f}  "
          f"median {np.median(hf):.4f}")
    print(f"  LM-clean : mean {lm.mean():.4f}  sd {lm.std(ddof=1):.4f}  "
          f"median {np.median(lm):.4f}")
    print(f"  HF-LMc   : mean {d.mean():+.4f}  median {np.median(d):+.4f}")
    print(f"  wins/tie/loss (HF perspective): {wins}/{ties}/{losses}")
    print(f"  Wilcoxon two-sided p : {p:.3e}  "
          f"(nonzero diffs used: {len(nonzero)})")
    print(f"  verdict  : {'HF significantly better' if (d.mean() > 0 and p < 0.05) else 'CHECK'}")
    print()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tag = None
    if "--tag" in sys.argv:
        tag = sys.argv[sys.argv.index("--tag") + 1]
    if args:
        files = args
    elif tag:
        files = [os.path.join(RESULTS, f"monte_carlo_{s}_iters_{tag}.json")
                 for s in SCENARIOS]
    else:
        files = [os.path.join(RESULTS, f"monte_carlo_{s}_iters.json")
                 for s in SCENARIOS]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        sys.exit("No matching _iters.json files found in " + RESULTS)
    print("=" * 64)
    print("Plan C independent verification (paired HF vs LM-clean)")
    print("=" * 64)
    for f in files:
        verify(f)


if __name__ == "__main__":
    main()
