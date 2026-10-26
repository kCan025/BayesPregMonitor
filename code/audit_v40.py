#!/usr/bin/env python3
"""audit_v40.py -- independent recomputation of the v4.0 Monte Carlo summary
from RAW per-replicate files.

Usage:
    python3 audit_v40.py CHUNK_DIR [--report merge_report.json]

CHUNK_DIR holds _000_040.json / _040_080.json (or monte_carlo__iters.json).
Reads only the raw rows; recomputes everything; with --report, asserts the
published merge_report.json agrees with the recomputation. Nothing here is
imported from the project code.
"""
import glob, json, math, os, sys
import numpy as np
from scipy.stats import wilcoxon

SCEN = ["baseline", "nonlinear", "heavy_tail", "combined"]

rows = {}
d = sys.argv[1]
for sc in SCEN:
    files = sorted(glob.glob(os.path.join(d, f"{sc}_[0-9]*_[0-9]*.json"))) or \
            sorted(glob.glob(os.path.join(d, f"monte_carlo_{sc}_iters.json")))
    if not files:
        sys.exit(f"no files for scenario {sc} in {d}")
    r = []
    for f in files:
        r += json.load(open(f))
    rows[sc] = r


def arr(r, k):
    return np.array([np.nan if x.get(k) is None else x.get(k, np.nan)
                     for x in r], float)


report = json.load(open(sys.argv[sys.argv.index("--report") + 1])) \
    if "--report" in sys.argv else None

bad = []


def check(name, mine, theirs, tol=1e-9):
    if theirs is None or abs(mine - theirs) > tol * max(1, abs(theirs)):
        bad.append(f"{name}: recomputed {mine!r} vs report {theirs!r}")


print(f"{'scenario':<11}{'n':>4}{'NaN':>5}{'HF':>8}{'LMc':>8}{'mean Δ':>9}"
      f"{'SD(Δ)':>8}{'95% CI of mean Δ':>20}{'per-rep 2.5–97.5%':>20}"
      f"{'wins':>8}{'Wilcoxon p':>12}")
for sc in SCEN:
    r = rows[sc]
    mc = [x.get("_mc") for x in r]
    if all(m is not None for m in mc):
        assert sorted(mc) == list(range(len(r))), (
            f"{sc}: replicate indices not 0..{len(r)-1} (gaps/duplicates)")
    h, l = arr(r, "iAUC_hf_planC"), arr(r, "iAUC_lm_clean")
    ok = ~np.isnan(h) & ~np.isnan(l)
    dif = (h - l)[ok]
    se = dif.std(ddof=1) / math.sqrt(len(dif))
    ci = (dif.mean() - 1.96 * se, dif.mean() + 1.96 * se)
    q = np.percentile(dif, [2.5, 97.5])
    p = wilcoxon(h[ok], l[ok]).pvalue
    print(f"{sc:<11}{len(r):>4}{int((~ok).sum()):>5}{h[ok].mean():>8.4f}"
          f"{l[ok].mean():>8.4f}{dif.mean():>+9.4f}{dif.std(ddof=1):>8.4f}"
          f"{'[%+.4f,%+.4f]' % ci:>20}{'[%+.3f,%+.3f]' % tuple(q):>20}"
          f"{'%d/%d' % ((dif > 0).sum(), (dif < 0).sum()):>8}{p:>12.2e}")
    er = arr(r, "event_rate")
    ev = [x["_audit"]["n_events_total"] for x in r
          if isinstance(x.get("_audit"), dict)
          and "n_events_total" in x["_audit"]]
    print(f"{'':<11}event rate mean {er.mean():.4%} (rep range "
          f"{er.min():.2%}-{er.max():.2%})"
          + (f" | events/rep mean {np.mean(ev):.1f} [{min(ev)}, {max(ev)}]"
             if ev else "")
          + (f" | PF r {np.nanmean(arr(r,'pf_r')):.4f} "
             f"| medESS {np.nanmean(arr(r,'median_ess')):.1f}"
             f" | Cox β_η {np.nanmean(arr(r,'cox_beta_eta')):.4f}"
             f" | ESS warnings {int(np.nansum(arr(r,'ess_warning')))}"))

    if report:
        s = report["scenarios"][sc]
        check(f"{sc} HF", h[ok].mean(),
              s["summary"]["HF (planC eval)"]["mean"])
        check(f"{sc} LMc", l[ok].mean(),
              s["summary"]["LM-clean (planC eval)"]["mean"])
        check(f"{sc} mean diff", dif.mean(),
              s["paired"]["HF_vs_LM_clean"]["mean_diff"])
        check(f"{sc} wilcoxon p", p,
              s["paired"]["HF_vs_LM_clean"]["wilcoxon_p"], 1e-6)
        check(f"{sc} event rate", er.mean(),
              s["diagnostics"]["event_rate_mean"])
        check(f"{sc} wins", int((dif > 0).sum()),
              s["paired"]["HF_vs_LM_clean"]["HF_wins"])

# structural sanity: events depend only on the latent process,
# static Cox only on baseline covariates
print("\nStructural checks (must be exactly equal):")
for a, b in (("baseline", "heavy_tail"), ("nonlinear", "combined")):
    same_ev = np.array_equal(arr(rows[a], "event_rate"),
                             arr(rows[b], "event_rate"))
    sa, sb = arr(rows[a], "iAUC_static"), arr(rows[b], "iAUC_static")
    print(f" {a} vs {b}: event_rate identical per rep = {same_ev}; "
          f"static-Cox iAUC identical = "
          f"{np.allclose(sa, sb, equal_nan=True, atol=1e-12)}")

# subgroup / window audit, if present in the rows
for sc in SCEN:
    au = [x.get("_audit") for x in rows[sc]
          if isinstance(x.get("_audit"), dict)]
    if au and "windows" in au[0]:
        print(f"\n[{sc}] keys in _audit: {sorted(au[0].keys())}")
        break

if report:
    print("\nReport vs raw recomputation:",
          "ALL MATCH" if not bad else "MISMATCH")
    for b in bad:
        print(" ", b)
sys.exit(1 if bad else 0)
