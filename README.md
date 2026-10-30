# BayesPregMonitor

> **Status (2026-10-26):** Two versions live here. **v4.0** (`results/v4.0/`, tag `v4.0-calibrated`) recalibrates the data-generating process (DGP) to a **realistic overall preeclampsia event rate of ~4%** at **N = 2,000**, adding two binary covariates (`age ≥ 33`, prior-preeclampsia history). The analysis (particle filter, multiple-imputation Cox, Plan-C iAUC) is unchanged except for passing the new covariates through. **v3.4** (tag `v3.4-archive`, the earlier `results/*.json`) used a DGP that produced an unrealistic **19–28%** event rate; its numbers are retained for history only and must not be compared in the same table with v4.0.

## v4.0 verified Plan C results (realistic ~4% event rate, N = 2,000, 80 paired replicates/scenario)

HF uses the fresh filtered latent state η(t); LM-clean freezes η at the last landmark; both share one MI-Cox score, and evaluation sits between refresh times (`eval = 22, 24, 26, 30, 34`). Independently recomputed from raw per-replicate records (`code/audit_v40.py`).

| Scenario | HF | LM-clean | mean Δ | 95% CI of mean Δ | Wilcoxon p |
|---|---|---|---|---|---|
| Baseline (linear, Gaussian) | 0.7506 | 0.7409 | +0.0097 | [+0.0064, +0.0130] | 8.4e-07 |
| Heavy-tail (linear, t₃) | 0.7504 | 0.7403 | +0.0101 | [+0.0072, +0.0129] | 8.2e-09 |
| Nonlinear (nonlinear, Gaussian) | 0.8315 | 0.8264 | +0.0051 | [+0.0035, +0.0068] | 2.2e-07 |
| Combined (nonlinear, t₃) | 0.8309 | 0.8265 | +0.0044 | [+0.0027, +0.0062] | 2.1e-06 |

All four mean differences are positive, exclude zero, and survive Bonferroni correction (α = 0.0125). **The absolute effects are small (0.4–1.0 AUC points).** The CIs above are confidence intervals for the *paired mean*; the per-replicate Δ is much more spread out (roughly −0.02 to +0.04) and is not a confidence interval.

**Realized event rates (80 replicates):** linear latent process 4.07% overall; nonlinear 4.14% (about 82 events/replicate, range 61–102). Subgroups: age ≥ 33 (n ≈ 320) 5.58% / 5.72%; age < 33 3.79% / 3.84%; prior preeclampsia (n ≈ 201) 9.82% / 9.84%; no prior history 3.43% / 3.50% (linear / nonlinear). The prior-history group has only ~20 events per replicate, so its **event rate** is reported but not subgroup discrimination. Every replicate produced a valid iAUC; in 11/80 (linear) and 5/80 (nonlinear) replicates, at least one evaluation window had fewer than five cases and was dropped.

## v4.0 DGP specification

- Cohort **N = 2,000**, horizon T = 40 weeks; Weibull baseline hazard calibrated **separately per latent mode** to ~4% overall: λ = 0.0002176 (linear) and λ = 0.0001227 (nonlinear), shape k = 1.2.
- New covariates appended after age and BMI (earlier random streams untouched): `age_ge33` (hazard ratio 1.5) and `prior_pe_history` (hazard ratio 3.0, prevalence 10%, drawn on its own `RandomState(seed+4)`, independent of the same replicate's other streams; see Limitations for cross-replicate overlap). **The hazard ratios and 10% prevalence are working assumptions, not literature estimates; reference values are being sought.**
- Particle filter, M = 5 multiple-imputation Cox, and Plan-C evaluation are unchanged. Static Cox now uses all four covariates; the legacy Landmarking comparison still uses age and BMI only.

## Limitations

- Effects are small in absolute terms; very small paired p-values reflect paired-design power, not clinical magnitude.
- **Replicate independence:** replicate k uses seeds 42+k (covariates), +1 (latent states), +2 (observations), +3 (events) and, new in v4.0, +4 (prior history), so neighbouring replicates share random-number lineages. Known overlaps: the latent-state initial draw of replicate k with the age/BMI draw of replicate k+1 (present since v3.4); and, introduced in v4.0, the prior-history draw of replicate k with the event uniforms of replicate k+1. The Wilcoxon test assumes independent pairs. The lag-1 autocorrelation of the paired difference across replicates is between -0.09 and +0.01 in the four scenarios (standard error under independence about 0.11); the effect on the p-values has not been quantified further. A stream-separation fix is planned for v4.1 and will regenerate all numbers.
- The estimated latent-state coefficient is β_η ≈ 0.769–0.774 versus the generating γ = 0.8 (v3.4: 0.777–0.790); the reason for the difference is not investigated here and no causal explanation is claimed.
- HRs and prior-history prevalence are assumptions pending external reference values.

## Project structure

```
BayesPregMonitor/
├── code/
│   ├── simulate_data.py        # DGP (v4.0: N=2000, 4% calibration, 4 covariates)
│   ├── bootstrap_pf.py         # Bootstrap particle filter (Np=100)
│   ├── time_varying_cox.py     # Multiple-imputation time-varying Cox
│   ├── benchmarks.py           # Static Cox, landmarking, time-dependent AUC
│   ├── adaptive_threshold.py   # Decision curve analysis
│   ├── run_monte_carlo.py      # Paired MC over 4 scenarios (output: results/$RESULTS_SUBDIR)
│   ├── run_step1.py / run_step2.py / run_step3.py
│   ├── verify_planC.py         # From-scratch Plan C check on v3.4 records
│   ├── regression_v34.py       # v4.0-code vs all 160 committed v3.4 event rates
│   ├── audit_v40.py            # Independent recomputation from raw v4.0 records
│   └── (diagnostic/plotting helpers)
├── results/
│   ├── v4.0/                   # v4.0 calibrated run: chunks/ + merge_report
│   └── *.json                  # archived outputs (v3.4 MC + older single-run/diagnostic); not comparable to v4.0
├── PLAN_C.md
├── BUGFIX_PLAN.md
├── LICENSE                     # MIT
└── README.md
```

> **Single-run scripts.** `run_step1.py`, `run_step2.py` and `run_step3.py` each call
> `simulate(seed=42)`, which now runs the v4.0 data-generating process, and they write to
> the single-run files in `results/`. Re-running them overwrites those files. They are not
> part of the v4.0 Monte Carlo pipeline.
>
> **Archived outputs.** The loose files in `results/` are not all v3.4. For example
> `data_summary.json` records N = 500, lambda = 0.0008 and a 10.6% event rate, which matches
> neither v3.4 (lambda = 0.0015, 19-28% event rate) nor v4.0 (~4% event rate).
>
> **Figures.** `code/plot_figures.py` reads the archived `results/monte_carlo_results.json`
> and `results/decision_results.json` for Figures 2, 4 and 5, while Figure 3 comes from a
> fresh `simulate(seed=42)` run. Re-running it therefore produces a figure set that mixes two
> data-generating processes.

## Quick start

Output is written to `results/$RESULTS_SUBDIR` (default `v4.0`) so a re-run never overwrites another version's files.

```bash
cd code
RESULTS_SUBDIR=smoke python run_monte_carlo.py --scenario baseline --n_mc 2   # smoke test, own folder
python run_monte_carlo.py --scenario baseline --n_mc 80         # one scenario
python run_monte_carlo.py --run_all --n_mc 80                   # all four
# Independent verification of a run (regression gate + raw-record audit):
python regression_v34.py . ../results
python audit_v40.py ../results/v4.0/chunks --report ../results/v4.0/merge_report.json
```

Runtime is hardware-dependent (single-core, roughly 20–25 minutes per scenario of 80 on the reference machine). Dependencies: Python ≥ 3.9, numpy, scipy, pandas, lifelines, matplotlib, scikit-learn.

## Version history

- **v4.0.1 (2026-10-28):** documentation and audit-script fixes only; all numbers unchanged.
- **v4.0 (2026-10-26):** realistic ~4% event rate, N = 2,000, two added risk-factor covariates; analysis unchanged.
- **v3.4 (2026-10-22, tag `v3.4-archive`; tag `v1.2-planc`):** Plan C under a 19–28% event DGP — archived, do not cite for calibrated claims. Plan C fixed two design flaws (in-sample optimism; evaluation coinciding with model refresh). See `PLAN_C.md`.
- **v1.1-bugfix:** independent per-generator seeds; true landmark Cox refit; decision-curve fix (see `BUGFIX_PLAN.md`). The NISS poster shows pre-fix numbers.

## Citation & acknowledgments

Wang, K. (2026). *BayesPregMonitor: A Paired Comparison Framework for Continuous vs. Discrete Dynamic Risk Monitoring.* NISS S3 Ingram Olkin Forum, Columbus, OH. Thanks to [@Osakazuha](https://github.com/Osakazuha) for the code review that caught the three issues fixed in v1.1.

## License

MIT License — see [LICENSE](./LICENSE).
