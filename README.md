# BayesPregMonitor

> **Status (2026-10-22, work in progress):** The `v1.1-bugfix` landmark benchmark below was found to be (i) in-sample (Cox fit and evaluated on the same outcome window), with EPV ≈ 3–4 per landmark, and (ii) structurally unable to measure cadence because evaluation times coincide with model-refresh times. Both the legacy and the refit landmark numbers are superseded by **Plan C** (`iAUC_hf_planC` / `iAUC_lm_clean` in `run_monte_carlo.py`): a single MI-Cox score, η frozen between landmarks, and evaluation between refresh times. Plan C results are under independent verification; validated numbers will replace the tables below. See `PLAN_C.md`.

Bayesian Particle Filter + Time-Varying Cox Model for Dynamic Maternal Risk Monitoring.

A simulation-based framework for comparing **continuous monitoring** (high-frequency particle filter + time-varying Cox) against **discrete landmark-based monitoring**, using a paired design that isolates the effect of monitoring cadence.

---

## ⚠️ Post-NISS Update: Bug Fixes (v1.1-bugfix)

After presenting at [NISS S3 Ingram Olkin Forum](https://www.niss.org/events/s3-ingram-olkin-forum-challenges-modeling-and-analyzing-maternal-health-data) (Oct 15–16, 2026), a code review by [@Osakazuha](https://github.com/Osakazuha) identified three issues. All three have been fixed in the `v1.1-bugfix` tag. The NISS poster shows pre-fix numbers.

| Bug | File | Severity | Fix |
|-----|------|----------|-----|
| #1: Same seed across all DGP generators | `simulate_data.py` | High | Independent seeds per generator (`seed`, `seed+1`, `seed+2`, `seed+3`) |
| #2: Landmarking copies PF eta instead of refitting Cox | `benchmarks.py` | Critical | True landmarking with `lifelines.CoxPHFitter` refit at each landmark time |
| #3: Cost-ratio argmax always picks minimum | `adaptive_threshold.py` | Medium | Removed optimization; standard decision curve with treat-all/treat-none references |

### Post-fix MC results (4 scenarios × 80 reps)

| Scenario | HF iAUC | LM iAUC | HF-LM diff | Wilcoxon p |
|----------|---------|---------|------------|------------|
| Baseline (linear + Gaussian) | 0.7173 | 0.7127 | +0.0045 | 0.008 ** |
| Heavy-tail (linear + t(3)) | 0.7158 | 0.7120 | +0.0038 | 0.026 * |
| Nonlinear (nonlinear + Gaussian) | 0.8352 | 0.8338 | +0.0014 | 0.086 |
| Combined (nonlinear + t(3)) | 0.8352 | 0.8339 | +0.0013 | 0.189 |

**Key findings:**
- Direction (HF ≥ LM) holds in all 4 scenarios.
- Statistical significance holds under **linear dynamics** (p < 0.05) but not under **nonlinear dynamics**.
- The gap shrinkage under nonlinear dynamics is consistent with a **ceiling effect**: more complex trajectories improve PF tracking accuracy (r: 0.978 → 0.993), pushing both methods toward their performance ceiling.
- Static Cox drops to ~0.53 (near chance) and no longer varies across scenarios — the old ~0.60–0.69 values were an artifact of Bug #1 (age/BMI acting as theta₀ proxy).

Full details in [BUGFIX_PLAN.md](./BUGFIX_PLAN.md).

---

## Project Structure

```
BayesPregMonitor/
├── code/
│   ├── simulate_data.py        # Data generation process (4 scenarios)
│   ├── bootstrap_pf.py         # Bootstrap particle filter (Np=100)
│   ├── benchmarks.py           # Static Cox, landmarking, low-freq benchmarks
│   ├── adaptive_threshold.py   # Decision layer + cost-sensitive threshold + DCA
│   ├── run_step12.py           # Layer 1+2: PF + Cox pipeline
│   ├── run_step3.py            # Layer 3: Decision analysis
│   ├── run_monte_carlo.py      # Full MC simulation (4 scenarios)
│   └── plot_figures.py         # Visualization
├── results/                    # MC output (JSON)
├── figures/                    # Plots
├── BUGFIX_PLAN.md              # Detailed bug analysis + fix plan
└── ISSUE_REPLY_DRAFT.md        # GitHub issue reply draft
```

---

## Method

**Three-layer pipeline:**

1. **Layer 1 — Particle Filter:** Bootstrap particle filter (Np=100) estimates the posterior distribution of the latent risk trajectory θ(t) from noisy longitudinal observations.
2. **Layer 2 — Time-Varying Cox:** Multiple imputation (Rubin's Rules) with M=5 imputations to combine PF uncertainty with a Cox proportional hazards model including baseline covariates (age, BMI).
3. **Layer 3 — Decision Curve Analysis:** Cost-sensitive thresholding evaluated across the full cost-ratio frontier (treat-all / treat-none reference lines, no spurious "optimal" selection).

**Paired comparison design:** HF and LM benchmarks share the same PF eta trajectory within each simulation, so the performance difference isolates the effect of monitoring cadence rather than data variability.

**2×2 factorial misspecification design:**
- Linear vs. nonlinear state dynamics
- Gaussian vs. heavy-tailed (t₃) observation noise

---

## Quick Start

```bash
cd code

# Single iteration (smoke test)
python run_monte_carlo.py --scenario baseline --n_mc 1

# Full MC for one scenario (80 reps, ~3.5 min)
python run_monte_carlo.py --scenario baseline --n_mc 80

# All four scenarios
python run_monte_carlo.py --scenario nonlinear --n_mc 80
python run_monte_carlo.py --scenario heavy_tail --n_mc 80
python run_monte_carlo.py --scenario combined --n_mc 80
```

**Dependencies:**
- Python ≥ 3.9
- numpy, scipy, pandas
- lifelines (for landmark Cox refitting)
- matplotlib (for plots)

---

## Citation & Acknowledgments

If you use this code or framework, please cite:

> Wang, K. (2026). BayesPregMonitor: A Paired Comparison Framework for Continuous vs. Discrete Dynamic Risk Monitoring. NISS S3 Ingram Olkin Forum, Columbus, OH.

Thanks to [@Osakazuha](https://github.com/Osakazuha) for the careful code review that caught the three bugs fixed in v1.1.

---

## License

MIT License

---

