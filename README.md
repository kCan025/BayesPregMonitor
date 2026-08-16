# BayesPregMonitor

**A Bayesian Pipeline for Dynamic Maternal Risk Monitoring: From Wearables to Cost-Sensitive Decisions**

Three-layer computational framework that translates high-frequency wearable data into real-time clinical intervention decisions for prenatal monitoring.

## Framework

```
Layer 1: Bootstrap Particle Filter
  - Latent risk state θ_i(t) as Gaussian random walk
  - 4 wearable signals (HRV, sleep, activity, mood) as linear Gaussian emissions
  - Output: scalar dynamic risk score η_i(t)
  - Marginal MI (M=5) propagates filtering uncertainty
       |
       v
Layer 2: Extended Cox Proportional Hazards
  - η_i(t) as time-varying covariate + baseline covariates X_i
  - Counting process format (start, stop, event)
  - Rubin's rules for MI pooling
       |
       v
Layer 3: Cost-Sensitive Decision Rule
  - P(T <= t+Δt | T > t, H_i(t)) = 1 - S(t+Δt)/S(t)
  - Threshold d* = C_FP / (C_FP + C_FN)
  - Cost-benefit frontier via ratio sweeping
```

## DGP Parameters (v3.3)

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| State noise variance | σ²_ω | 0.04 | Controls latent trajectory smoothness |
| Observation loadings | α | (1.0, -0.8, -0.6, -0.7)' | 4-dim wearable linear mapping |
| Observation noise cov | Σ_ε | diag(0.25, 0.16, 0.36, 0.09) | Diagonal noise matrix |
| Risk-hazard link | γ | 0.5 | Latent state to event hazard |
| Weibull shape | k | 1.2 | Baseline hazard time-variation |
| Weibull scale | λ | 0.008 | Baseline hazard level |
| Baseline effects | β | (0.03, 0.05)' | Age and BMI effects |

## Running

```bash
# Step 1: Data generation + Bootstrap PF + Time-varying Cox
cd code
python run_step1.py

# Individual modules
python simulate_data.py        # Generate synthetic data
python bootstrap_pf.py         # Run particle filter
python time_varying_cox.py     # Fit Cox models (naive + MI)
python plot_fig1.py            # Generate Figure 1
```

## Project Structure

```
BayesPregMonitor/
├── README.md
├── code/
│   ├── simulate_data.py         # DGP: N=500 subjects, 4 signals + event times
│   ├── bootstrap_pf.py          # Layer 1: Bootstrap PF + marginal MI
│   ├── time_varying_cox.py      # Layer 2: Extended Cox with counting process
│   ├── adaptive_threshold.py    # Layer 3: Cost-sensitive decision rule
│   ├── benchmarks.py            # Static Cox, Landmarking, JMbayes2
│   ├── plot_fig1.py             # Visualization
│   ├── run_step1.py             # Main pipeline (Steps 1-2)
│   └── requirements.txt
├── figures/                     # Generated figures
└── results/                     # Saved results (JSON, NPZ)
```

## Design Notes

- **Marginal MI approximation**: At each time point, weighted resampling from the filtered posterior produces M=5 imputed trajectories. This is computationally tractable but does not account for temporal dependence in the joint smoothing distribution. Future: backward simulation.
- **Fair comparison**: To isolate data-frequency effects from methodological advantage, wearables are downsampled to 4-week clinical frequency for comparison against JMbayes2.
- **Event rates and covariate distributions** are informed by the PIERS-ML cohort.

## Reference

SCD 2026 / NISS Ingram Olkin Forum 2026

## License

MIT
