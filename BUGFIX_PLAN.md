# BayesPregMonitor Bugfix Plan

**Date:** 2026-10-16  
**Trigger:** GitHub issue from @Osakazuha ("Some errors")  
**Scope:** Three code-level defects affecting DGP, benchmarking, and decision layer

---

## Bug #1: Same Seed Reused Across DGP Generators

### Problem
`simulate_data.py`, `simulate()` function (line 269-286):

```python
# CURRENT (BUGGY)
X = generate_baseline_covariates(params.N, seed=seed)           # seed=42
theta = generate_latent_trajectories(..., seed=seed, ...)      # seed=42
Y = generate_observations(theta, ..., seed=seed)               # seed=42
event_times = generate_event_times(theta, X, params, seed=seed) # seed=42
```

All four generators create `np.random.RandomState(seed)` from the same seed → identical random streams → `corr(age, theta_0) = 0.998` (verified).

### Fix
```python
# FIXED
X = generate_baseline_covariates(params.N, seed=seed)
theta = generate_latent_trajectories(params.N, params.T_weeks,
                                     params.sigma2_omega,
                                     seed=seed + 1,           # independent stream
                                     mode=theta_mode)
if obs_noise == "heavy_tail":
    Y = generate_observations_heavy_tail(theta, params.alpha,
                                         params.Sigma_eps, df=heavy_tail_df,
                                         seed=seed + 2)
else:
    Y = generate_observations(theta, params.alpha, params.Sigma_eps,
                              seed=seed + 2)
event_times, event_indicators = generate_event_times(
    theta, X, params, seed=seed + 3)
```

### Verification
```python
from simulate_data import simulate
import numpy as np

# Before fix: corr ≈ 0.998
# After fix: corr should be near 0
data = simulate(seed=42, theta_mode="linear", obs_noise="gaussian", verbose=False)
r = np.corrcoef(data.X[:, 0], data.theta[0, :])[0, 1]
print(f"corr(age, theta_0) = {r:.4f}")  # expect |r| < 0.1
```

### Impact Assessment
- **DGP validity**: HIGH — age/BMI no longer proxies for theta_0; Cox beta_X estimates become interpretable
- **Paired comparison**: MINIMAL — paired design shares theta trajectory and event times within each pair; the seed change affects both arms equally within a pair
- **MC reproducibility**: results from all 80 MC reps will change numerically, but paired differences may remain directionally consistent
- **Posters/papers**: all numerical results need re-computation; directional conclusions need re-verification

---

## Bug #2: Landmarking Benchmark Reuses PF Output

### Problem
`benchmarks.py`, `landmarking_benchmark()` function (line 317-318):

```python
# CURRENT (BUGGY)
risk_scores_tv = eta_all.copy()  # directly copies PF posterior mean
```

This is NOT landmarking. The function takes the HF pipeline's PF output and evaluates AUC at 3 time points. The only difference between HF and LM is that HF adds a Cox layer (`beta_eta * eta + beta_X * X`) while LM uses raw `eta`. The "HF > LM" gap thus reflects Cox covariate adjustment, not continuous vs. discrete monitoring.

### Fix: True Landmarking with Independent Cox Refitting

```python
def landmarking_benchmark(
    data: SimulatedData,
    eta_all: np.ndarray,          # still receives PF eta for paired design
    landmark_times: List[int] = None,
    delta_t: int = 4,
    eval_times: List[int] = None
) -> Dict:
    """
    True landmark analysis: at each landmark t_L, refit Cox using
    cumulative data up to t_L, then predict event in (t_L, t_L+delta_t].

    This is an INDEPENDENT baseline — does not reuse HF pipeline's
    Cox coefficients. The only shared component is the PF eta trajectory
    (shared by paired design), but Cox is refit independently at each landmark.

    Key difference from HF:
    - HF: uses ALL time points (weekly) for Cox risk score
    - LM: only updates risk assessment at 3 landmark times
    """
    from lifelines import CoxPHFitter
    import pandas as pd

    if landmark_times is None:
        landmark_times = [20, 28, 32]

    if eval_times is None:
        eval_times = list(range(12, 37, 4))

    T, N = eta_all.shape
    # Risk scores: only defined at landmark times, held constant between
    risk_scores_lm = np.zeros((T, N))

    for t_L in landmark_times:
        t_idx = min(t_L - 1, T - 1)  # 0-indexed

        # Subjects at risk at t_L
        at_risk = data.event_times > t_L
        if at_risk.sum() < 10:
            continue

        # Build landmark dataset: baseline covariates + eta at t_L
        lm_df = pd.DataFrame({
            'eta': eta_all[t_idx, at_risk],
            'age': data.X[at_risk, 0],
            'bmi': data.X[at_risk, 1],
        })

        # Landmark outcome: event in (t_L, t_L + delta_t]
        t_end = t_L + delta_t
        event_in_window = (
            (data.event_times[at_risk] > t_L) &
            (data.event_times[at_risk] <= t_end) &
            (data.event_indicators[at_risk] == 1)
        )
        # Observed time for landmark subjects
        obs_time = np.minimum(data.event_times[at_risk], t_end) - t_L
        lm_df['duration'] = obs_time
        lm_df['event'] = event_in_window.astype(int)

        # Refit Cox at this landmark
        try:
            cph = CoxPHFitter()
            cph.fit(lm_df, duration_col='duration', event_col='event',
                    formula='eta + age + bmi')

            # Predict risk for ALL subjects (including those not at risk,
            # for consistent risk_scores array shape)
            full_df = pd.DataFrame({
                'eta': eta_all[t_idx, :],
                'age': data.X[:, 0],
                'bmi': data.X[:, 1],
            })
            pred_risk = cph.predict_partial_hazard(full_df).values

            # Fill risk scores from t_L until next landmark (or end)
            next_lm = min(
                [lm for lm in landmark_times if lm > t_L] + [T + 1]
            )
            for t in range(t_L, min(next_lm, T + 1)):
                if t - 1 < T:
                    risk_scores_lm[t - 1, :] = pred_risk
        except Exception as e:
            print(f"  Landmark t={t_L}: Cox fit failed: {e}")
            # Fallback: use eta directly at this landmark only
            risk_scores_lm[t_idx, :] = eta_all[t_idx, :]

    # Compute iAUC at landmark evaluation times only
    auc_result = time_dependent_auc(
        data.event_times, data.event_indicators, risk_scores_lm,
        eval_times=landmark_times, delta_t=delta_t)

    print(f"\n--- Landmarking (true refit) ---")
    print(f"  iAUC (at landmarks, pair-weighted): {auc_result['iAUC']:.4f}")

    return {
        'auc_result': auc_result,
        'risk_scores': risk_scores_lm,
        'iAUC': auc_result['iAUC'],
        'method': 'Landmarking (refit)',
        'landmark_times': landmark_times,
    }
```

### Key Design Decisions
1. **Still uses PF eta as input** — this preserves paired design fairness (both HF and LM use the same PF trajectory, only differ in update frequency)
2. **Refits Cox independently at each landmark** — LM no longer piggybacks on HF's Cox coefficients
3. **Risk held constant between landmarks** — this is the defining feature of landmark monitoring (discrete updates vs. continuous)
4. **If Cox fit fails** (too few events at a landmark), falls back to raw eta at that timepoint only

### Alternative: Fully Independent Landmarking (No PF)
If you want LM to be completely independent of PF:
- At each landmark, use cumulative raw observations Y[:, :t_L, :] to compute summary features (e.g., rolling mean, slope, max deviation)
- Fit Cox with these features + baseline covariates
- This makes LM a genuinely different methodology, not just a downsampled version of HF

**Recommendation**: Start with the refit version above (shares PF eta but refits Cox). If the HF-LM gap is still suspiciously small after this fix, escalate to the fully independent version.

### Verification
```python
# After fix, check that HF and LM risk scores are NOT near-identical
from scipy.stats import spearmanr
rho, _ = spearmanr(hf_risk.flatten(), lm_risk.flatten())
print(f"Spearman(HF, LM) = {rho:.4f}")
# Before fix: 0.998 (Osakazuha reported)
# After fix: should be substantially lower
```

### Impact Assessment
- **Core conclusion**: CRITICAL — if HF-LM gap changes substantially, the "HF > LM" narrative needs revision
- **Possible outcomes**:
  - (a) HF still > LM with similar margin → original conclusion holds, just needs cleaner benchmark
  - (b) HF still > LM but margin grows → original conclusion was understated
  - (c) HF-LM gap shrinks to near-zero → core claim weakens; pivot to "robustness of paired framework" as primary contribution
  - (d) LM > HF in some scenarios → complete rethink needed
- **Poster/abstract**: cannot update before NISS; need honest on-site framing

---

## Bug #3: Cost Ratio Sweep Always Picks Minimum

### Problem
`adaptive_threshold.py`, `sweep_cost_ratios()` (line 234-235):

```python
net_benefits = [r['net_benefit'] for r in results]
opt_idx = np.argmax(net_benefits)
```

Net benefit = TP/N - FP/N × cost_ratio. Lower cost_ratio → less FP penalty → higher NB. So argmax always returns the smallest cost_ratio (0.01), which flags ~80% of decision points. The "optimal cost ratio" concept is meaningless in Decision Curve Analysis — cost ratio is a clinical prior, not a data-optimized parameter.

### Fix: Remove Optimization, Use Decision Curve

```python
def sweep_cost_ratios(
    prob_event, event_times, event_indicators,
    cost_ratios=None, delta_t=4, eval_times=None
) -> Dict:
    """
    Sweep cost ratios for decision curve visualization.
    NOTE: Does NOT select an 'optimal' cost ratio — that is a clinical
    prior, not a data-optimized parameter. Returns the full frontier
    for the decision-maker to inspect.
    """
    if cost_ratios is None:
        cost_ratios = np.logspace(-2, 2, 30)

    results = []
    for cr in cost_ratios:
        res = cost_sensitive_threshold(
            prob_event, event_times, event_indicators,
            cost_ratio=cr, delta_t=delta_t, eval_times=eval_times)
        res['cost_ratio'] = float(cr)
        results.append(res)

    net_benefits = [r['net_benefit'] for r in results]

    # REMOVED: argmax selection of "optimal" cost ratio
    # ADDED: treat-all and treat-none reference lines for decision curve
    N_total = len(event_times)
    n_events = int(event_indicators.sum())
    n_nonevents = N_total - n_events
    treat_all_nb = n_events / N_total  # at any cost_ratio

    return {
        'frontier': results,
        'cost_ratios': cost_ratios.tolist(),
        'net_benefits': net_benefits,
        'treat_all_nb': treat_all_nb,
        'treat_none_nb': 0.0,
        # No 'optimal_cost_ratio' or 'max_net_benefit' — removed by design
    }
```

In `compare_decision_strategies()` and any downstream code that references `sweep_result['optimal_cost_ratio']` or `sweep_result['max_net_benefit']`, replace with:

```python
# Instead of selecting "optimal", evaluate at a clinically motivated cost ratio
# (e.g., 1.0 = equal cost for FP and FN, or a domain-specific value)
clinical_cost_ratio = 1.0
strategies['Cost-sensitive (clinical prior)'] = cost_sensitive_threshold(
    prob_event, event_times, event_indicators,
    cost_ratio=clinical_cost_ratio, delta_t=delta_t, eval_times=eval_times)
```

### Impact Assessment
- **Core conclusion**: LOW — Layer 3 (cost-sensitive alerting) is not the poster's primary claim
- **Poster**: the decision layer section is supplementary; removing "optimal cost ratio" claims is straightforward
- **Future thesis**: the leaky-integrator / allostatic-load direction (Direction A) does not depend on Layer 3

---

## Execution Sequence

### Phase 1: Fix & Verify (Local, ~2-3 hours)

| Step | File | Action | Verification |
|------|------|--------|-------------|
| 1.1 | simulate_data.py | Decouple seeds (seed, seed+1, seed+2, seed+3) | corr(age, theta_0) ≈ 0 |
| 1.2 | benchmarks.py | Rewrite landmarking_benchmark with true refit | Spearman(HF, LM) << 0.998 |
| 1.3 | adaptive_threshold.py | Remove argmax, add decision curve refs | No "optimal_cost_ratio" in output |
| 1.4 | run_monte_carlo.py | Update to pass landmark_times and eval_times correctly to new LM | Single-seed smoke test passes |

### Phase 2: Single-Seed Diagnostic (Local, ~30 min)

Run one seed per scenario (4 runs total instead of 80):
```bash
cd /path/to/BayesPregMonitor/code
python run_monte_carlo.py --n_iters 1 --scenarios all
```

Check:
- DGP produces reasonable corr(age, theta_0) (near 0, not 0.998)
- LM benchmark runs without error (Cox refit at each landmark)
- HF-LM iAUC gap is no longer suspiciously tiny
- No "optimal_cost_ratio" in Layer 3 output

### Phase 3: Full MC Rerun (Local, ~4-8 hours)

```bash
python run_monte_carlo.py --n_iters 80 --scenarios all
```

### Phase 4: Impact Assessment — COMPLETE (2026-10-17)

**Full 4-scenario MC rerun (80 reps each), post-fix results:**

| Metric | Baseline | Heavy_tail | Nonlinear | Combined |
|--------|----------|------------|-----------|----------|
| HF iAUC | 0.7173 | 0.7158 | 0.8352 | 0.8352 |
| LM iAUC | 0.7127 | 0.7120 | **0.8338** | 0.8339 |
| LF iAUC | 0.6947 | 0.6953 | 0.8243 | 0.8250 |
| Static Cox | 0.5314 | 0.5314 | 0.5316 | 0.5316 |
| HF-LM diff | +0.0045 | +0.0038 | +0.0014 | +0.0013 |
| Wilcoxon p | 0.008 | 0.026 | 0.086 | 0.189 |
| HF/LM wins | 49/31 | 47/33 | 42/38 | 41/39 |
| Event rate | 0.189 | 0.189 | 0.280 | 0.280 |
| PF r | 0.978 | 0.980 | 0.993 | 0.994 |
| Cox βη (true=0.8) | 0.777 | 0.779 | 0.788 | 0.790 |
| medESS | 53.7 | 54.4 | 53.2 | 53.8 |

> **Note:** All values are `round(mean, 4)` computed directly from raw JSON files (zero manual transcription). The LM nonlinear value was previously reported as 0.8339 due to a transcription error; the correct value is 0.8338.

**Answers to the 4 assessment questions:**

1. **Did HF > LM hold in all 4 scenarios?** — YES in direction (HF always ≥ LM). Nominally significant in 2/4 (baseline p=0.008, heavy_tail p=0.026), borderline in nonlinear (p=0.086), not significant in combined (p=0.189). With Bonferroni correction for 4 comparisons (α=0.0125), only baseline remains significant.

2. **Did the iAUC gap change materially?** — Gap INCREASED from pre-fix ~0.002 to post-fix 0.0045 in baseline. The true landmarking refit made LM slightly worse than the old copy-eta version, which is expected: independent Cox refitting at only 3 landmarks loses information vs. continuous Cox updating.

3. **Did the robustness conclusion still hold?** — Direction (HF ≥ LM) is robust; significance range contracts under nonlinear dynamics. Heavy-tail noise alone barely changes the gap (0.0045→0.0038). Nonlinear dynamics is the primary driver of gap shrinkage (0.0045→0.0014), consistent with a ceiling effect: nonlinear trajectories produce more signal → PF tracking improves (r: 0.978→0.993) → eta quality approaches ceiling → both HF and LM converge. **This is a working hypothesis, not a confirmed finding** — it is based on 4 scenario-level data points; a σ–ω gradient sweep is needed to establish dose-response.

4. **Did event rates change?** — Event rate unchanged within each dynamics type (linear=0.189, nonlinear=0.280). The seed decoupling did not alter event rates because event times use an independent stream (seed+3), decoupled from covariates.

**Decision: Outcome (a) — Direction holds, significance is conditional on dynamics.** The directional finding (HF ≥ LM) is robust across all 4 scenarios. Statistical significance depends on dynamics complexity — an interpretable finding, not a failure. The "strengthened, not overturned" framing from earlier drafts is not accurate for all scenarios: the gap grew under linear dynamics but the significance claim narrowed under nonlinear dynamics. The honest framing is: direction holds in all, significance holds in some, and the pattern is consistent with (but not proven to be) a ceiling effect.

**Important collateral finding (Static Cox panel):** The old Static Cox iAUC of 0.60–0.69, which was presented as "independent validation" of the PF result, was entirely a Bug #1 artifact (seed coupling made age/BMI near-perfect proxies for θ₀). After the fix, Static Cox drops to ~0.53 (near chance) and is flat across all scenarios. This means the old Transparency / Static-Cox evidence panel does not survive the bug fix.

### Phase 5: Commit & Push

```bash
git add -A
git commit -m "Fix #1: decouple DGP seeds; Fix #2: true landmarking refit; Fix #3: remove cost-ratio argmax"
git tag -a v1.1-bugfix -m "Bug fixes: independent DGP seeds, true landmarking benchmark, decision curve sweep"
git push origin master
git push origin v1.1-bugfix
```

### Phase 6: Update Issue on GitHub

Reply to Osakazuha with fix summary and invitation to review.

---

## Long-term Implications

1. **Thesis (Direction A — allostatic load)**: The augmented state-space model (θ(t), A(t)) reuses the same PF infrastructure. The seed fix is essential before starting thesis work — otherwise age/BMI contamination propagates into the new A(t) state.

2. **Publication track**: If results hold after fixes, the paper is stronger. If results change, the paper may need a reframed contribution (e.g., "a framework for stress-testing monitoring pipelines" rather than "HF > LM").

3. **Reproducibility**: The seed fix changes all MC results. Any prior communications (SCD LT, NISS poster) reference old numbers. A clear changelog should accompany the v1.1 tag.

4. **Osakazuha's contribution**: This is a legitimate, high-quality code review. Acknowledging it publicly strengthens your reputation for research integrity. Consider offering co-authorship or acknowledgment in the eventual paper, depending on the depth of their continued involvement.

---

> 本内容由 Coze AI 生成，请遵循相关法律法规及《人工智能生成合成内容标识办法》使用与传播。
