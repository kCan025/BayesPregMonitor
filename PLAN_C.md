---
AIGC:
    Label: "1"
    ContentProducer: 001191110102MACQD9K64018705
    ProduceID: 1329787388049676_0/project_7668539694338130210-files/BayesPregMonitor/PLAN_C.md
    ReservedCode1: ""
    ContentPropagator: 001191110102MACQD9K64028705
    PropagateID: 1329787388049676#1792687628159
    ReservedCode2: ""
---
# Plan C — Cadence-Isolated Comparison (2026-10-22)

## Why the previous benchmarks cannot answer the cadence question

Two successive landmark implementations were each found inadequate:

- **v0 (raw η):** the landmark arm scored subjects directly with particle-filter posterior means, while the high-frequency arm used a multiple-imputation time-varying Cox. The two arms used different scoring models, so the comparison was not apples-to-apples.
- **v1 (per-landmark Cox refit, the `v1.1-bugfix` table in README):** fitting a separate Cox at each landmark created two problems:
  1. **Low EPV.** Each landmark window contains ~8–13 events for 3 predictors (EPV ≈ 3–4), well below the EPV ≥ 10 rule of thumb.
  2. **In-sample optimism.** The Cox is fitted and scored on the same subjects/outcome window. An 80-replicate 2-fold cross-fitting diagnostic showed optimism of +0.09 (linear scenario) and +0.04 (nonlinear); cross-fitted landmark AUC falls below the high-frequency benchmark, reversing the apparent "landmark wins" result.
- **Structural issue (decisive).** Both arms were evaluated at the refresh times (weeks 20/28/32). At those times both the weekly arm and the landmark arm use freshly updated η, so monitoring frequency is **not observed at any evaluation point**. No choice of Cox fitting method can recover an unobserved variable.

## Plan C design

- One scoring model: the high-frequency MI-Cox coefficients are shared by both arms.
- **HF:** risk score `β_η·η(t) + β_X′(X − X̄)` with current-week posterior mean.
- **LM-clean:** identical score, but η is frozen at the most recent landmark (20, 28, 32) and carried forward until the next one — the score literally goes stale between visits.
- Evaluation times are moved **between** landmarks: weeks 22, 24, 26, 30, 34 (δ_t = 4), where HF uses fresh η and LM-clean uses η that is 2–6 weeks old.

Under this design the two arms differ in exactly one thing: the freshness of η with respect to the prediction time.

Implementation: `code/run_monte_carlo.py`, fields `iAUC_hf_planC` and `iAUC_lm_clean`; paired Wilcoxon comparison `HF_vs_LM_clean`. Per-iteration records are saved to `results/monte_carlo_<scenario>_iters_<timestamp>.json`.

## Status

- 4 misspecification scenarios × 80 MC replicates: direction (HF > LM-clean) is consistent and statistically significant in all four. Effect sizes are small (~0.005–0.016 iAUC) and shrink toward zero when particle-filter recovery is near-perfect (nonlinear dynamics, PF–θ correlation ≈ 0.99) — i.e. the value of frequent updating depends on filter/signal quality.
- Exact effect sizes are **not yet finalized**: independent re-computation from `_iters.json` and sensitivity analyses (evaluation grid, δ horizon, landmark spacing) are in progress. Do not cite specific numbers until these complete.

---

> 本内容由 Coze AI 生成，请遵循相关法律法规及《人工智能生成合成内容标识办法》使用与传播。
