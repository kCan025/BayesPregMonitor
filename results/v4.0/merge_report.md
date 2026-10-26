# v4.0 N=2000 4% Monte-Carlo — merge & independent audit


## baseline  (n=80, missing=0)

**HF vs LM-clean [Plan C]**: n=80, wins H/L/t = 62/18/0, mean Δ=+0.0097, Wilcoxon p=8.402150866837891e-07

event rate mean=0.0407; PF corr=0.9785; medESS=53.7; Cox β_eta=0.7715


| subgroup | pooled rate | per-rep mean±sd | events/rep | n/rep |
|---|---|---|---|---|
| overall | 4.07% | 4.07±0.41% | 81.5 [61, 102] | 2000 |
| age_ge33 | 5.58% | 5.58±1.25% | 17.9 [9, 32] | 320 |
| age_lt33 | 3.79% | 3.79±0.42% | 63.6 [49, 77] | 1680 |
| prior_pe_yes | 9.82% | 9.83±2.13% | 19.8 [9, 31] | 201 |
| prior_pe_no | 3.43% | 3.43±0.44% | 61.7 [43, 86] | 1799 |

| window t | cases mean | cases min–max | controls mean | reps skipped (<5) |
|---|---|---|---|---|
| t22 | 9.00 | [3, 19] | 1954 | 3 |
| t24 | 9.49 | [3, 21] | 1949 | 4 |
| t26 | 9.07 | [5, 23] | 1945 | 0 |
| t30 | 10.29 | [2, 22] | 1935 | 2 |
| t34 | 10.46 | [4, 18] | 1924 | 2 |

reps with ≥1 window skipped: 11; reps ALL skipped (iAUC NaN): 0; valid-windows/rep dist: {0: 0, 1: 0, 2: 0, 3: 0, 4: 11, 5: 69}


## nonlinear  (n=80, missing=0)

**HF vs LM-clean [Plan C]**: n=80, wins H/L/t = 59/21/0, mean Δ=+0.0051, Wilcoxon p=2.1626251342223988e-07

event rate mean=0.0414; PF corr=0.9934; medESS=53.2; Cox β_eta=0.7735


| subgroup | pooled rate | per-rep mean±sd | events/rep | n/rep |
|---|---|---|---|---|
| overall | 4.14% | 4.14±0.42% | 82.8 [64, 100] | 2000 |
| age_ge33 | 5.72% | 5.71±1.21% | 18.3 [9, 31] | 320 |
| age_lt33 | 3.84% | 3.84±0.44% | 64.5 [48, 81] | 1680 |
| prior_pe_yes | 9.84% | 9.85±1.90% | 19.8 [11, 29] | 201 |
| prior_pe_no | 3.50% | 3.50±0.43% | 63.0 [48, 79] | 1799 |

| window t | cases mean | cases min–max | controls mean | reps skipped (<5) |
|---|---|---|---|---|
| t22 | 9.38 | [4, 19] | 1963 | 2 |
| t24 | 9.71 | [4, 18] | 1958 | 1 |
| t26 | 11.11 | [4, 20] | 1952 | 3 |
| t30 | 12.71 | [5, 27] | 1939 | 0 |
| t34 | 14.46 | [6, 23] | 1925 | 0 |

reps with ≥1 window skipped: 5; reps ALL skipped (iAUC NaN): 0; valid-windows/rep dist: {0: 0, 1: 0, 2: 0, 3: 1, 4: 4, 5: 75}


## heavy_tail  (n=80, missing=0)

**HF vs LM-clean [Plan C]**: n=80, wins H/L/t = 63/17/0, mean Δ=+0.0101, Wilcoxon p=8.159143618471062e-09

event rate mean=0.0407; PF corr=0.9797; medESS=54.4; Cox β_eta=0.7690


| subgroup | pooled rate | per-rep mean±sd | events/rep | n/rep |
|---|---|---|---|---|
| overall | 4.07% | 4.07±0.41% | 81.5 [61, 102] | 2000 |
| age_ge33 | 5.58% | 5.58±1.25% | 17.9 [9, 32] | 320 |
| age_lt33 | 3.79% | 3.79±0.42% | 63.6 [49, 77] | 1680 |
| prior_pe_yes | 9.82% | 9.83±2.13% | 19.8 [9, 31] | 201 |
| prior_pe_no | 3.43% | 3.43±0.44% | 61.7 [43, 86] | 1799 |

| window t | cases mean | cases min–max | controls mean | reps skipped (<5) |
|---|---|---|---|---|
| t22 | 9.00 | [3, 19] | 1954 | 3 |
| t24 | 9.49 | [3, 21] | 1949 | 4 |
| t26 | 9.07 | [5, 23] | 1945 | 0 |
| t30 | 10.29 | [2, 22] | 1935 | 2 |
| t34 | 10.46 | [4, 18] | 1924 | 2 |

reps with ≥1 window skipped: 11; reps ALL skipped (iAUC NaN): 0; valid-windows/rep dist: {0: 0, 1: 0, 2: 0, 3: 0, 4: 11, 5: 69}


## combined  (n=80, missing=0)

**HF vs LM-clean [Plan C]**: n=80, wins H/L/t = 59/21/0, mean Δ=+0.0044, Wilcoxon p=2.051089423109458e-06

event rate mean=0.0414; PF corr=0.9938; medESS=53.9; Cox β_eta=0.7732


| subgroup | pooled rate | per-rep mean±sd | events/rep | n/rep |
|---|---|---|---|---|
| overall | 4.14% | 4.14±0.42% | 82.8 [64, 100] | 2000 |
| age_ge33 | 5.72% | 5.71±1.21% | 18.3 [9, 31] | 320 |
| age_lt33 | 3.84% | 3.84±0.44% | 64.5 [48, 81] | 1680 |
| prior_pe_yes | 9.84% | 9.85±1.90% | 19.8 [11, 29] | 201 |
| prior_pe_no | 3.50% | 3.50±0.43% | 63.0 [48, 79] | 1799 |

| window t | cases mean | cases min–max | controls mean | reps skipped (<5) |
|---|---|---|---|---|
| t22 | 9.38 | [4, 19] | 1963 | 2 |
| t24 | 9.71 | [4, 18] | 1958 | 1 |
| t26 | 11.11 | [4, 20] | 1952 | 3 |
| t30 | 12.71 | [5, 27] | 1939 | 0 |
| t34 | 14.46 | [6, 23] | 1925 | 0 |

reps with ≥1 window skipped: 5; reps ALL skipped (iAUC NaN): 0; valid-windows/rep dist: {0: 0, 1: 0, 2: 0, 3: 1, 4: 4, 5: 75}
