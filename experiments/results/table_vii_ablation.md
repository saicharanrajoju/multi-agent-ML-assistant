# Ablation Study: Pipeline Component Contributions (Table VII)

| Configuration | F1 (mean +/- 95% CI) | p-val |
|---|---|---|
| Baseline (raw, no pipeline) | 0.724 +/- 0.034 | --- |
| + Cleaner Agent | 0.732 +/- 0.032 | 0.033 |
| + Feature Engineer | 0.713 +/- 0.028 | 0.143 |
| + Pre-Screening Mechanism | 0.750 +/- 0.010 | 0.068 |
| + Critic Self-Correction (1 iter.) | 0.754 +/- 0.007 | 0.541 |
| + HITL Feedback Injection | 0.754 +/- 0.012 | 0.959 |

**Cumulative gain: +0.030**
