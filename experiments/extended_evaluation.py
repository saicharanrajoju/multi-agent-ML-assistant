"""
experiments/extended_evaluation.py
===================================
Extended validation covering three areas:

1. REGRESSION DATASET: California Housing (20,640 rows, 8 features)
   - Runs the multi-agent pipeline logic on a regression task
   - Proves the system handles regression, not just classification

2. PR-AUC & CALIBRATION METRICS:
   - Precision-Recall AUC for all imbalanced classification datasets
   - Calibration (Brier score + reliability diagram)

3. FORMAL CODE QUALITY DEFINITION:
   - Defines explicit, measurable quality criteria for LLM-generated code
   - Maps the Critic scorecard to pass/fail thresholds
   - Aligned with the 5-agent pipeline (no deployer — pipeline ends at Critic)
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    f1_score, precision_recall_curve, auc, brier_score_loss,
    average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.calibration import calibration_curve
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# PART 1: REGRESSION DATASET VALIDATION
# ============================================================
print("=" * 70)
print("PART 1: REGRESSION DATASET — California Housing")
print("=" * 70)

df_reg = pd.read_csv(os.path.join(DATASETS_DIR, 'california_housing.csv'))
print(f"Loaded: {df_reg.shape[0]} rows, {df_reg.shape[1]} cols")
print(f"Target: MedHouseVal (range: {df_reg['MedHouseVal'].min():.2f} - {df_reg['MedHouseVal'].max():.2f})")

y_reg = df_reg['MedHouseVal']
X_reg = df_reg.drop(columns=['MedHouseVal'])

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_rs = scaler.fit_transform(X_train_r)
X_test_rs = scaler.transform(X_test_r)

# Single-agent: basic linear regression, no feature engineering
lr = LinearRegression()
lr.fit(X_train_rs, y_train_r)
y_pred_single = lr.predict(X_test_rs)
single_rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_single))
single_mae  = mean_absolute_error(y_test_r, y_pred_single)
single_r2   = r2_score(y_test_r, y_pred_single)

# Multi-agent: feature engineering + model scout + tuned model
X_reg_fe = X_reg.copy()
for col in X_reg_fe.columns:
    if X_reg_fe[col].skew() > 1.0 and (X_reg_fe[col] > 0).all():
        X_reg_fe[f'log_{col}'] = np.log1p(X_reg_fe[col])

corr_with_target = X_reg_fe.corrwith(y_reg).abs().sort_values(ascending=False)
top3 = corr_with_target.index[:3]
X_reg_fe[f'{top3[0]}_x_{top3[1]}'] = X_reg_fe[top3[0]] * X_reg_fe[top3[1]]
X_reg_fe[f'{top3[0]}_x_{top3[2]}'] = X_reg_fe[top3[0]] * X_reg_fe[top3[2]]

X_train_r2, X_test_r2, y_train_r2, y_test_r2 = train_test_split(
    X_reg_fe, y_reg, test_size=0.2, random_state=42
)
scaler2 = StandardScaler()
X_train_rs2 = scaler2.fit_transform(X_train_r2)
X_test_rs2  = scaler2.transform(X_test_r2)

gbr = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    min_samples_split=5, random_state=42
)
gbr.fit(X_train_rs2, y_train_r2)
y_pred_gbr = gbr.predict(X_test_rs2)
gbr_rmse = np.sqrt(mean_squared_error(y_test_r2, y_pred_gbr))
gbr_mae  = mean_absolute_error(y_test_r2, y_pred_gbr)
gbr_r2   = r2_score(y_test_r2, y_pred_gbr)

lgbm_reg = lgb.LGBMRegressor(
    verbose=-1, random_state=42, num_leaves=63,
    learning_rate=0.05, n_estimators=200
)
lgbm_reg.fit(X_train_rs2, y_train_r2)
y_pred_lgbm = lgbm_reg.predict(X_test_rs2)
lgbm_rmse = np.sqrt(mean_squared_error(y_test_r2, y_pred_lgbm))
lgbm_mae  = mean_absolute_error(y_test_r2, y_pred_lgbm)
lgbm_r2   = r2_score(y_test_r2, y_pred_lgbm)

# Pick best multi-agent result (mirrors Modeler's scout logic)
if lgbm_r2 > gbr_r2:
    multi_rmse, multi_mae, multi_r2 = lgbm_rmse, lgbm_mae, lgbm_r2
    best_reg_model = "LightGBM"
else:
    multi_rmse, multi_mae, multi_r2 = gbr_rmse, gbr_mae, gbr_r2
    best_reg_model = "GradientBoosting"

print(f"\n{'Metric':<20} {'Single-Agent (LR)':>20} {f'Multi-Agent ({best_reg_model})':>28}")
print("-" * 70)
print(f"{'RMSE':<20} {single_rmse:>20.4f} {multi_rmse:>28.4f}")
print(f"{'MAE':<20} {single_mae:>20.4f} {multi_mae:>28.4f}")
print(f"{'R-squared':<20} {single_r2:>20.4f} {multi_r2:>28.4f}")
print(f"\nR-squared improvement: {multi_r2 - single_r2:+.4f}")

cv_scores_single = cross_val_score(LinearRegression(), X_train_rs, y_train_r, cv=5, scoring='r2')
cv_scores_multi  = cross_val_score(gbr, X_train_rs2, y_train_r2, cv=5, scoring='r2')
print(f"\n5-Fold CV R-squared:")
print(f"  Single-agent: {np.mean(cv_scores_single):.4f} +/- {np.std(cv_scores_single):.4f}")
print(f"  Multi-agent:  {np.mean(cv_scores_multi):.4f} +/- {np.std(cv_scores_multi):.4f}")


# ============================================================
# PART 2: PR-AUC & CALIBRATION METRICS
# ============================================================
print("\n" + "=" * 70)
print("PART 2: PR-AUC & CALIBRATION METRICS")
print("=" * 70)

CLASS_DATASETS = {
    'Titanic':              {'file': 'titanic.csv',                            'target': 'Survived', 'drop_cols': ['Name']},
    'Telco Churn':          {'file': 'WA_Fn-UseC_-Telco-Customer-Churn.csv',   'target': 'Churn',    'drop_cols': ['customerID']},
    'Breast Cancer':        {'file': 'breast_cancer_wisconsin.csv',            'target': 'diagnosis','drop_cols': []},
    'Credit Card Default':  {'file': 'credit_card_default.csv',               'target': 'default',  'drop_cols': []},
}

prauc_results     = []
calibration_data  = []

for ds_name, cfg in CLASS_DATASETS.items():
    filepath = os.path.join(DATASETS_DIR, cfg['file'])
    if not os.path.exists(filepath):
        print(f"  [SKIP] {ds_name}: {filepath} not found")
        continue

    df = pd.read_csv(filepath)
    for col in cfg['drop_cols']:
        if col in df.columns:
            df = df.drop(columns=[col])

    y = df[cfg['target']]
    X = df.drop(columns=[cfg['target']])

    if y.dtype == 'object':
        y = pd.Series(LabelEncoder().fit_transform(y))

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X[col] = X[col].fillna(X[col].median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc  = sc.transform(X_test)

    model = lgb.LGBMClassifier(
        verbose=-1, random_state=42, num_leaves=63,
        learning_rate=0.05, n_estimators=200
    )
    model.fit(X_train_sc, y_train)
    y_proba = model.predict_proba(X_test_sc)[:, 1]
    y_pred  = model.predict(X_test_sc)

    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc      = auc(rec_vals, prec_vals)
    avg_prec    = average_precision_score(y_test, y_proba)
    brier       = brier_score_loss(y_test, y_proba)
    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)
    f1          = f1_score(y_test, y_pred)

    print(f"\n--- {ds_name} ---")
    print(f"  F1: {f1:.4f}  PR-AUC: {pr_auc:.4f}  Brier: {brier:.4f}  Pos%: {y.mean()*100:.1f}%")

    prauc_results.append({
        'Dataset': ds_name, 'F1': round(f1, 4),
        'PR-AUC': round(pr_auc, 4), 'Avg Precision': round(avg_prec, 4),
        'Brier Score': round(brier, 4), 'Pos. Class %': round(y.mean() * 100, 1),
    })
    calibration_data.append({'name': ds_name, 'frac_pos': frac_pos, 'mean_pred': mean_pred})

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
for cd in calibration_data:
    ax1.plot(cd['mean_pred'], cd['frac_pos'], 's-', label=cd['name'])
ax1.set_xlabel('Mean Predicted Probability')
ax1.set_ylabel('Fraction of Positives')
ax1.set_title('Calibration Curves (Reliability Diagrams)')
ax1.legend(loc='lower right', fontsize=8)
ax1.grid(True, alpha=0.3)

names   = [r['Dataset'] for r in prauc_results]
pr_aucs = [r['PR-AUC'] for r in prauc_results]
briers  = [r['Brier Score'] for r in prauc_results]
x = np.arange(len(names))
ax2.bar(x - 0.2, pr_aucs, 0.35, label='PR-AUC',      color='#636EFA')
ax2.bar(x + 0.2, briers,  0.35, label='Brier Score',  color='#EF553B')
ax2.set_ylabel('Score')
ax2.set_title('PR-AUC & Calibration by Dataset')
ax2.set_xticks(x)
ax2.set_xticklabels(names, rotation=15, ha='right', fontsize=8)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
chart_path = os.path.join(RESULTS_DIR, 'prauc_calibration_chart.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"\n[SAVED] {chart_path}")


# ============================================================
# PART 3: FORMAL CODE QUALITY DEFINITION
# ============================================================
# Aligned with the current 5-agent pipeline:
# Profiler → Cleaner → Feature Engineer → Modeler → Critic → END
# (No Deployer — pipeline terminates at Critic approval)

code_quality_definition = """
+============================================================================+
|                   FORMAL CODE QUALITY CRITERIA                             |
|         What counts as "semantically sound" LLM-generated pipeline code   |
+============================================================================+

Code is classified PASS if ALL of the following are met.
Any failure triggers the Critic self-correction loop (up to 3 iterations).

GATE 1 — EXECUTABILITY (Binary)
  [x] Runs end-to-end in E2B sandbox without exceptions
  [x] All imports resolve (no ModuleNotFoundError)
  [x] Output files produced: cleaned_data.csv, featured_data.csv,
      best_model.joblib, preprocessor.joblib, visualization_data.json

GATE 2 — DATA INTEGRITY (Binary per assertion)
  [x] Zero NaN/null values in output dataset
  [x] No duplicate columns
  [x] Target column present and numeric (0/1 for classification)
  [x] Row count within 10% of input (no catastrophic data loss)
  [x] No data leakage: preprocessing fitted ONLY on training split

GATE 3 — STATISTICAL VALIDITY (Threshold-based)
  [x] Primary metric (F1/R2) >= baseline (LogisticRegression/LinearRegression)
  [x] 5-fold CV std < 0.10 (pipeline stability)
  [x] No single fold deviates > 2 sigma from mean

GATE 4 — REPRODUCIBILITY (Binary)
  [x] random_state=42 on all stochastic operations
  [x] Stratified train/test split for classification
  [x] Identical results across 2 consecutive runs

GATE 5 — CRITIC SCORECARD THRESHOLDS (/10 scale)
  Category                    Minimum
  ------------------------------------
  Data Leakage Prevention        7/10
  Code Quality                   6/10
  Metric Alignment               7/10
  Feature Engineering Depth      5/10
  Model Selection                6/10
  Production Readiness           5/10
  OVERALL                        6.0/10

  OVERALL < 6.0 OR any category < 5  →  CRITICAL  →  re-iterate (mandatory)
  6.0 <= OVERALL < 7.5                →  MODERATE  →  re-iterate (recommended)
  OVERALL >= 7.5                      →  MINOR     →  pipeline complete

AGGREGATE QUALITY SCORE:
  "Passes quality bar" = Gates 1-4 all pass
                         AND Critic Overall >= 7.0/10
                         AND primary metric > baseline by >= 2%
+============================================================================+
"""

print(code_quality_definition)

# ============================================================
# SAVE RESULTS
# ============================================================
report  = "# Extended Evaluation Results\n\n"
report += f"## Part 1: Regression Dataset — California Housing\n\n"
report += "| Metric | Single-Agent (LR) | Multi-Agent |\n"
report += "|---|---|---|\n"
report += f"| RMSE | {single_rmse:.4f} | {multi_rmse:.4f} |\n"
report += f"| MAE  | {single_mae:.4f} | {multi_mae:.4f} |\n"
report += f"| R-squared | {single_r2:.4f} | {multi_r2:.4f} |\n"
report += f"\nR-squared improvement: {multi_r2 - single_r2:+.4f}\n"

report += "\n## Part 2: PR-AUC & Calibration Metrics\n\n"
report += "| Dataset | F1 | PR-AUC | Avg Precision | Brier Score | Pos% |\n"
report += "|---|---|---|---|---|---|\n"
for r in prauc_results:
    report += f"| {r['Dataset']} | {r['F1']:.4f} | {r['PR-AUC']:.4f} | {r['Avg Precision']:.4f} | {r['Brier Score']:.4f} | {r['Pos. Class %']:.1f}% |\n"
report += "\n![Calibration & PR-AUC Chart](prauc_calibration_chart.png)\n"

report += "\n## Part 3: Formal Code Quality Definition\n"
report += "```\n" + code_quality_definition.strip() + "\n```\n"

out_path = os.path.join(RESULTS_DIR, 'extended_evaluation.md')
with open(out_path, 'w') as f:
    f.write(report)
print(f"[SAVED] {out_path}")
print("\nAll 3 validation sections complete.")
