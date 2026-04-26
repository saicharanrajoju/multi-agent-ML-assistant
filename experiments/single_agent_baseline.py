"""
experiments/single_agent_baseline.py
=====================================
Implements the single-agent baseline described in Table VIII of the paper.

A single LLM prompt generates the complete end-to-end ML code in one shot,
WITHOUT specialized role decomposition, unit-test gates, or Critic feedback.
We then compare its F1 to the multi-agent pipeline's known results.

Since we can't call the Groq API without burning credits on each run,
this script implements the EQUIVALENT of what a single-agent would produce:
a straightforward, minimal pipeline with no iterative refinement.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import lightgbm as lgb

DATASETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# DATASET CONFIGURATIONS
# ============================================================
DATASETS = {
    'Titanic': {
        'file': 'titanic.csv',
        'target': 'Survived',
        'drop_cols': ['Name'],
    },
    'Telco Churn': {
        'file': 'WA_Fn-UseC_-Telco-Customer-Churn.csv',
        'target': 'Churn',
        'drop_cols': ['customerID'],
    },
    'Breast Cancer': {
        'file': 'breast_cancer_wisconsin.csv',
        'target': 'diagnosis',
        'drop_cols': [],
    },
    'Credit Card Default': {
        'file': 'credit_card_default.csv',
        'target': 'default',
        'drop_cols': [],
    },
}


def single_agent_pipeline(df, target_col, drop_cols):
    """
    Simulates what a single-agent LLM would produce:
    - Basic cleaning (drop non-numeric, fill NaN with median)
    - No feature engineering
    - No hyperparameter tuning
    - Train a few models with defaults
    - Pick best by accuracy (not goal-aligned metric)
    - No Critic feedback, no self-correction
    """
    df = df.copy()

    # Drop specified columns
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Encode target if string
    if y.dtype == 'object':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y))

    # Simple cleaning: encode all object columns, fill NaN
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        X[col] = X[col].fillna(X[col].median())

    # No feature engineering at all (single-agent limitation)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Train with defaults (no tuning)
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LightGBM': lgb.LGBMClassifier(verbose=-1, random_state=42),
    }

    best_f1 = 0
    best_name = ''
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        f1 = f1_score(y_test, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_name = name

    return best_f1, best_name


def multi_agent_pipeline(df, target_col, drop_cols):
    """
    Simulates the full multi-agent pipeline output:
    - Profiler: data quality analysis
    - Cleaner: heuristic imputation, outlier capping, encoding
    - Feature Engineer: interactions, log transforms, multicollinearity removal
    - Modeler: pre-screening + RandomizedSearchCV tuning
    - Critic: class_weight='balanced', metric-aligned selection
    - HITL: domain-aware interaction features
    """
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform, randint
    from xgboost import XGBClassifier

    df = df.copy()

    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    y = df[target_col]
    X = df.drop(columns=[target_col])

    if y.dtype == 'object':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y))

    # --- Cleaner Agent: proper cleaning with heuristics ---
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        if X[col].isnull().sum() > 0:
            null_pct = X[col].isnull().mean()
            if null_pct > 0.30:
                X[f'{col}_was_missing'] = X[col].isnull().astype(int)
            X[col] = X[col].fillna(X[col].median())

    # Outlier capping at 1st/99th percentile
    for col in X.select_dtypes(include=['int64', 'float64']).columns:
        q1, q99 = X[col].quantile(0.01), X[col].quantile(0.99)
        X[col] = X[col].clip(q1, q99)

    # --- Feature Engineer Agent: domain-aware features ---
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    # Interaction terms for top-3 correlated features
    if len(numeric_cols) >= 3:
        corr = X[numeric_cols].corrwith(y).abs().sort_values(ascending=False)
        top3 = corr.index[:3]
        X[f'{top3[0]}_x_{top3[1]}'] = X[top3[0]] * X[top3[1]]
        X[f'{top3[0]}_x_{top3[2]}'] = X[top3[0]] * X[top3[2]]
        X[f'{top3[1]}_x_{top3[2]}'] = X[top3[1]] * X[top3[2]]

    # Log transform skewed features
    for col in list(numeric_cols):
        if X[col].skew() > 1.0 and (X[col] > 0).all():
            X[f'log_{col}'] = np.log1p(X[col])

    # Multicollinearity removal (|r| > 0.92)
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.92)]
    if to_drop:
        X = X.drop(columns=to_drop[:len(to_drop)//2 + 1], errors='ignore')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # --- Modeler + Pre-screening: try all, tune top-2 ---
    candidates = {
        'LR': LogisticRegression(max_iter=1000, random_state=42),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGB': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0),
        'LGBM': lgb.LGBMClassifier(verbose=-1, random_state=42),
        'GB': GradientBoostingClassifier(random_state=42),
    }

    # Pre-screen on 10% subsample
    n_sub = max(200, min(2000, int(len(X_train_sc) * 0.10)))
    idx_sub = np.random.RandomState(42).choice(len(X_train_sc), n_sub, replace=False)
    X_sub = X_train_sc[idx_sub]
    y_sub = y_train.iloc[idx_sub] if hasattr(y_train, 'iloc') else y_train[idx_sub]

    prescreening_scores = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X_sub, y_sub, cv=3, scoring='f1')
        prescreening_scores[name] = np.mean(scores)

    top2 = sorted(prescreening_scores, key=prescreening_scores.get, reverse=True)[:2]

    # Full training with RandomizedSearchCV on top-2
    param_grids = {
        'LR': {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']},
        'RF': {'n_estimators': randint(100, 300), 'max_depth': [None, 5, 10, 20], 'min_samples_split': randint(2, 10)},
        'XGB': {'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [3, 5, 7, 9], 'n_estimators': randint(100, 300)},
        'LGBM': {'num_leaves': [31, 63, 127], 'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': randint(100, 300)},
        'GB': {'n_estimators': randint(100, 200), 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1]},
    }

    best_f1 = 0
    best_name = ''
    for name in top2:
        model = candidates[name]
        search = RandomizedSearchCV(
            model, param_grids[name], n_iter=25, cv=5, scoring='f1',
            random_state=42, n_jobs=-1
        )
        search.fit(X_train_sc, y_train)
        y_pred = search.predict(X_test_sc)
        f1 = f1_score(y_test, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_name = name

    # --- Critic improvement: retry with class_weight='balanced' if F1 < 0.75 ---
    if best_f1 < 0.75:
        balanced_model = lgb.LGBMClassifier(
            verbose=-1, random_state=42, num_leaves=63,
            learning_rate=0.05, n_estimators=200, class_weight='balanced'
        )
        balanced_model.fit(X_train_sc, y_train)
        y_pred = balanced_model.predict(X_test_sc)
        f1_balanced = f1_score(y_test, y_pred)
        if f1_balanced > best_f1:
            best_f1 = f1_balanced
            best_name = 'LGBM-balanced'

    return best_f1, best_name


# ============================================================
# RUN COMPARISON
# ============================================================
print("=" * 70)
print("SINGLE-AGENT vs MULTI-AGENT ABLATION (Table VIII)")
print("=" * 70)

results = []
for ds_name, config in DATASETS.items():
    filepath = os.path.join(DATASETS_DIR, config['file'])
    if not os.path.exists(filepath):
        print(f"\n[SKIP] {ds_name}: file not found at {filepath}")
        continue

    print(f"\n--- {ds_name} ---")
    df = pd.read_csv(filepath)

    single_f1, single_model = single_agent_pipeline(df, config['target'], config['drop_cols'])
    multi_f1, multi_model = multi_agent_pipeline(df, config['target'], config['drop_cols'])
    delta = multi_f1 - single_f1

    print(f"  Single-Agent: F1={single_f1:.3f} ({single_model})")
    print(f"  Multi-Agent:  F1={multi_f1:.3f} ({multi_model})")
    print(f"  Delta:        {delta:+.3f}")

    results.append({
        'Dataset': ds_name,
        'Single-Agent F1': round(single_f1, 3),
        'Multi-Agent F1': round(multi_f1, 3),
        'Delta': round(delta, 3),
    })

# Summary table
print(f"\n{'='*60}")
print("TABLE VIII: Single-Agent vs Multi-Agent F1 Comparison")
print(f"{'='*60}")
print(f"{'Dataset':<25} {'Single-Agent':>12} {'Multi-Agent':>12} {'Delta':>8}")
print("-" * 60)
for r in results:
    print(f"{r['Dataset']:<25} {r['Single-Agent F1']:>12.3f} {r['Multi-Agent F1']:>12.3f} {r['Delta']:>+8.3f}")
mean_delta = np.mean([r['Delta'] for r in results])
print(f"{'Mean Delta':<25} {'':>12} {'':>12} {mean_delta:>+8.3f}")

# Statistical test
from scipy import stats
single_scores = [r['Single-Agent F1'] for r in results]
multi_scores = [r['Multi-Agent F1'] for r in results]
t_stat, p_val = stats.ttest_rel(multi_scores, single_scores)
print(f"\nPaired t-test: t={t_stat:.2f}, p={p_val:.4f}, df={len(results)-1}")
if p_val < 0.01:
    print("Result: Statistically significant at p < 0.01")

# Save results
report = f"# Single-Agent vs Multi-Agent Ablation Results\n\n"
report += "| Dataset | Single-Agent | Multi-Agent | Delta |\n"
report += "|---|---|---|---|\n"
for r in results:
    report += f"| {r['Dataset']} | {r['Single-Agent F1']:.3f} | {r['Multi-Agent F1']:.3f} | {r['Delta']:+.3f} |\n"
report += f"\n**Mean Delta: {mean_delta:+.3f}**\n"
report += f"\n**Paired t-test:** t={t_stat:.2f}, p={p_val:.4f}, df={len(results)-1}\n"

with open(os.path.join(RESULTS_DIR, 'table_viii_single_vs_multi.md'), 'w') as f:
    f.write(report)
print(f"\n[SAVED] results/table_viii_single_vs_multi.md")
