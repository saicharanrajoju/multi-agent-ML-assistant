"""
experiments/automl_baselines.py
================================
Comparison against AutoML baselines (Phase 3 Validation).
Modified to capture Efficiency (Time) and Robustness (Std Dev).
"""

import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score
from scipy.stats import randint
import lightgbm as lgb

DATASETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = {
    'Titanic': {'file': 'titanic.csv', 'target': 'Survived', 'drop_cols': ['Name', 'Ticket', 'Cabin']},
    'Telco Churn': {'file': 'WA_Fn-UseC_-Telco-Customer-Churn.csv', 'target': 'Churn', 'drop_cols': ['customerID']},
}

def prepare_data(filepath, target_col, drop_cols):
    df = pd.read_csv(filepath)
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Handling edge cases
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    df = df.dropna()

    y = df[target_col]
    X = df.drop(columns=[target_col])

    if y.dtype == 'object':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y))

    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def automl_proxy(X_train, X_test, y_train, y_test):
    # Simulated Auto-Sklearn (Reduced space for speed)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    start_time = time.time()
    
    model = RandomForestClassifier(random_state=42)
    params = {'n_estimators': [50, 100], 'max_depth': [None, 5]}
    search = RandomizedSearchCV(model, params, n_iter=3, cv=3, scoring='f1', random_state=42)
    search.fit(X_train_sc, y_train)
    
    train_time = time.time() - start_time
    cv_std = search.cv_results_['std_test_score'][search.best_index_]
    
    y_pred = search.predict(X_test_sc)
    return f1_score(y_test, y_pred), train_time, cv_std

def tpot_proxy(X_train, X_test, y_train, y_test):
    # TPOT proxy
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    start_time = time.time()
    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model.fit(X_train_sc, y_train)
    
    train_time = time.time() - start_time
    # Simulated CV for TPOT
    cv_scores = cross_val_score(model, X_train_sc, y_train, cv=3, scoring='f1')
    
    y_pred = model.predict(X_test_sc)
    return f1_score(y_test, y_pred), train_time, cv_scores.std()

def our_pipeline(X_train, X_test, y_train, y_test):
    # Multi-Agent Generated Logic
    scaler = StandardScaler()
    
    start_time = time.time()

    numeric = X_train.select_dtypes(include=['int64', 'float64']).columns
    X_train_fe = X_train.copy()
    X_test_fe = X_test.copy()
    
    for col in numeric:
        if X_train_fe[col].skew() > 1.0 and (X_train_fe[col] >= 0).all():
            X_train_fe[f'log_{col}'] = np.log1p(X_train_fe[col])
            X_test_fe[f'log_{col}'] = np.log1p(X_test_fe[col])

    X_train_sc = scaler.fit_transform(X_train_fe)
    X_test_sc = scaler.transform(X_test_fe)

    model = lgb.LGBMClassifier(verbose=-1, random_state=42, num_leaves=31, learning_rate=0.05, n_estimators=100)
    model.fit(X_train_sc, y_train)
    
    train_time = time.time() - start_time
    
    # We estimate our stability via fast CV
    cv_scores = cross_val_score(model, X_train_sc, y_train, cv=3, scoring='f1')
    
    y_pred = model.predict(X_test_sc)
    return f1_score(y_test, y_pred), train_time, cv_scores.std()

if __name__ == "__main__":
    results = []
    print("Running Multi-Agent Validation Benchmarks...")
    for ds_name, config in DATASETS.items():
        filepath = os.path.join(DATASETS_DIR, config['file'])
        if not os.path.exists(filepath):
            continue
        print(f"Processing {ds_name}...")
        
        X_train, X_test, y_train, y_test = prepare_data(filepath, config['target'], config['drop_cols'])
        
        a_f1, a_time, a_std = automl_proxy(X_train, X_test, y_train, y_test)
        t_f1, t_time, t_std = tpot_proxy(X_train, X_test, y_train, y_test)
        o_f1, o_time, o_std = our_pipeline(X_train, X_test, y_train, y_test)
        
        results.append({
            'Dataset': ds_name,
            'AutoML_F1': a_f1, 'AutoML_Time': a_time, 'AutoML_Std': a_std,
            'TPOT_F1': t_f1, 'TPOT_Time': t_time, 'TPOT_Std': t_std,
            'Ours_F1': o_f1, 'Ours_Time': o_time, 'Ours_Std': o_std,
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'validation_metrics.csv'), index=False)
    print("Saved validation metrics to results/validation_metrics.csv")
