"""
Code scaffold builders.

Each builder returns (preamble, postamble, llm_instruction).

The scaffold handles all plumbing:
  - imports, data loading, X/y split, column-list building,
    train/test split, final validation, saving

The LLM writes ONLY the business-logic block that goes between them.
This eliminates the whole class of load/save/split/target-leak bugs.
"""

from __future__ import annotations


def _MARKER_START(label: str) -> str:
    return f"# ══ BEGIN {label} ════════════════════════════════════════════"


def _MARKER_END(label: str) -> str:
    return f"# ══ END {label} ══════════════════════════════════════════════"


# ── Cleaning scaffold ─────────────────────────────────────────────────────────

def build_cleaning_scaffold(target_col: str, sandbox_path: str) -> tuple[str, str, str]:
    preamble = f"""import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# SCAFFOLD — do not modify this section
df = pd.read_csv('{sandbox_path}')
print(f"Loaded: {{df.shape[0]}} rows x {{df.shape[1]}} cols")
target_col = '{target_col}'

{_MARKER_START('YOUR CLEANING CODE')}
"""

    postamble = f"""
{_MARKER_END('YOUR CLEANING CODE')}

# SCAFFOLD — final validation & save
df.fillna(df.median(numeric_only=True), inplace=True)
for _col in df.select_dtypes(include='object').columns:
    if _col != target_col:
        _mode = df[_col].mode()
        df[_col].fillna(_mode.iloc[0] if not _mode.empty else 'unknown', inplace=True)

_bool_cols = df.select_dtypes(include='bool').columns.tolist()
if _bool_cols:
    df[_bool_cols] = df[_bool_cols].astype('int8')
    print(f"BOOL→INT8: converted {{len(_bool_cols)}} boolean columns")

_remaining_nulls = df.isnull().sum().sum()
if _remaining_nulls > 0:
    print(f"WARNING: {{_remaining_nulls}} nulls remain — force-filling with 0")
    df.fillna(0, inplace=True)

print(f"FINAL SHAPE: {{df.shape}}")
print(f"TARGET COLUMN: {{target_col}} — unique: {{sorted(df[target_col].unique().tolist())}}, dtype: {{df[target_col].dtype}}")
df.to_csv('/home/user/cleaned_data.csv', index=False)
print(f"SAVED: cleaned_data.csv — shape {{df.shape}}")
"""

    instruction = (
        "Generate ONLY the cleaning logic (steps 2–11 from the system prompt). "
        "Do NOT include: import statements, pd.read_csv(), df.to_csv(), or the final fillna — "
        "the scaffold handles all of that. "
        "The variables `df` and `target_col` are already defined. "
        "Return the inner block inside a single ```python code block."
    )
    return preamble, postamble, instruction


# ── Feature engineering scaffold ─────────────────────────────────────────────

def build_feature_eng_scaffold(
    target_col: str,
    feature_strategies: list,
    actual_columns: list,
) -> tuple[str, str, str]:

    n_input_features = max(len(actual_columns) - 1, 1)

    preamble = f"""import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# SCAFFOLD — do not modify this section
df = pd.read_csv('/home/user/cleaned_data.csv')
print(f"Loaded cleaned_data.csv: {{df.shape}}")

TARGET_COL = '{target_col}'
assert TARGET_COL in df.columns, f"ERROR: target '{{TARGET_COL}}' missing from cleaned data"

# Split BEFORE any feature work — prevents target leakage
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].copy()

# Column-type lists built from X (target excluded by construction — NEVER use df.columns here)
numeric_cols = [c for c in X.columns if X[c].dtype != 'object']
categorical_cols = [c for c in X.columns if X[c].dtype == 'object']

# Profiler's strategy list — already resolved, just use it
feature_strategies = {repr(feature_strategies)}

print(f"Input: {{X.shape[1]}} features | {{len(numeric_cols)}} numeric | {{len(categorical_cols)}} categorical")

{_MARKER_START('YOUR FEATURE ENGINEERING CODE')}
"""

    postamble = f"""
{_MARKER_END('YOUR FEATURE ENGINEERING CODE')}

# SCAFFOLD — validate & save
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))

assert TARGET_COL not in X.columns, f"LEAKAGE: '{{TARGET_COL}}' ended up in feature matrix!"
_nulls = X.isnull().sum().sum()
assert _nulls == 0, f"{{_nulls}} null values remain after feature engineering"
_infs = np.isinf(X.select_dtypes(include='number').values).sum()
assert _infs == 0, f"{{_infs}} infinite values remain"

result_df = X.copy()
result_df[TARGET_COL] = y.values
result_df.to_csv('/home/user/featured_data.csv', index=False)
_new = max(0, X.shape[1] - {n_input_features})
print(f"SAVED: featured_data.csv — shape {{result_df.shape}} — {{_new}} new features added")
"""

    instruction = (
        "Generate ONLY the feature engineering logic block. "
        "Do NOT include: import statements, pd.read_csv(), df.to_csv(), assertions, or variable definitions — "
        "the scaffold handles all of that. "
        f"Pre-defined variables you MUST use: X (DataFrame, no target), y (target Series), "
        f"numeric_cols, categorical_cols, TARGET_COL='{target_col}', feature_strategies (list). "
        "Return the inner block in a single ```python code block."
    )
    return preamble, postamble, instruction


# ── Modeler scaffold ──────────────────────────────────────────────────────────

def build_modeler_scaffold(
    target_col: str,
    problem_type: str,
    top_models: list,
    n_rows: int,
) -> tuple[str, str, str]:

    stratify_arg = ", stratify=y" if "classification" in problem_type else ""

    preamble = f"""import pandas as pd
import numpy as np
import json
import os
import warnings
import joblib
warnings.filterwarnings('ignore')

class NpEncoder(json.JSONEncoder):
    # Custom JSON encoder for NumPy types to prevent serialization crashes.
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, KFold, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             mean_squared_error, mean_absolute_error, r2_score,
                             confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor)

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = LGBMRegressor = None

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
except ImportError:
    ImbPipeline = Pipeline
    SMOTE = None

# SCAFFOLD — do not modify this section
df = pd.read_csv('/home/user/featured_data.csv')
print(f"Loaded featured_data.csv: {{df.shape}}")

TARGET_COL = '{target_col}'
assert TARGET_COL in df.columns, f"ERROR: target '{{TARGET_COL}}' not in featured data"

# Encode any remaining categoricals before split
for _col in df.select_dtypes(include='object').columns:
    if _col != TARGET_COL:
        df[_col] = LabelEncoder().fit_transform(df[_col].astype(str))

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# CRITICAL: build column lists from X — target excluded by construction
# Never pass df.columns to ColumnTransformer
numeric_cols = [c for c in X.columns if X[c].dtype != 'object']
categorical_cols = [c for c in X.columns if X[c].dtype == 'object']
print(f"Features: {{len(numeric_cols)}} numeric | {{len(categorical_cols)}} categorical")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42{stratify_arg})
X_train = X_train.reset_index(drop=True)
X_test  = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)
print(f"Train: {{X_train.shape}} | Test: {{X_test.shape}}")

# Pre-resolved decisions
TOP_MODELS   = {repr(top_models)}
N_ROWS       = {n_rows}
PROBLEM_TYPE = '{problem_type}'
TARGET_COL_NAME = '{target_col}'

{_MARKER_START('YOUR MODELING CODE')}
"""

    postamble = f"""
{_MARKER_END('YOUR MODELING CODE')}
"""

    instruction = (
        "Generate ONLY the modeling logic block (preprocessing pipeline, training, evaluation, "
        "tuning, threshold optimisation, CV, feature importance, saving artifacts). "
        "Do NOT include: import statements, pd.read_csv(), train_test_split(), or variable "
        "definitions — the scaffold handles all of that. "
        f"Pre-defined: X_train, X_test, y_train, y_test, X, y, "
        f"numeric_cols, categorical_cols, TARGET_COL='{target_col}', "
        f"TOP_MODELS={top_models}, N_ROWS={n_rows}, PROBLEM_TYPE='{problem_type}', "
        "and all sklearn/xgb/lgbm classes already imported. "
        "Return the inner block in a single ```python code block."
    )
    return preamble, postamble, instruction


# ── Assembly helpers ──────────────────────────────────────────────────────────

def assemble(preamble: str, llm_block: str, postamble: str) -> str:
    """Combine scaffold + LLM inner block + scaffold postamble."""
    return preamble + "\n" + llm_block.strip() + "\n" + postamble


def extract_inner_block(llm_code: str, label: str) -> str:
    """
    If the LLM returned a complete script despite instructions, extract only
    the inner block between the scaffold markers.  Falls back to full code.
    """
    start_marker = _MARKER_START(label)
    end_marker   = _MARKER_END(label)

    if start_marker in llm_code and end_marker in llm_code:
        start = llm_code.index(start_marker) + len(start_marker)
        end   = llm_code.index(end_marker)
        return llm_code[start:end].strip()

    # LLM returned only the inner block (correct behaviour) — return as-is
    return llm_code.strip()
