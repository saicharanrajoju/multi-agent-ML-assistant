from src.agents.deployer import deployer_node

# Use mock state with realistic data from previous runs
mock_state = {
    "dataset_path": "datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "user_goal": "Predict customer churn with high F1-score and low false positive rate",
    "messages": [],
    "iteration_count": 1,
    "model_result": """
Model Comparison:
| Model              | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------------------|----------|-----------|--------|----------|---------|
| LogisticRegression | 0.80     | 0.65      | 0.57   | 0.61     | 0.83    |
| RandomForest       | 0.79     | 0.63      | 0.48   | 0.55     | 0.82    |
| XGBoost            | 0.80     | 0.65      | 0.53   | 0.58     | 0.83    |
| LightGBM           | 0.79     | 0.63      | 0.52   | 0.57     | 0.82    |

Best model: LogisticRegression (selected by F1-score)
Cross-validation F1 (5-fold): 0.59 +/- 0.02
    """,
    "cleaning_code": """
import pandas as pd
df = pd.read_csv('/home/user/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
df = pd.get_dummies(df, drop_first=True)
df.to_csv('/home/user/cleaned_data.csv', index=False)
print(f"Cleaned shape: {df.shape}")
    """,
    "feature_code": """
import pandas as pd
import numpy as np
df = pd.read_csv('/home/user/cleaned_data.csv')
df['charge_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)
df['tenure_squared'] = df['tenure'] ** 2
df.to_csv('/home/user/featured_data.csv', index=False)
print(f"Featured shape: {df.shape}")
    """,
    "model_code": """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
df = pd.read_csv('/home/user/featured_data.csv')
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pipe = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=1000, random_state=42))])
pipe.fit(X_train, y_train)
joblib.dump(pipe, '/home/user/best_model.joblib')
joblib.dump(pipe[:-1], '/home/user/preprocessor.joblib')
print("Model saved")
    """,
    "profile_report": "Telco churn dataset, 7043 rows, 21 columns",
    "data_issues": ["TotalCharges stored as string"],
    "column_info": {},
    "cleaning_result": "Cleaned shape: (7043, 31)",
    "feature_result": "Featured shape: (7043, 33)",
    "critique_report": "Minor issues found",
    "improvement_suggestions": [],
    "should_iterate": False,
}

print("Testing Deployer Agent with mock state...")
result = deployer_node(mock_state)

print("\n" + "="*60)
print("DEPLOYER RESULTS")
print("="*60)
print(f"Success: {'deployment_code' in result}")
print(f"API endpoint: {result.get('api_endpoint', 'N/A')}")
print(f"Output preview: {result.get('deployment_code', '')[:300]}")
