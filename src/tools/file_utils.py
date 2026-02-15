import os
import json
import pandas as pd

def get_dataset_path(filename: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv") -> str:
    """Get the full path to a dataset file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(base_dir, "datasets", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return path

def get_output_path(filename: str) -> str:
    """Get a path in the outputs directory, creating it if needed."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)

def save_code_to_file(code: str, filename: str) -> str:
    """Save generated code to the outputs directory."""
    path = get_output_path(filename)
    with open(path, "w") as f:
        f.write(code)
    print(f"💾 Code saved to {path}")
    return path

def save_report(report: str, filename: str) -> str:
    """Save a markdown report to the outputs directory."""
    path = get_output_path(filename)
    with open(path, "w") as f:
        f.write(report)
    print(f"📄 Report saved to {path}")
    return path

def load_dataset_preview(path: str, n_rows: int = 5) -> str:
    """Load a CSV and return a text preview for the LLM."""
    df = pd.read_csv(path)
    preview = f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n"
    preview += f"Columns: {list(df.columns)}\n\n"
    preview += f"Data Types:\n{df.dtypes.to_string()}\n\n"
    preview += f"First {n_rows} rows:\n{df.head(n_rows).to_string()}\n\n"
    preview += f"Missing Values:\n{df.isnull().sum().to_string()}\n\n"
    preview += f"Basic Stats:\n{df.describe().to_string()}\n"
    return preview
