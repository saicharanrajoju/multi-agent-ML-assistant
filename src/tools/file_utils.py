import os
from pathlib import Path
import re
import pandas as pd


# --- Shared LLM output parsing utilities ---

def extract_code_block(text: str) -> str:
    """Extract Python code from markdown code blocks, with fallback to generic blocks."""
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if not match:
        match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_section(text: str, start_header: str, end_header: str = None) -> str:
    """Extract content between ## headers. If end_header is None, reads until next ## or end."""
    if end_header:
        pattern = rf"## {start_header}\s*\n(.*?)(?=## {end_header}|$)"
    else:
        pattern = rf"## {start_header}\s*\n(.*?)(?=## |$)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text


# --- File path utilities ---

def get_dataset_path(filename: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv") -> str:
    """Get the full path to a dataset file."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    datasets_dir = (base_dir / "datasets").resolve()
    path = (datasets_dir / filename).resolve()
    if not str(path).startswith(str(datasets_dir)):
        raise ValueError(f"Path traversal detected: {filename}")
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return str(path)


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


def build_fix_prompt(code: str, error_msg: str, attempt: int) -> str:
    """
    Build a rich retry prompt that gives the LLM full context to self-correct:
    - The exact code it wrote (with line numbers)
    - The exact error and the failing line
    - Error-type-specific defensive coding rules
    """
    # Annotate code with line numbers so the LLM can locate the error
    numbered = "\n".join(f"{i+1:>4}: {line}" for i, line in enumerate(code.splitlines()))

    # Extract the failing line number and content from the traceback
    failing_hint = ""
    line_match = re.search(r"line (\d+)", error_msg)
    if line_match:
        lineno = int(line_match.group(1))
        lines = code.splitlines()
        if 0 < lineno <= len(lines):
            failing_hint = f"\nThe error points to line {lineno}:\n  {lines[lineno - 1].strip()}\n"

    # Error-specific defensive rules
    rules = []
    err_lower = error_msg.lower()
    if "keyerror" in err_lower or "column" in err_lower:
        rules += [
            "Always check if a column exists before accessing it: `if col in df.columns`",
            "When iterating and dropping columns, check both col1 and col2 still exist before every access",
            "After dropping a column inside a loop, do not reference it again in the same iteration",
            "CRITICAL: Never include the target column in numeric_columns or categorical_columns lists passed to ColumnTransformer — X_train does NOT contain the target column",
            "Build feature column lists AFTER calling X = df.drop(target, axis=1): numeric_columns = [c for c in X.columns if X[c].dtype != 'object'] — never use df.columns",
        ]
    if "valueerror" in err_lower and "shape" in err_lower:
        rules += [
            "Ensure X and y have matching row counts after any filtering or dropping",
            "Re-align indices with `.reset_index(drop=True)` after filtering",
        ]
    if "attributeerror" in err_lower:
        rules += [
            "Check the object type before calling methods on it",
            "Avoid chaining operations that may return None",
        ]
    if "indexerror" in err_lower or "out of bounds" in err_lower:
        rules += [
            "Always guard list/array access with a length check",
        ]
    if "unsupported format string" in err_lower or "numpy" in err_lower and "format" in err_lower:
        rules += [
            "When printing a numpy value with :.3f, always extract a Python scalar first: `val = float(array_or_series_element)`",
            "Never use .values[0] directly inside an f-string with a format spec — it returns an array. Use float(corr_matrix.loc[col, other_col]) instead.",
        ]
    if not rules:
        rules = [
            "Add defensive checks (e.g. `if col in df.columns`) before accessing any DataFrame column",
            "Use `.get()` for dict access instead of direct indexing",
        ]

    rules_text = "\n".join(f"  - {r}" for r in rules)

    # Escalate temperature hint for later attempts
    temperature_note = ""
    if attempt >= 2:
        temperature_note = "\nThis is your final attempt. Take a completely different approach if the previous logic keeps failing.\n"

    return f"""Your code failed on attempt {attempt + 1}. Here is the EXACT code you wrote:

```python
{numbered}
```
{failing_hint}
Error:
{error_msg[:1500]}
{temperature_note}
Defensive coding rules you MUST follow in the fix:
{rules_text}

Return ONLY the complete fixed Python code block. Fix the specific bug — do not rewrite unrelated parts.
```python
# fixed code here
```"""


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
