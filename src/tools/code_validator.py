"""
Pre-execution code validator.

Cross-references column names referenced in generated code against
actual CSV headers. Catches hallucinated or stale column names before
wasting a sandbox execution.
"""

from __future__ import annotations

import os
import re
import logging

logger = logging.getLogger(__name__)


def validate_columns_against_csv(
    code: str,
    csv_path: str,
    target_col: str = "",
) -> list[str]:
    """
    Returns a list of warning strings (empty list = all good).
    Each warning names a column referenced in the code that does not
    exist in the CSV at csv_path.
    """
    if not csv_path or not os.path.exists(csv_path):
        return []

    try:
        import pandas as pd
        actual_cols = set(pd.read_csv(csv_path, nrows=0).columns.tolist())
    except Exception as exc:
        logger.debug(f"code_validator: could not read {csv_path}: {exc}")
        return []

    referenced = _extract_column_references(code)

    warnings: list[str] = []
    for col in sorted(referenced):
        # Skip: target col (intentionally referenced for drop), format strings,
        # numeric-only strings, very short strings (likely variables not columns)
        if col == target_col:
            continue
        if "{" in col or col.isdigit() or len(col) < 2:
            continue
        if col not in actual_cols:
            # Suggest close matches to help the fix prompt
            close = [c for c in actual_cols
                     if col.lower() in c.lower() or c.lower() in col.lower()]
            if close:
                hint = f" — did you mean '{close[0]}'?"
            else:
                sample = sorted(actual_cols)[:6]
                hint = f" — available cols include: {sample}"
            warnings.append(f"Column '{col}' not found in CSV{hint}")

    return warnings


def _extract_column_references(code: str) -> set[str]:
    """Extract column name strings from common pandas patterns."""
    refs: set[str] = set()

    # df['col']  df["col"]  X['col']  data['col']
    for m in re.finditer(
        r'''(?:df|X|data|features?|result_df)\s*\[\s*['"]([^'"]{2,80})['"]\s*\]''',
        code,
    ):
        refs.add(m.group(1).strip())

    # .drop('col')  .drop(columns=['col', ...])  .drop(columns='col')
    for m in re.finditer(
        r'''\.drop\s*\(\s*(?:columns\s*=\s*)?['"]([^'"]{2,80})['"]''',
        code,
    ):
        refs.add(m.group(1).strip())

    # .drop(columns=['col1', 'col2'])  — inside the list
    for m in re.finditer(r'''\.drop\s*\(\s*columns\s*=\s*\[([^\]]+)\]''', code):
        for inner in re.finditer(r'''['"]([^'"]{2,80})['"]''', m.group(1)):
            refs.add(inner.group(1).strip())

    # .rename(columns={'old': 'new'})
    for m in re.finditer(r'''\.rename\s*\(\s*columns\s*=\s*\{([^}]+)\}''', code):
        for inner in re.finditer(r'''['"]([^'"]{2,80})['"]''', m.group(1)):
            refs.add(inner.group(1).strip())

    # corr()[col]  — target correlation lookups
    for m in re.finditer(r'''\.corr\(\)\s*\[\s*['"]([^'"]{2,80})['"]\s*\]''', code):
        refs.add(m.group(1).strip())

    return refs
