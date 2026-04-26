import ast
from typing import List

class LeakageNodeVisitor(ast.NodeVisitor):
    def __init__(self, target_col: str):
        self.target_col = target_col
        self.warnings: List[str] = []
        self.has_train_test_split = False
        self.split_executed_line = -1
        # Collect fit calls first; emit warnings only after full traversal
        # so we know whether a split exists anywhere in the code.
        self._early_fit_calls: List[tuple] = []  # (lineno, method_name)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id == 'train_test_split':
                self.has_train_test_split = True
                self.split_executed_line = node.lineno
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr == 'train_test_split':
                self.has_train_test_split = True
                self.split_executed_line = node.lineno
            elif node.func.attr in ('fit', 'fit_transform'):
                if not self.has_train_test_split:
                    # Record for deferred check — only emit if a split exists elsewhere
                    self._early_fit_calls.append((node.lineno, node.func.attr))
        self.generic_visit(node)

    def finalize(self):
        """Call after visit() to emit deferred fit-before-split warnings.
        Only fires when the code contains a train_test_split (wrong ordering).
        Cleaner/feature-eng have no split by design — fit calls there are valid.
        """
        if self.has_train_test_split:
            for lineno, method in self._early_fit_calls:
                self.warnings.append(
                    f"Line {lineno}: Method '{method}' called before train_test_split. "
                    "This causes data leakage. Fit your preprocessors ONLY on the training data."
                )

    def visit_Assign(self, node):
        # Target leakage detection: df['new_feat'] = df['target'] * 2
        for target in node.targets:
            if self._is_dataframe_assignment_to_col(target):
                col_name = self._get_col_name_from_subscript(target)
                if col_name and col_name != self.target_col:
                    if self._uses_target_col(node.value):
                        self.warnings.append(
                            f"Line {node.lineno}: Feature '{col_name}' is derived using the target column '{self.target_col}'. "
                            "This is strong data leakage."
                        )
        self.generic_visit(node)

    def _is_dataframe_assignment_to_col(self, node):
        # matches things like df['col'] = ...
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                return True
        return False

    def _get_col_name_from_subscript(self, node):
        if isinstance(node.slice, ast.Constant):
            return node.slice.value
        return None

    def _uses_target_col(self, node):
        uses_target = False
        for child in ast.walk(node):
            if isinstance(child, ast.Subscript):
                if isinstance(child.slice, ast.Constant) and child.slice.value == self.target_col:
                    uses_target = True
                    break
        return uses_target


def detect_leakage(code: str, target_col: str, check_split: bool = False) -> List[str]:
    """
    Parses generated code to detect common data leakage anti-patterns.

    Args:
        code: Python source code string to analyze.
        target_col: Name of the target column to watch for leakage.
        check_split: If True, warn when train_test_split is absent.
                     Pass True only for modeler code — feature engineering
                     intentionally has no split.
    """
    if not code or not isinstance(code, str):
        return []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["SyntaxError: Code failed to parse before leakage detection."]

    visitor = LeakageNodeVisitor(target_col)
    visitor.visit(tree)
    visitor.finalize()

    if check_split and not visitor.has_train_test_split:
        visitor.warnings.append(
            "No 'train_test_split' call detected. A proper train/test split is required to avoid leakage."
        )

    return visitor.warnings
