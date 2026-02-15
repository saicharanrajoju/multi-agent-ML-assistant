from src.agents.profiler import profiler_node

test_state = {
    "dataset_path": "datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "user_goal": "Predict customer churn with high F1-score and low false positive rate",
    "messages": [],
    "iteration_count": 0,
}

print("Testing Profiler Agent...")
result = profiler_node(test_state)
print("\n\nRESULTS:")
print(f"Issues found: {result['data_issues']}")
try:
    print(f"Columns analyzed: {len(result['column_info'])}")
except:
    print(f"Columns analyzed: {result['column_info']}")
print(f"Report length: {len(result['profile_report'])} chars")
