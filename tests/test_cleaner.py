from src.agents.profiler import profiler_node
from src.agents.cleaner import cleaner_node

# First run the profiler to get the state we need
print("Step 1: Running profiler first to get data profile...")
initial_state = {
    "problem_type": "binary_classification",
    "recommended_metric": "f1",
    "reasoning_context": {
        "imbalance_strategy": "class_weight_balanced",
        "recommended_models": ["LogisticRegression", "RandomForestClassifier", "XGBClassifier", "LGBMClassifier"],
    },
    "dataset_path": "datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "user_goal": "Predict customer churn with high F1-score and low false positive rate",
    "messages": [],
    "iteration_count": 0,
}

profiler_result = profiler_node(initial_state)

# Now run the cleaner with the profiler's output
print("\n\nStep 2: Running cleaner with profiler results...")
cleaner_state = {**initial_state, **profiler_result}
cleaner_result = cleaner_node(cleaner_state)

print("\n\nFINAL RESULTS:")
print(f"Cleaning success: {'cleaning_result' in cleaner_result}")
print(f"Code length: {len(cleaner_result.get('cleaning_code', ''))} chars")
print(f"Output preview: {cleaner_result.get('cleaning_result', '')[:300]}")
