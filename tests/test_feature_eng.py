from src.agents.profiler import profiler_node
from src.agents.cleaner import cleaner_node
from src.agents.feature_eng import feature_engineer_node

initial_state = {
    "dataset_path": "datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "user_goal": "Predict customer churn with high F1-score and low false positive rate",
    "messages": [],
    "iteration_count": 0,
}

print("Step 1: Running Profiler...")
profiler_result = profiler_node(initial_state)
state = {**initial_state, **profiler_result}

print("\nStep 2: Running Cleaner...")
cleaner_result = cleaner_node(state)
state = {**state, **cleaner_result}

print("\nStep 3: Running Feature Engineer...")
feature_result = feature_engineer_node(state)

print("\n\nFINAL RESULTS:")
print(f"Feature engineering success: {bool(feature_result.get('feature_result', ''))}")
print(f"Code length: {len(feature_result.get('feature_code', ''))} chars")
print(f"Output: {feature_result.get('feature_result', '')[:500]}")
