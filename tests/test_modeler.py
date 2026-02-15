from src.agents.profiler import profiler_node
from src.agents.cleaner import cleaner_node
from src.agents.feature_eng import feature_engineer_node
from src.agents.modeler import modeler_node

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
state = {**state, **feature_result}

print("\nStep 4: Running Model Trainer...")
model_result = modeler_node(state)

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Model training success: {'FAILED' not in model_result.get('model_result', 'FAILED')}")
print(f"Code length: {len(model_result.get('model_code', ''))} chars")
print(f"\nModel Results:\n{model_result.get('model_result', 'No results')[:1000]}")
