from src.agents.profiler import profiler_node
from src.agents.cleaner import cleaner_node
from src.agents.feature_eng import feature_engineer_node
from src.agents.modeler import modeler_node
from src.agents.critic import critic_node

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
state = {**state, **model_result}

print("\nStep 5: Running Critic...")
critic_result = critic_node(state)

print("\n" + "="*60)
print("CRITIC RESULTS")
print("="*60)
print(f"Severity: {critic_result.get('improvement_suggestions', [])}")
print(f"Should iterate: {critic_result.get('should_iterate', False)}")
print(f"Iteration count: {critic_result.get('iteration_count', 0)}")
print(f"\nSuggestions:")
for i, s in enumerate(critic_result.get('improvement_suggestions', []), 1):
    print(f"  {i}. {s}")
print(f"\nCritique Report Preview:")
print(critic_result.get('critique_report', '')[:500])
