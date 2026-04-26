from src.graph import graph
from src.tools.code_executor import SandboxManager

config = {"configurable": {"thread_id": "test-full-pipeline"}}

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
    "should_iterate": False,
    "cleaning_approved": False,
    "feature_approved": False,
    "model_approved": False,
    "deployment_approved": False,
    "human_feedback": "",
    "error": "",
}

print("=" * 70)
print("🧪 FULL PIPELINE TEST (Auto-approve all steps)")
print("=" * 70)

# Run until first interrupt
for event in graph.stream(initial_state, config, stream_mode="values"):
    agent = event.get("current_agent", "")
    if agent:
        print(f"✅ {agent} completed")

# Keep resuming through all interrupts
max_steps = 20  # safety limit
step = 0
while step < max_steps:
    state = graph.get_state(config)
    if not state.next:
        print("\n🎉 Pipeline finished!")
        break
    
    next_node = state.next[0]
    print(f"\n⏭️ Auto-approving and running: {next_node}")
    
    for event in graph.stream(None, config, stream_mode="values"):
        agent = event.get("current_agent", "")
        if agent:
            print(f"✅ {agent} completed")
    
    step += 1

# Print final state
final_state = graph.get_state(config)
print("\n" + "=" * 70)
print("FINAL STATE:")
print(f"  Iterations: {final_state.values.get('iteration_count', 0)}")
print(f"  Last agent: {final_state.values.get('current_agent', 'unknown')}")
print(f"  API endpoint: {final_state.values.get('api_endpoint', 'N/A')}")
print(f"  Has dataset_summary: {'dataset_summary' in final_state.values}")
print(f"  Has cleaning_summary: {'cleaning_summary' in final_state.values}")
print("=" * 70)

# Clean up shared sandbox
SandboxManager.reset()
