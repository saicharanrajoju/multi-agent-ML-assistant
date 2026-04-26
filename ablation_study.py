import os
import json
import re
from langgraph.graph import StateGraph, START, END
# Note: we import workflow directly to reuse its node definitions
from src.graph import workflow, memory
from src.state import AgentState
from src.agents.profiler import validated_profiler_node
from src.agents.cleaner import cleaner_node
from src.agents.feature_eng import feature_engineer_node
from src.agents.modeler import modeler_node

def extract_metrics(state: dict) -> dict:
    """Extract model metrics from state's visualization_data or fallback to regex on model_result."""
    metrics = {
        "best_model": "-",
        "F1-score": "-",
        "Precision": "-",
        "Recall": "-",
        "Accuracy": "-",
        "AUC-ROC": "-"
    }
    
    # Try from visualization_data first
    viz_data = state.get("visualization_data", {})
    if isinstance(viz_data, dict):
        if "best_model" in viz_data:
            metrics["best_model"] = viz_data["best_model"]
        elif "model_name" in viz_data:
            metrics["best_model"] = viz_data["model_name"]
            
        metrics_dict = viz_data.get("metrics", viz_data)
        if isinstance(metrics_dict, dict):
            for k, v in metrics_dict.items():
                if isinstance(v, (int, float)):
                    k_lower = k.lower()
                    if "f1" in k_lower: metrics["F1-score"] = round(float(v), 4)
                    elif "prec" in k_lower: metrics["Precision"] = round(float(v), 4)
                    elif "rec" in k_lower: metrics["Recall"] = round(float(v), 4)
                    elif "acc" in k_lower: metrics["Accuracy"] = round(float(v), 4)
                    elif "auc" in k_lower or "roc" in k_lower: metrics["AUC-ROC"] = round(float(v), 4)

    # Fallback to model_result via regex
    model_result = str(state.get("model_result", ""))
    
    if metrics["best_model"] == "-":
        bm_match = re.search(r'(?i)(?:best model|selected model|chosen model|model chosen)(?:\s+is|[:=])?\s*([\w\s\-]+?)(?:\s*-|\n|$)', model_result)
        if bm_match:
            metrics["best_model"] = bm_match.group(1).strip()
            
    for m in ["F1-score", "Precision", "Recall", "Accuracy", "AUC-ROC"]:
        if metrics[m] == "-":
            pattern = m
            if m == "F1-score": pattern = r"f1[\s\-]?score|f1"
            elif m == "AUC-ROC": pattern = r"auc[\s\-]?roc|roc[\s\-]?auc|auc"
            
            match = re.search(fr'(?i)({pattern})[^\d]*?([0-9]*\.[0-9]+)', model_result)
            if match:
                metrics[m] = round(float(match.group(2)), 4)
                
    return metrics

def run_pipeline(dataset_path: str, goal: str, with_critic: bool, thread_id: str) -> dict:
    print(f"\n{'='*80}")
    print(f"Running pipeline on {dataset_path}")
    print(f"Goal: {goal}")
    print(f"With Critic: {with_critic}")
    print(f"Thread ID: {thread_id}")
    print(f"{'='*80}")
    
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "dataset_path": dataset_path,
        "user_goal": goal,
        "messages": [],
        "iteration_count": 0,
        "should_iterate": False,
        "cleaning_approved": False,
        "feature_approved": False,
        "model_approved": False,
        "human_feedback": "",
        "error": "",
        "problem_type": "binary_classification",
        "recommended_metric": "f1",
        "reasoning_context": {},
    }
    
    if with_critic:
        # Patch interrupt mechanism by recompiling WITHOUT interrupt_before
        graph = workflow.compile(checkpointer=memory)
    else:
        # Re-build graph, replacing critic with a passthrough
        custom_workflow = StateGraph(AgentState)
        custom_workflow.add_node("profiler", validated_profiler_node)
        custom_workflow.add_node("cleaner", cleaner_node)
        custom_workflow.add_node("feature_engineer", feature_engineer_node)
        custom_workflow.add_node("modeler", modeler_node)
        
        def passthrough_critic(state: AgentState):
            return {"should_iterate": False, "iteration_count": 0}
            
        custom_workflow.add_node("critic", passthrough_critic)

        custom_workflow.add_edge(START, "profiler")
        custom_workflow.add_edge("profiler", "cleaner")
        custom_workflow.add_edge("cleaner", "feature_engineer")
        custom_workflow.add_edge("feature_engineer", "modeler")
        custom_workflow.add_edge("modeler", "critic")
        custom_workflow.add_edge("critic", END)
        
        # Compile WITHOUT interrupt_before to auto-approve
        graph = custom_workflow.compile(checkpointer=memory)

    # Execute graph completely (will not pause because we removed the interrupts)
    final_state = None
    try:
        for evt in graph.stream(initial_state, config, stream_mode="values"):
            final_state = evt
            agent = evt.get("current_agent", "")
            if agent:
                print(f"✅ {agent} completed")
    except Exception as e:
        print(f"Pipeline error: {e}")
        # Make sure we still grab the latest state if it failed
        final_state = graph.get_state(config).values

    return final_state

def main():
    results = {}
    
    tasks = [
        {
            "name": "Telco_Churn",
            "dataset": "datasets/telco_churn.csv",
            "goal": "Predict Churn with high F1-score",
            "thread_with": "ablation-with-critic-telco",
            "thread_no": "ablation-no-critic-telco"
        },
        {
            "name": "Titanic",
            "dataset": "datasets/titanic.csv",
            "goal": "Predict Survived with high recall",
            "thread_with": "ablation-with-critic-titanic",
            "thread_no": "ablation-no-critic-titanic"
        }
    ]
    
    for task in tasks:
        print(f"\n\n🚀 Starting tasks for {task['name']}...")
        
        # 1. Run WITH Critic
        state_with = run_pipeline(
            dataset_path=task["dataset"], 
            goal=task["goal"], 
            with_critic=True, 
            thread_id=task["thread_with"]
        )
        metrics_with = extract_metrics(state_with)
        iterations_with = state_with.get("iteration_count", 0)
        
        # Safely extract critic scorecard — key is "overall", not "overall_score"
        scorecard = state_with.get("scorecard", {})
        # Key is "overall" to match critic.py scorecard output
        overall_score = scorecard.get("overall", None) if isinstance(scorecard, dict) else None
        
        # 2. Run WITHOUT Critic
        state_without = run_pipeline(
            dataset_path=task["dataset"], 
            goal=task["goal"], 
            with_critic=False, 
            thread_id=task["thread_no"]
        )
        metrics_without = extract_metrics(state_without)
        
        # 3. Store result
        results[task["name"]] = {
            "with": {
                "metrics": metrics_with,
                "iterations": iterations_with,
                "scorecard": overall_score
            },
            "without": {
                "metrics": metrics_without,
                "iterations": 0,
                "scorecard": None
            }
        }
        
    os.makedirs("outputs", exist_ok=True)
    
    # Save raw JSON results
    with open("outputs/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Build cleanly formatted MarkDown string
    md_content = "# Ablation Study Results\n\n"
    
    for task_name, data in results.items():
        md_content += f"## Task: {task_name}\n\n"
        md_content += "Metric | With Critic | Without Critic | Delta\n"
        md_content += "---|---|---|---\n"
        
        w_data = data["with"]
        wo_data = data["without"]
        
        # Rows
        bm_with = w_data["metrics"].get("best_model", "-")
        bm_wo = wo_data["metrics"].get("best_model", "-")
        md_content += f"**Best Model** | {bm_with} | {bm_wo} | -\n"
        
        for metric in ["F1-score", "Precision", "Recall", "Accuracy", "AUC-ROC"]:
            val_with = w_data["metrics"].get(metric, "-")
            val_wo = wo_data["metrics"].get(metric, "-")
            
            delta = "-"
            if val_with != "-" and val_wo != "-":
                try:
                    diff = float(val_with) - float(val_wo)
                    delta = f"{diff:+.4f}"
                except ValueError:
                    pass
                
            md_content += f"**{metric}** | {val_with} | {val_wo} | {delta}\n"
            
        md_content += f"**Critic Iterations** | {w_data['iterations']} | 0 | -\n"
        md_content += f"**Overall Score** | {w_data['scorecard']} | None | -\n\n"
        
    # Print the markdown table to console
    print("\n" + md_content)
    
    # Save the markdown results
    with open("outputs/ablation_results.md", "w") as f:
        f.write(md_content)
        
    print("✅ Results saved to outputs/ablation_results.md and outputs/ablation_results.json")

if __name__ == "__main__":
    main()
