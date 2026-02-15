"""
Test the pipeline on multiple datasets to prove generalizability.
Tests Telco Churn and Titanic datasets.
"""
import sys
import os
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph import graph
from src.tools.code_executor import SandboxManager

DATASETS = [
    {
        "name": "Telco Customer Churn",
        "path": "datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "goal": "Predict customer churn with high F1-score and low false positive rate",
    },
    {
        "name": "Titanic Survival",
        "path": "datasets/titanic.csv",
        "goal": "Predict passenger survival with high recall to catch all survivors",
    },
]

results = {}

def run_test():
    print("🚀 STARTING MULTI-DATASET TEST")
    
    for dataset_info in DATASETS:
        print("\n" + "=" * 70)
        print(f"🧪 TESTING: {dataset_info['name']}")
        print("=" * 70)
        
        # Check if dataset exists
        if not os.path.exists(dataset_info["path"]):
            print(f"⚠️ Dataset not found: {dataset_info['path']} — SKIPPING")
            results[dataset_info["name"]] = {"status": "SKIPPED - file not found"}
            continue
        
        config = {"configurable": {"thread_id": f"test-{dataset_info['name'].lower().replace(' ', '-')}"}}
        
        initial_state = {
            "dataset_path": dataset_info["path"],
            "user_goal": dataset_info["goal"],
            "messages": [],
            "iteration_count": 0,
            "should_iterate": False,
            "cleaning_approved": False,
            "feature_approved": False,
            "model_approved": False,
            "deployment_approved": False,
            "human_feedback": "",
            "error": "",
            "visualization_data": {},
            "scorecard": {},
            "code_fixes": [],
            "iteration_history": [],
            "dataset_summary": {},
            "cleaning_summary": {},
            "target_column": "",
        }
        
        try:
            # Run until first interrupt
            print(f"  ▶️ Starting pipeline for {dataset_info['name']}...")
            for event in graph.stream(initial_state, config, stream_mode="values"):
                agent = event.get("current_agent", "")
                if agent:
                    print(f"  ✅ {agent} completed")
            
            # Auto-approve through all interrupts
            max_steps = 25 
            step = 0
            while step < max_steps:
                state = graph.get_state(config)
                if not state.next:
                    print("  🏁 Pipeline finished (no next steps)")
                    break
                
                next_node = state.next[0]
                print(f"  ⏭️ Auto-approving: {next_node}")
                
                # Update state to approve current step
                current_values = state.values
                updates = {}
                if next_node == "cleaner":
                    updates = {"cleaning_approved": True}
                elif next_node == "feature_engineer":
                    updates = {"feature_approved": True}
                elif next_node == "modeler":
                    updates = {"model_approved": True}
                elif next_node == "deployer":
                    updates = {"deployment_approved": True}
                
                if updates:
                    graph.update_state(config, updates)
                
                # Resume execution
                for event in graph.stream(None, config, stream_mode="values"):
                    agent = event.get("current_agent", "")
                    if agent:
                        print(f"  ✅ {agent} completed")
                step += 1
            
            final_state = graph.get_state(config)
            model_result = final_state.values.get("model_result", "No results")
            target = final_state.values.get("target_column", "Unknown")
            iterations = final_state.values.get("iteration_count", 0)
            error = final_state.values.get("error", "")
            
            if error:
                results[dataset_info["name"]] = {
                    "status": "FAILED (Pipeline Error)",
                    "error": error,
                }
                print(f"\n  ❌ {dataset_info['name']}: FAILED with error: {error}")
            else:
                results[dataset_info["name"]] = {
                    "status": "SUCCESS",
                    "target_column": target,
                    "iterations": iterations,
                    "model_result_preview": str(model_result)[:200] + "...",
                }
                print(f"\n  🎉 {dataset_info['name']}: SUCCESS")
            
        except Exception as e:
            results[dataset_info["name"]] = {
                "status": "FAILED (Exception)",
                "error": str(e)[:300],
            }
            import traceback
            traceback.print_exc()
            print(f"\n  ❌ {dataset_info['name']}: FAILED — {str(e)[:200]}")
        
        finally:
            # Reset sandbox between datasets
            SandboxManager.reset()

    # Final summary
    print("\n" + "=" * 70)
    print("📊 MULTI-DATASET TEST RESULTS")
    print("=" * 70)
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Status: {result.get('status', 'Unknown')}")
        if 'SUCCESS' in result.get('status', ''):
            print(f"  Target: {result.get('target_column')}")
            print(f"  Iterations: {result.get('iterations')}")
        elif 'error' in result:
             print(f"  Error: {result.get('error')}")

if __name__ == "__main__":
    run_test()
