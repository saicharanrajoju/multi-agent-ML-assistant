import sys
from src.graph import graph
from src.state import AgentState
from src.tools.code_executor import SandboxManager

def run_pipeline(dataset_path: str, user_goal: str):
    """Run the full ML pipeline with human-in-the-loop review."""
    
    print("=" * 70)
    print("🚀 MULTI-AGENT ML ASSISTANT")
    print("=" * 70)
    print(f"📁 Dataset: {dataset_path}")
    print(f"🎯 Goal: {user_goal}")
    print("=" * 70)
    
    config = {"configurable": {"thread_id": "pipeline-run-1"}}
    
    initial_state = {
        "dataset_path": dataset_path,
        "user_goal": user_goal,
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
    
    # Start the pipeline - it will run profiler then pause before cleaner
    print("\n📍 Starting pipeline...\n")
    
    for event in graph.stream(initial_state, config, stream_mode="values"):
        current_agent = event.get("current_agent", "")
        if current_agent:
            print(f"\n✅ {current_agent.upper()} completed")
    
    # Now handle interrupt points in a loop
    while True:
        state = graph.get_state(config)
        
        # Check if the graph is finished
        if not state.next:
            print("\n" + "=" * 70)
            print("🎉 PIPELINE COMPLETE!")
            print("=" * 70)
            SandboxManager.reset()
            print_final_summary(state.values)
            break
        
        next_node = state.next[0] if state.next else None
        print(f"\n{'=' * 70}")
        print(f"⏸️  HUMAN REVIEW POINT — Next step: {next_node}")
        print(f"{'=' * 70}")
        
        # Show relevant info based on which node is next
        show_review_info(state.values, next_node)
        
        # Ask for human approval
        while True:
            choice = input("\n👉 Choose action:\n  [a] Approve and continue\n  [s] Skip review and continue\n  [f] Provide feedback\n  [q] Quit pipeline\n\nYour choice: ").strip().lower()
            
            if choice == 'a':
                print(f"\n✅ Approved! Continuing to {next_node}...")
                # Resume the graph with None to continue
                for event in graph.stream(None, config, stream_mode="values"):
                    current_agent = event.get("current_agent", "")
                    if current_agent:
                        print(f"\n✅ {current_agent.upper()} completed")
                break
                
            elif choice == 's':
                print(f"\n⏭️ Skipping review, continuing to {next_node}...")
                for event in graph.stream(None, config, stream_mode="values"):
                    current_agent = event.get("current_agent", "")
                    if current_agent:
                        print(f"\n✅ {current_agent.upper()} completed")
                break
                
            elif choice == 'f':
                feedback = input("\n📝 Enter your feedback: ").strip()
                print(f"📝 Feedback recorded: {feedback}")
                # Update state with feedback and resume
                graph.update_state(config, {"human_feedback": feedback})
                for event in graph.stream(None, config, stream_mode="values"):
                    current_agent = event.get("current_agent", "")
                    if current_agent:
                        print(f"\n✅ {current_agent.upper()} completed")
                break
                
            elif choice == 'q':
                print("\n🛑 Pipeline stopped by user.")
                SandboxManager.reset()
                print_final_summary(state.values)
                return
            else:
                print("Invalid choice. Please enter a, s, f, or q.")


def show_review_info(state: dict, next_node: str):
    """Display relevant information for the human reviewer."""
    
    if next_node == "cleaner":
        print("\n📊 PROFILER RESULTS:")
        report = state.get("profile_report", "No report")
        print(report[:1000] if len(report) > 1000 else report)
        issues = state.get("data_issues", [])
        if issues:
            print(f"\n⚠️ Issues found ({len(issues)}):")
            for issue in issues[:10]:
                print(f"  - {issue}")
    
    elif next_node == "feature_engineer":
        print("\n🧹 CLEANING RESULTS:")
        result = state.get("cleaning_result", "No result")
        print(result[:800] if len(result) > 800 else result)
        print("\n📝 CLEANING CODE:")
        code = state.get("cleaning_code", "No code")
        print(code[:1000] if len(code) > 1000 else code)
    
    elif next_node == "modeler":
        print("\n⚙️ FEATURE ENGINEERING RESULTS:")
        result = state.get("feature_result", "No result")
        print(result[:800] if len(result) > 800 else result)
    
    elif next_node == "deployer":
        print("\n🤖 MODEL TRAINING RESULTS:")
        result = state.get("model_result", "No result")
        print(result[:1000] if len(result) > 1000 else result)
        print("\n🧐 CRITIC REPORT:")
        critique = state.get("critique_report", "No critique")
        print(critique[:800] if len(critique) > 800 else critique)
        suggestions = state.get("improvement_suggestions", [])
        if suggestions:
            print(f"\n💡 Suggestions ({len(suggestions)}):")
            for s in suggestions[:5]:
                print(f"  - {s}")


def print_final_summary(state: dict):
    """Print the final pipeline summary."""
    print("\n📊 FINAL SUMMARY:")
    print(f"  Dataset: {state.get('dataset_path', 'N/A')}")
    print(f"  Goal: {state.get('user_goal', 'N/A')}")
    print(f"  Critique iterations: {state.get('iteration_count', 0)}")
    print(f"  API endpoint: {state.get('api_endpoint', 'Not deployed')}")
    
    print("\n📁 Generated Artifacts (check outputs/ folder):")
    print("  - outputs/01_data_profile.md")
    print("  - outputs/02_cleaning_report.md")
    print("  - outputs/03_feature_engineering_report.md")
    print("  - outputs/04_model_training_report.md")
    print("  - outputs/05_critique_report_iter1.md")
    print("  - outputs/06_deployment_report.md")
    print("  - outputs/deployment/ (FastAPI + Docker files)")


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    goal = sys.argv[2] if len(sys.argv) > 2 else "Predict customer churn with high F1-score and low false positive rate"
    
    run_pipeline(dataset, goal)
