from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.state import AgentState

# Import agents
from src.tools.code_executor import close_sandbox_for_run
from src.agents.profiler import profiler_node
from src.agents.cleaner import cleaner_node
from src.agents.feature_eng import feature_engineer_node
from src.agents.modeler import modeler_node
from src.agents.critic import critic_node
def validated_profiler_node(state: AgentState, config=None) -> dict:
    """Wrapper around profiler that validates target_column is detected."""
    result = profiler_node(state, config)
    target = result.get("target_column", "")
    if not target or target == "target":
        raise ValueError(
            "Could not detect target column from the dataset. "
            "Please specify the target column in your goal (e.g., 'predict Churn', 'predict Survived', 'predict income')."
        )
    return result


def critic_node_with_cleanup(state: AgentState, config=None) -> dict:
    """Wraps critic_node and closes the sandbox when the pipeline is finishing."""
    result = critic_node(state, config)
    # Close sandbox only when critic won't iterate further
    will_finish = not result.get("should_iterate", False) or (state.get("iteration_count", 0) + 1) >= 3
    if will_finish:
        run_id = (config or {}).get("configurable", {}).get("thread_id", "default")
        close_sandbox_for_run(run_id)
    return result


# Routing function for the critic's conditional edge
def route_after_critic(state: AgentState) -> Literal["cleaner", "feature_engineer", "modeler", "done"]:
    if not state.get("should_iterate", False) or state.get("iteration_count", 0) >= 3:
        print("✅ Critic satisfied (or max iterations reached) - pipeline complete")
        return "done"

    code_fixes = state.get("code_fixes", [])
    files_with_fixes = {fix.get("file", "").lower() for fix in code_fixes}

    if any("clean" in f for f in files_with_fixes):
        print(f"🔄 Critic found cleaning issues - routing back to cleaner (iteration {state.get('iteration_count')})")
        return "cleaner"
    elif any("feature" in f for f in files_with_fixes):
        print(f"🔄 Critic found feature engineering issues - routing back to feature engineering (iteration {state.get('iteration_count')})")
        return "feature_engineer"
    elif any("model" in f for f in files_with_fixes):
        print(f"🔄 Critic found modeling issues - routing back to modeler (iteration {state.get('iteration_count')})")
        return "modeler"
    else:
        print(f"🔄 Critic wants iteration {state.get('iteration_count')} - routing back to feature engineering")
        return "feature_engineer"


# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("profiler", validated_profiler_node)
workflow.add_node("cleaner", cleaner_node)
workflow.add_node("feature_engineer", feature_engineer_node)
workflow.add_node("modeler", modeler_node)
workflow.add_node("critic", critic_node_with_cleanup)

# Linear flow
workflow.add_edge(START, "profiler")
workflow.add_edge("profiler", "cleaner")
workflow.add_edge("cleaner", "feature_engineer")
workflow.add_edge("feature_engineer", "modeler")
workflow.add_edge("modeler", "critic")

# Conditional edge after critic — routes back for iteration or ends
workflow.add_conditional_edges(
    "critic",
    route_after_critic,
    {"cleaner": "cleaner", "feature_engineer": "feature_engineer", "modeler": "modeler", "done": END}
)

# Compile with memory checkpointer and interrupts
memory = MemorySaver()
graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["cleaner", "feature_engineer", "modeler"]
)
