from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.state import AgentState

# Import real agents
from src.agents.profiler import profiler_node
from src.agents.cleaner import cleaner_node
from src.agents.feature_eng import feature_engineer_node
from src.agents.modeler import modeler_node
from src.agents.critic import critic_node
from src.agents.deployer import deployer_node

# Routing function for the critic's conditional edge
def route_after_critic(state: AgentState) -> Literal["feature_engineer", "deployer"]:
    if state.get("should_iterate", False) and state.get("iteration_count", 0) < 3:
        print(f"🔄 Critic wants iteration {state.get('iteration_count')} - routing back to feature engineering")
        return "feature_engineer"
    else:
        print("✅ Critic satisfied (or max iterations reached) - routing to deployment")
        return "deployer"

# Build the graph
workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("profiler", profiler_node)
workflow.add_node("cleaner", cleaner_node)
workflow.add_node("feature_engineer", feature_engineer_node)
workflow.add_node("modeler", modeler_node)
workflow.add_node("critic", critic_node)
workflow.add_node("deployer", deployer_node)

# Add edges (linear flow with one conditional)
workflow.add_edge(START, "profiler")
workflow.add_edge("profiler", "cleaner")
workflow.add_edge("cleaner", "feature_engineer")
workflow.add_edge("feature_engineer", "modeler")
workflow.add_edge("modeler", "critic")

# Conditional edge after critic
workflow.add_conditional_edges(
    "critic",
    route_after_critic,
    {"feature_engineer": "feature_engineer", "deployer": "deployer"}
)

workflow.add_edge("deployer", END)

# Compile with memory checkpointer and interrupts
memory = MemorySaver()
graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["cleaner", "feature_engineer", "modeler", "deployer"]
)
