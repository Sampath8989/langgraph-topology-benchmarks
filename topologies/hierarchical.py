from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel

from agents.state import AgentState
from agents.research_agent import ResearchAgent
from agents.planning_agent import PlanningAgent
from agents.hierarchical_supervisor import HierarchicalSupervisor


def build_hierarchical_graph(llm: Optional[BaseChatModel] = None):
    supervisor = HierarchicalSupervisor(llm)
    researcher = ResearchAgent(llm)
    planner = PlanningAgent(llm)

    workflow = StateGraph(AgentState)

    def supervisor_node(state: AgentState) -> Dict[str, Any]:
        return {"steps_taken": state.get("steps_taken", 0) + 1}

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research", researcher.run)
    workflow.add_node("plan", planner.run)
    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges(
        "supervisor", supervisor.route,
        {"research": "research", "plan": "plan", "finish": END},
    )

    workflow.add_edge("research", "supervisor")
    workflow.add_edge("plan", "supervisor")

    return workflow.compile()
