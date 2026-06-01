from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel

from agents.state import AgentState
from agents.research_agent import ResearchAgent
from agents.planning_agent import PlanningAgent
from agents.reflection_agent import ReflectionAgent

MAX_REFLECTION_ROUNDS = 3


def build_reflection_graph(llm: Optional[BaseChatModel] = None):
    researcher = ResearchAgent(llm)
    planner = PlanningAgent(llm)
    reflector = ReflectionAgent(llm)

    workflow = StateGraph(AgentState)
    workflow.add_node("research", researcher.run)
    workflow.add_node("plan", planner.run)
    workflow.add_node("reflect", reflector.reflect)
    workflow.set_entry_point("research")
    workflow.add_edge("research", "plan")
    workflow.add_edge("plan", "reflect")

    def should_continue(state: AgentState) -> str:
        final = state.get("final_answer", "")
        steps = state.get("steps_taken", 0)
        if "Critique" in final and steps < MAX_REFLECTION_ROUNDS * 3:
            return "retry"
        return "finish"

    workflow.add_conditional_edges(
        "reflect", should_continue,
        {"retry": "plan", "finish": END},
    )

    return workflow.compile()
