from typing import Optional

from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel

from agents.state import AgentState
from agents.research_agent import ResearchAgent
from agents.planning_agent import PlanningAgent


def build_sequential_graph(llm: Optional[BaseChatModel] = None):
    researcher = ResearchAgent(llm)
    planner = PlanningAgent(llm)

    workflow = StateGraph(AgentState)
    workflow.add_node("research", researcher.run)
    workflow.add_node("plan", planner.run)
    workflow.set_entry_point("research")
    workflow.add_edge("research", "plan")
    workflow.add_edge("plan", END)

    return workflow.compile()
