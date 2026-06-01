from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel

from agents.state import AgentState
from agents.research_agent import ResearchAgent
from agents.planning_agent import PlanningAgent


def build_parallel_graph(llm: Optional[BaseChatModel] = None):
    researcher = ResearchAgent(llm)
    planner = PlanningAgent(llm)

    workflow = StateGraph(AgentState)

    def start_node(state: AgentState) -> Dict[str, Any]:
        return {"steps_taken": state.get("steps_taken", 0) + 1}

    def analyze_task_node(state: AgentState) -> Dict[str, Any]:
        task = state.get("task", "")
        # quick task decomposition — gives the planner a second signal
        analysis = f"Task breakdown: '{task[:100]}' — requires research, synthesis, output."
        return {
            "research_data": [analysis],
            "steps_taken": state.get("steps_taken", 0) + 1,
        }

    workflow.add_node("start", start_node)
    workflow.add_node("research", researcher.run)
    workflow.add_node("analyze", analyze_task_node)
    workflow.add_node("plan", planner.run)

    workflow.set_entry_point("start")
    workflow.add_edge("start", "research")
    workflow.add_edge("start", "analyze")
    workflow.add_edge("research", "plan")
    workflow.add_edge("analyze", "plan")
    workflow.add_edge("plan", END)

    return workflow.compile()
