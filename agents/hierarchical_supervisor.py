from typing import Dict, Any

MAX_STEPS = 6


class HierarchicalSupervisor:
    def __init__(self, llm=None):
        self.llm = llm

    def route(self, state: Dict[str, Any]) -> str:
        steps = state.get("steps_taken", 0)
        research = state.get("research_data", [])
        plan = state.get("plan", [])

        if steps >= MAX_STEPS:
            return "finish"
        if not research:
            return "research"
        elif not plan:
            return "plan"
        return "finish"
