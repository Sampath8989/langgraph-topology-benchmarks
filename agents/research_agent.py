from typing import Dict, Any, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage


class ResearchAgent:
    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        task = state.get("task", "")

        if self.llm is not None:
            response = self.llm.invoke([
                HumanMessage(content=f"Research this and extract key facts:\n{task}")
            ])
            info = response.content
        else:
            # no api key — stub mode for tests
            info = f"research::{task[:60]}"

        existing = list(state.get("research_data", []))
        existing.append(info)

        return {
            "research_data": existing,
            "steps_taken": state.get("steps_taken", 0) + 1,
        }
