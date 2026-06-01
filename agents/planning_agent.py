from typing import Dict, Any, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage


class PlanningAgent:
    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        task = state.get("task", "")
        research = state.get("research_data", [])
        final_answer = state.get("final_answer", "")

        if self.llm is not None:
            context = "\n".join(research) if research else "(none)"
            critique = ""
            if final_answer and "Critique" in final_answer:
                critique = f"\nPrevious attempt was critiqued: {final_answer}\nAddress the issues above."

            response = self.llm.invoke([HumanMessage(content=(
                f"Create a step-by-step plan to solve: {task}\n"
                f"Research context:\n{context}{critique}\n\n"
                "Return a numbered list."
            ))])
            plan = [l.strip() for l in response.content.strip().split("\n") if l.strip()]
            if not plan:
                plan = [response.content.strip()]
        else:
            plan = [
                f"1. Review research findings ({len(research)} entries)",
                f"2. Synthesize answer for: {task[:80]}",
            ]
            if "Critique" in final_answer:
                plan.append("3. Address feedback and revise approach")

        return {"plan": plan, "steps_taken": state.get("steps_taken", 0) + 1}
