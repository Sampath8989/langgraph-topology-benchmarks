from typing import Dict, Any


class ReflectionAgent:
    def __init__(self, llm=None):
        self.llm = llm

    def reflect(self, state: Dict[str, Any]) -> Dict[str, Any]:
        plan = state.get("plan", [])
        research = state.get("research_data", [])
        steps = state.get("steps_taken", 0)
        task = state.get("task", "")

        issues = []

        if len(plan) < 2:
            issues.append(f"plan too short ({len(plan)} steps)")

        # check for lazy single-word steps
        terse = sum(1 for s in plan if len(str(s).strip()) < 10)
        if terse:
            issues.append(f"{terse}/{len(plan)} steps are too terse")

        if len(research) == 0:
            issues.append("no research before planning")

        # harder tasks need more thorough plans
        if any(w in task.lower() for w in ("prove", "verify", "analyze")):
            if len(plan) < 3:
                issues.append("complex task needs 3+ steps")

        if issues:
            critique = "Critique: " + "; ".join(issues) + "."
        else:
            critique = "Success: Plan meets quality criteria."

        return {
            "final_answer": critique,
            "steps_taken": steps + 1,
        }
