from typing import Dict, Any


def calculate_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
    steps = state.get("steps_taken", 0)
    final = state.get("final_answer", "")

    has_plan = len(state.get("plan", [])) >= 2
    has_research = len(state.get("research_data", [])) > 0
    success = has_plan and has_research and "Critique" not in final

    return {
        "success": success,
        "steps": steps,
        "latency": round(steps * 0.35, 2),
        "cost": round(steps * 0.0015, 4),
    }
