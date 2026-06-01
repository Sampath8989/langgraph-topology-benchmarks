"""
Unit tests — run with: python -m pytest tests/ -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.state import merge_research, merge_steps
from agents.research_agent import ResearchAgent
from agents.planning_agent import PlanningAgent
from agents.reflection_agent import ReflectionAgent
from agents.hierarchical_supervisor import HierarchicalSupervisor, MAX_STEPS
from agents.adaptive_router import AdaptiveRouter


# -- state helpers --

def test_merge_research_dedup():
    assert merge_research(["a", "b"], ["b", "c"]) == ["a", "b", "c"]

def test_merge_research_none():
    assert merge_research(None, ["a"]) == ["a"]
    assert merge_research(None, None) == []

def test_merge_steps():
    assert merge_steps(3, 5) == 5
    assert merge_steps(0, 0) == 0


# -- research agent --

def test_research_appends():
    agent = ResearchAgent()
    result = agent.run({"task": "test", "research_data": [], "steps_taken": 0})
    assert len(result["research_data"]) == 1
    assert "test" in result["research_data"][0]

def test_research_preserves():
    agent = ResearchAgent()
    result = agent.run({"task": "t2", "research_data": ["old"], "steps_taken": 2})
    assert result["research_data"][0] == "old"
    assert result["steps_taken"] == 3


# -- planning agent --

def test_plan_generates():
    result = PlanningAgent().run({
        "task": "solve X", "research_data": ["f1", "f2"],
        "final_answer": "", "steps_taken": 1,
    })
    assert len(result["plan"]) >= 2

def test_plan_expands_on_critique():
    result = PlanningAgent().run({
        "task": "solve X", "research_data": ["f1"],
        "final_answer": "Critique: too short", "steps_taken": 1,
    })
    assert len(result["plan"]) >= 3


# -- reflection agent --

def test_reflect_criticizes_short_plan():
    r = ReflectionAgent().reflect({
        "plan": ["step 1"], "research_data": ["d"],
        "steps_taken": 1, "task": "simple",
    })
    assert "Critique" in r["final_answer"]

def test_reflect_passes_good_plan():
    r = ReflectionAgent().reflect({
        "plan": [
            "1. Gather relevant facts and context",
            "2. Synthesize findings into a coherent response",
            "3. Review and finalize the output",
        ],
        "research_data": ["d"], "steps_taken": 1, "task": "simple",
    })
    assert "Success" in r["final_answer"]

def test_reflect_fails_no_research():
    r = ReflectionAgent().reflect({
        "plan": ["step 1", "step 2"], "research_data": [],
        "steps_taken": 1, "task": "task",
    })
    assert "Critique" in r["final_answer"]

def test_reflect_complex_needs_more():
    r = ReflectionAgent().reflect({
        "plan": ["step 1", "step 2"], "research_data": ["d"],
        "steps_taken": 1, "task": "prove that x works",
    })
    assert "Critique" in r["final_answer"]


# -- supervisor --

def test_supervisor_routes_research_first():
    s = HierarchicalSupervisor()
    assert s.route({"steps_taken": 0, "research_data": [], "plan": []}) == "research"

def test_supervisor_routes_plan():
    s = HierarchicalSupervisor()
    assert s.route({"steps_taken": 1, "research_data": ["d"], "plan": []}) == "plan"

def test_supervisor_finishes():
    s = HierarchicalSupervisor()
    assert s.route({"steps_taken": 2, "research_data": ["d"], "plan": ["s"]}) == "finish"

def test_supervisor_max_steps():
    s = HierarchicalSupervisor()
    assert s.route({"steps_taken": MAX_STEPS, "research_data": [], "plan": []}) == "finish"


# -- router heuristic --

def test_router_easy():
    # short factual questions should always be easy
    r = AdaptiveRouter(api_key="fake")
    assert r.classify_complexity("What is 2+2?") == "easy"

def test_router_hard():
    r = AdaptiveRouter(api_key="fake")
    assert r.classify_complexity("Prove all primes > 2 are odd") == "hard"

def test_router_medium():
    # this one's borderline — heuristic might say medium or hard depending on keywords
    r = AdaptiveRouter(api_key="fake")
    assert r.classify_complexity("Design a notification system") in ("medium", "hard")

def test_router_reset():
    AdaptiveRouter._api_call_count = 100
    AdaptiveRouter.reset_counter()
    assert AdaptiveRouter._api_call_count == 0


# -- topology compilation smoke tests --

def test_sequential_compiles():
    from topologies.sequential import build_sequential_graph
    assert build_sequential_graph() is not None

def test_parallel_compiles():
    from topologies.parallel import build_parallel_graph
    assert build_parallel_graph() is not None

def test_hierarchical_compiles():
    from topologies.hierarchical import build_hierarchical_graph
    assert build_hierarchical_graph() is not None

def test_reflection_compiles():
    from topologies.reflection import build_reflection_graph
    assert build_reflection_graph() is not None

def test_adaptive_compiles():
    from topologies.adaptive_router_topology import build_adaptive_router_graph
    assert build_adaptive_router_graph() is not None


# -- topology e2e execution --

def _base():
    return {
        "task": "What is the capital of France?",
        "difficulty": "easy", "plan": [],
        "research_data": [], "steps_taken": 0, "final_answer": "",
    }

def test_sequential_runs():
    from topologies.sequential import build_sequential_graph
    r = build_sequential_graph().invoke(_base())
    assert len(r["plan"]) >= 1
    assert r["steps_taken"] >= 2

def test_parallel_runs():
    from topologies.parallel import build_parallel_graph
    r = build_parallel_graph().invoke(_base())
    assert r["steps_taken"] >= 3

def test_hierarchical_runs():
    from topologies.hierarchical import build_hierarchical_graph
    r = build_hierarchical_graph().invoke(_base())
    assert r["steps_taken"] >= 3

def test_reflection_runs():
    from topologies.reflection import build_reflection_graph
    r = build_reflection_graph().invoke(_base())
    assert len(r["plan"]) >= 2
    assert r["steps_taken"] >= 3

def test_adaptive_runs():
    from topologies.adaptive_router_topology import build_adaptive_router_graph
    s = _base()
    s["task"] = "What is 2+2?"
    r = build_adaptive_router_graph().invoke(s)
    assert "plan" in r


# -- metrics --

def test_metrics_success():
    from evaluation.trajectory_metrics import calculate_metrics
    m = calculate_metrics({
        "plan": ["a", "b"], "research_data": ["d"],
        "steps_taken": 3, "final_answer": "Success",
    })
    assert m["success"] is True
    assert m["cost"] > 0

def test_metrics_fail():
    from evaluation.trajectory_metrics import calculate_metrics
    m = calculate_metrics({
        "plan": ["a", "b"], "research_data": [],
        "steps_taken": 2, "final_answer": "",
    })
    assert m["success"] is False
