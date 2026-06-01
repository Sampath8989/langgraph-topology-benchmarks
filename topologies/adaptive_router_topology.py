from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.adaptive_router import AdaptiveRouter
from topologies.sequential import build_sequential_graph
from topologies.parallel import build_parallel_graph
from topologies.reflection import build_reflection_graph

def build_adaptive_router_graph(llm=None):
    router = AdaptiveRouter()
    
    # Precompile graphs to prevent sub-graph construction overhead at runtime
    sequential_graph = build_sequential_graph(llm)
    parallel_graph = build_parallel_graph(llm)
    reflection_graph = build_reflection_graph(llm)
    
    workflow = StateGraph(AgentState)
    
    def route_node(state: AgentState):
        task = state.get("task", "")
        difficulty = router.classify_complexity(task)
        
        # Branch execution to candidate sub-graph
        if difficulty == "easy":
            return sequential_graph.invoke(state)
        elif difficulty == "medium":
            return parallel_graph.invoke(state)
        else:
            return reflection_graph.invoke(state)
            
    workflow.add_node("route", route_node)
    workflow.set_entry_point("route")
    workflow.add_edge("route", END)
    
    return workflow.compile()
