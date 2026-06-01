from typing import TypedDict, List, Annotated

# Custom merge reducers to prevent LangGraph concurrent write crashes.
# merge_research deduplicates incoming entries.
def merge_research(left: List[str], right: List[str]) -> List[str]:
    merged = list(left) if left else []
    for item in (right or []):
        if item not in merged:
            merged.append(item)
    return merged

# merge_steps returns the maximum step count.
def merge_steps(left: int, right: int) -> int:
    return max(left or 0, right or 0)

class AgentState(TypedDict):
    task: str
    difficulty: str
    plan: List[str]
    research_data: Annotated[List[str], merge_research]
    steps_taken: Annotated[int, merge_steps]
    final_answer: str
