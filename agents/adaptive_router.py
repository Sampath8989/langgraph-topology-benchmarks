import os
from groq import Groq
from dotenv import load_dotenv

class AdaptiveRouter:
    # TODO: Add dynamic window-based tracking instead of static session limits
    _api_call_count = 0
    _MAX_LIVE_CALLS = 5

    def __init__(self, api_key=None):
        if not api_key:
            load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
            api_key = os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=api_key)

    def _heuristic_classify(self, task: str) -> str:
        # Fallback if Groq rates-limits us
        if len(task) < 35:
            return "easy"
        elif "prove" in task.lower() or "verify" in task.lower():
            return "hard"
        else:
            return "medium"

    def classify_complexity(self, task: str) -> str:
        # API Throttling Guard: Switch to heuristics if we hit the evaluation loop volume limit
        if AdaptiveRouter._api_call_count >= AdaptiveRouter._MAX_LIVE_CALLS:
            return self._heuristic_classify(task)

        prompt = f"""You are a multi-agent complexity classifier. Classify the following task into exactly one word: "easy", "medium", or "hard".
- "easy": Simple lookups, basic facts, one-step queries.
- "medium": Detailed writing, parsing, structured comparisons.
- "hard": Mathematical proofs, verification tasks, complex logic puzzles.

Return only: easy, medium, or hard.

Task: {task}"""
        try:
            AdaptiveRouter._api_call_count += 1
            completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            val = completion.choices[0].message.content.strip().lower()
            if "easy" in val:
                return "easy"
            elif "hard" in val:
                return "hard"
            elif "medium" in val:
                return "medium"
        except Exception:
            pass
            
        return self._heuristic_classify(task)
