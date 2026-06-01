import os
import json
from groq import Groq
from dotenv import load_dotenv

def generate_synthetic_tasks(n: int = 200) -> list:
    # Tasks are generated fresh each run unless cached to data/tasks.json.
    # For reproducibility, generated tasks are saved and reused if the file already exists.
    base_dir = os.path.dirname(os.path.dirname(__file__))
    tasks_json_path = os.path.join(base_dir, "data", "tasks.json")
    if os.path.exists(tasks_json_path):
        try:
            with open(tasks_json_path, "r") as f:
                return json.load(f)
        except Exception:
            pass

    load_dotenv(os.path.join(base_dir, ".env"))
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return _generate_fallback_tasks(n)
        
    client = Groq(api_key=api_key)
    tasks = []
    difficulties = ["easy", "medium", "hard"]
    
    batch_size = 50
    batches = n // batch_size
    
    # Request batches of 50 to avoid API token limits
    for b in range(batches):
        diff = difficulties[b % 3]
        prompt = f"""Generate {batch_size} unique AI agent benchmark tasks of difficulty: {diff}.
Output strictly as a JSON object with a key "tasks" containing a list of objects.
Each object must have:
  - "id": integer starting at {b * batch_size + 1}
  - "task": string question or task
  - "difficulty": "{diff}"

Schema example:
{{
  "tasks": [
    {{"id": 1, "task": "example task...", "difficulty": "{diff}"}}
  ]
}}
"""
        try:
            # Force JSON mode
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            raw = completion.choices[0].message.content
            parsed = json.loads(raw)
            tasks.extend(parsed.get("tasks", []))
        except Exception:
            # Fallback for individual batch failure
            tasks.extend(_generate_fallback_batch(b * batch_size + 1, batch_size, diff))
            
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "synthetic_200.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(tasks, f, indent=2)
        
    with open(tasks_json_path, "w") as f:
        json.dump(tasks, f, indent=2)
        
    return tasks

def _generate_fallback_tasks(n: int) -> list:
    tasks = []
    difficulties = ["easy", "medium", "hard"]
    for i in range(1, n + 1):
        diff = difficulties[(i - 1) % 3]
        tasks.append({
            "id": i,
            "task": f"Synthetic benchmark task {i} ({diff}).",
            "difficulty": diff
        })
    return tasks

def _generate_fallback_batch(start_id: int, size: int, diff: str) -> list:
    batch = []
    for i in range(start_id, start_id + size):
        batch.append({
            "id": i,
            "task": f"Synthetic task {i} ({diff}).",
            "difficulty": diff
        })
    return batch
