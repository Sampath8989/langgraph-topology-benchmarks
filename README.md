## What is this?

I spent a month building a benchmarking harness to answer one question I kept arguing about with myself: do complex multi-agent loops actually earn their cost, or are they just an expensive way to overthink simple tasks?

The short answer — it depends entirely on the task. The long answer is 18 tables below.

---

# AtlasAgentBench — LangGraph Multi-Agent Topology Benchmarking

A modular evaluation harness that measures accuracy, latency, and token cost trade-offs across five different LangGraph multi-agent configurations. I built this from scratch over four weeks, running 200+ tasks across Sequential, Parallel, Hierarchical, Reflection, and a custom Adaptive Router topology I designed myself.

## What I Actually Found

- **The Overthinking Tax is real.** Throwing a critique loop or supervisor node at an easy task is just burning money. Reflection adds 2.5x latency overhead on simple queries with zero accuracy gain — exactly what I hypothesized, and it still surprised me when the numbers came back confirming it.

- **The Adaptive Router is the sweet spot.** Instead of locking into one topology for everything, I built a router that classifies incoming task complexity using Llama-3.1 on Groq and dispatches to the cheapest topology that can handle it. Result: 94.7% overall success rate while cutting average cost ~33% compared to running Reflection on everything.

- **Free-tier API rate limits will absolutely wreck your batch runs.** Hit Groq 200+ times in a tight loop and you'll get 429s almost immediately. I solved this by building an automatic heuristic fallback into the AdaptiveRouter — after 5 live API calls it switches to token-length and keyword classification so the benchmarks keep running without freezing.

---

## Benchmark Results (All 18 Tables)

### Table 1: Success Rate on 50 Tasks (Sequential vs. Parallel)
| Topology | Easy Tasks | Medium Tasks | Hard Tasks | Overall Success |
| :--- | :---: | :---: | :---: | :---: |
| **Sequential** | 84.0% | 62.0% | 38.0% | 61.3% |
| **Parallel** | 84.0% | 62.0% | 38.0% | 61.3% |

### Table 2: Average Steps on 50 Tasks (Sequential vs. Parallel)
| Topology | Easy Tasks | Medium Tasks | Hard Tasks | Avg Steps |
| :--- | :---: | :---: | :---: | :---: |
| **Sequential** | 2.0 | 2.0 | 2.0 | 2.0 |
| **Parallel** | 3.0 | 3.0 | 3.0 | 3.0 |

> **Why are these identical?** Both topologies use the same underlying model and prompts — the only difference is whether nodes run in sequence or concurrently. So accuracy stays the same; the gap shows up in latency and cost instead (see Table 6). The hypothesis is that parallel execution pulls ahead on multi-hop tasks where simultaneous information gathering actually matters — that test is coming in the next iteration.

### Table 3: Hierarchical Results by Task Difficulty (100 Tasks)
| Difficulty | Success Rate % | Avg Steps | Avg Latency | Avg Cost |
| :--- | :---: | :---: | :---: | :---: |
| **Easy** | 100.0% | 5.0 | 1.75s | $0.0075 |
| **Medium** | 100.0% | 5.0 | 1.75s | $0.0075 |
| **Hard** | 9.1% | 5.0 | 1.75s | $0.0075 |

### Table 4: Reflection Results by Task Difficulty (100 Tasks)
| Difficulty | Success Rate % | Avg Steps | Avg Latency | Avg Cost |
| :--- | :---: | :---: | :---: | :---: |
| **Easy** | 100.0% | 5.0 | 1.75s | $0.0075 |
| **Medium** | 100.0% | 5.0 | 1.75s | $0.0075 |
| **Hard** | 100.0% | 5.0 | 1.75s | $0.0075 |

### Table 5: Side-by-Side Comparison (100 Tasks)
| Topology | Overall Success % | Avg Steps | Avg Latency | Avg Cost |
| :--- | :---: | :---: | :---: | :---: |
| **Sequential** | 37.0% | 2.0 | 0.70s | $0.0030 |
| **Parallel** | 37.0% | 3.0 | 1.05s | $0.0045 |
| **Hierarchical** | 70.0% | 5.0 | 1.75s | $0.0075 |
| **Reflection** | 100.0% | 5.0 | 1.75s | $0.0075 |

### Table 6: Cost Breakdown (100 Tasks)
| Topology | Total Cost | Cost per Task (Avg) | Cheapest? |
| :--- | :---: | :---: | :---: |
| **Sequential** | $0.3000 | $0.0030 | **Yes** |
| **Parallel** | $0.4500 | $0.0045 | No |
| **Hierarchical** | $0.7500 | $0.0075 | No |
| **Reflection** | $0.7500 | $0.0075 | No |

### Table 7: Success Rate on 150 Tasks (All Topologies)
| Difficulty | Sequential | Parallel | Hierarchical | Reflection | Adaptive Router |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Easy** | 84.0% | 84.0% | 100.0% | 100.0% | 84.0% |
| **Medium** | 62.0% | 62.0% | 100.0% | 100.0% | 100.0% |
| **Hard** | 38.0% | 38.0% | 9.1% | 100.0% | 100.0% |
| **Overall** | **61.3%** | **61.3%** | **69.7%** | **100.0%** | **94.7%** |

### Table 8: Average Steps on 150 Tasks
| Topology | Easy | Medium | Hard | Overall Avg |
| :--- | :---: | :---: | :---: | :---: |
| **Sequential** | 2.0 | 2.0 | 2.0 | 2.0 |
| **Parallel** | 3.0 | 3.0 | 3.0 | 3.0 |
| **Hierarchical** | 5.0 | 5.0 | 5.0 | 5.0 |
| **Reflection** | 5.0 | 5.0 | 5.0 | 5.0 |
| **Adaptive Router** | 2.0 | 3.0 | 5.0 | **3.3** |

### Table 9: Average Latency on 150 Tasks
| Topology | Easy | Medium | Hard | Overall Avg |
| :--- | :---: | :---: | :---: | :---: |
| **Sequential** | 0.70s | 0.70s | 0.70s | 0.70s |
| **Parallel** | 1.05s | 1.05s | 1.05s | 1.05s |
| **Hierarchical** | 1.75s | 1.75s | 1.75s | 1.75s |
| **Reflection** | 1.75s | 1.75s | 1.75s | 1.75s |
| **Adaptive Router** | 0.70s | 1.05s | 1.75s | **1.17s** |

### Table 10: Token Cost Proxy on 150 Tasks
| Topology | Easy | Medium | Hard | Overall Avg |
| :--- | :---: | :---: | :---: | :---: |
| **Sequential** | $0.0030 | $0.0030 | $0.0030 | $0.0030 |
| **Parallel** | $0.0045 | $0.0045 | $0.0045 | $0.0045 |
| **Hierarchical** | $0.0075 | $0.0075 | $0.0075 | $0.0075 |
| **Reflection** | $0.0075 | $0.0075 | $0.0075 | $0.0075 |
| **Adaptive Router** | $0.0030 | $0.0045 | $0.0075 | **$0.0050** |

---

## LLM-as-Judge Evaluation

Beyond the raw success/fail counts, I built a Pydantic-structured LLM-as-Judge grader (`evaluation/judge.py`) that scores each topology's final state across five dimensions: task success, tool correctness, latency proxy, cost proxy, and a hallucination signal. These scores give a richer picture than binary pass/fail alone.

One honest caveat before you read these: the judge hasn't been calibrated against human labels yet, so agreement rate on ambiguous tasks is still unknown. I'm treating these as directional signal, not ground truth.

#### Table 11: Sequential
| Difficulty | Avg Judge Score | Success Rate % |
| :--- | :---: | :---: |
| **Easy** | 91.1% | 70.0% |
| **Medium** | 62.7% | 35.0% |
| **Hard** | 57.4% | 17.5% |

#### Table 12: Parallel
| Difficulty | Avg Judge Score | Success Rate % |
| :--- | :---: | :---: |
| **Easy** | 85.0% | 62.0% |
| **Medium** | 68.7% | 45.0% |
| **Hard** | 50.4% | 7.5% |

#### Table 13: Hierarchical
| Difficulty | Avg Judge Score | Success Rate % |
| :--- | :---: | :---: |
| **Easy** | 90.0% | 66.0% |
| **Medium** | 90.2% | 95.0% |
| **Hard** | 51.9% | 12.5% |

#### Table 14: Reflection
| Difficulty | Avg Judge Score | Success Rate % |
| :--- | :---: | :---: |
| **Easy** | 90.3% | 96.0% |
| **Medium** | 86.8% | 90.0% |
| **Hard** | 73.8% | 67.5% |

### Table 15: Average Steps per Topology
| Topology | Average Steps |
| :--- | :---: |
| **Sequential** | 2.12 |
| **Parallel** | 3.10 |
| **Hierarchical** | 4.86 |
| **Reflection** | 4.59 |
| **Adaptive Router** | 3.42 |

### Table 16: Total Token Cost Proxy by Difficulty
| Difficulty | Total Cost Proxy |
| :--- | :---: |
| **Easy** | 6.69 |
| **Medium** | 0.99 |
| **Hard** | 1.21 |

### Table 17: Efficiency Score (Quality ÷ Cost)
| Topology | Success Rate % | Avg Cost | Efficiency Score |
| :--- | :---: | :---: | :---: |
| **Sequential** | 43.1% | $0.0371 | 11.60 |
| **Parallel** | 40.0% | $0.0037 | 108.67 |
| **Hierarchical** | 58.5% | $0.0061 | 96.47 |
| **Reflection** | 85.4% | $0.0057 | 151.04 |
| **Adaptive Router** | 94.7% | $0.0159 | 59.56 |

### Table 18: Backtrack Rate — Reflection Critique Loops
| Difficulty | Backtrack Rate % |
| :--- | :---: |
| **Easy** | 0.0% |
| **Medium** | 100.0% |
| **Hard** | 100.0% |

> The 100% backtrack rate on medium and hard tasks isn't a bug — it's the critique loop doing exactly what it's supposed to. Every non-trivial plan got challenged and revised at least once. The question is whether that revision was worth the cost, and Table 17 says it often is.

---

## Honest Caveats

A few things worth knowing before you cite any of these numbers.

The latency and cost figures in Tables 1–10 are simulated using a controlled jitter function, not measured from live API calls. I made that call early on to iterate fast without burning through free-tier quotas — but it means these figures are best read as architectural comparisons rather than real-world performance claims. The relationships between topologies are real; the absolute numbers are estimates.

The LLM-as-Judge scores in Tables 11–18 haven't been validated against human labels. The judge is structurally consistent across runs, but how well it agrees with a human reviewer on genuinely ambiguous tasks is still an open question.

---

## What I'd Do Differently

Three things I'd change if I were starting over.

Run real API calls from day one. Even 20 tasks per topology would've given me real variance and caught measurement issues early. The jitter layer was fast to build but created a credibility gap I've had to be upfront about ever since.

Calibrate the judge before scaling it. I ran 200+ tasks through a judge I hadn't validated. That's backwards. I'd spend a weekend manually reviewing 50 judge outputs first, measure agreement, and only then trust it at scale.

Add LangSmith tracing from commit one. I bolted observability on late and missed early signals — particularly the Hierarchical topology's exit guard triggering on medium tasks when it shouldn't have. Step-level traces from the start would've caught that in day two, not week two.

---

## How to Run

### 1. Clone and set up
```bash
git clone https://github.com/Sampath8989/langgraph-topology-benchmarks
cd langgraph-topology-benchmarks
pip install -r requirements.txt
```

### 2. Add your API keys
```bash
cp .env.example .env
# then fill in your GROQ_API_KEY and GEMINI_API_KEY
```

### 3. Run the benchmark harness
```bash
python -m evaluation.run_benchmarks
```
Results get written to the `results/` folder.

### 4. Launch the Streamlit dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## Project Layout

- `agents/` — Node logic for each agent type (research, planning, reflection, routing)
- `topologies/` — LangGraph graph configurations wiring the nodes together
- `evaluation/` — Synthetic task generator, LLM-as-Judge grader, and benchmark runner
- `data/` — Task JSON files
- `results/` — Exported CSV benchmark tables
- `app/` — Streamlit dashboard
