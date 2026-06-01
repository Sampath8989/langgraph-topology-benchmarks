"""Topology explorer — streamlit dashboard for running/competing agent workflows."""

import os
import sys
import time

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # HACK: needed when running from app/ dir

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Add it to your .env file before running.")
    st.stop()

from topologies.sequential import build_sequential_graph
from topologies.parallel import build_parallel_graph
from topologies.hierarchical import build_hierarchical_graph
from topologies.reflection import build_reflection_graph
from evaluation.trajectory_metrics import calculate_metrics

st.set_page_config(page_title="AtlasAgentBench", page_icon="🤖", layout="wide")
st.title("AtlasAgentBench — Multi-Agent Topology Explorer")

# sidebar
st.sidebar.header("Config")
topology_name = st.sidebar.selectbox(
    "Topology", ["Sequential", "Parallel", "Hierarchical", "Reflection"]
)
st.sidebar.markdown(
    "- **Sequential**: Research → Plan (fast, cheap)\n"
    "- **Parallel**: Research ‖ Analysis → Plan\n"
    "- **Hierarchical**: Supervisor routes dynamically\n"
    "- **Reflection**: Plan → Critique → Re-plan loop"
)

# main
col_input, col_output = st.columns([3, 2])

with col_input:
    st.subheader("Task")
    task = st.text_area("Enter a task:", value="What are the main causes of climate change?", height=80)
    run_btn = st.button("Run", type="primary")

if run_btn and task.strip():
    builders = {
        "Sequential": build_sequential_graph,
        "Parallel": build_parallel_graph,
        "Hierarchical": build_hierarchical_graph,
        "Reflection": build_reflection_graph,
    }

    init = {
        "task": task.strip(), "difficulty": "medium",
        "plan": [], "research_data": [], "steps_taken": 0, "final_answer": "",
    }

    with st.spinner(f"Running {topology_name}..."):
        graph = builders[topology_name]()
        t0 = time.time()
        result = graph.invoke(init)
        elapsed = round(time.time() - t0, 2)

    m = calculate_metrics(result)

    with col_output:
        st.subheader("Results")
        if m["success"]:
            st.success("✅ Done")
        else:
            st.warning("⚠️ Completed with issues")

        c1, c2, c3 = st.columns(3)
        c1.metric("Steps", m["steps"])
        c2.metric("Latency", f"{m['latency']}s")
        c3.metric("Cost", f"${m['cost']:.4f}")

        st.divider()

        tab1, tab2, tab3 = st.tabs(["Plan", "Research", "Final Answer"])

        with tab1:
            for i, s in enumerate(result.get("plan", []), 1):
                st.write(f"**{i}.** {s}")

        with tab2:
            for i, r in enumerate(result.get("research_data", []), 1):
                st.info(f"**{i}:** {r}")

        with tab3:
            fa = result.get("final_answer", "")
            if fa:
                st.code(fa, language=None)
            else:
                st.info("No final answer")
else:
    with col_output:
        st.info("Enter a task and click **Run**.")
