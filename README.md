<div align="center">

# 🔬 Autonomous Research Agent

Elegant, multi-stage research ideation — from literature search to refined, distinct, well‑reviewed ideas — with clear logs, reproducible outputs, and a minimal setup.

<br/>

<img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/Status-Active-34C759" alt="Status" />
<img src="https://img.shields.io/badge/Interface-CLI%20%26%20Web%20UI-8E8E93" alt="Interface" />

</div>

![pipeline](pipeline.png)

---

## 🧩 What It Does

The Autonomous Research Agent takes a topic and produces a polished set of research ideas by orchestrating a practical, literature‑aware pipeline:

- Retrieves relevant papers via the Semantic Scholar API with concurrency, rate‑limiting, retries, de‑duplication, and relevance/citation ranking.
- Builds a topic‑anchored knowledge graph to maintain external memory during ideation.
- Over‑generates ideas through planning, faceted decomposition, exploration, and self‑critique, then removes duplicates.
- Quickly evaluates candidates with weighted criteria and configurable distinctness thresholds.
- Checks distinctness against retrieved papers to avoid overlap.
- Runs reviewer, novelty, and proofreading agents in parallel and aggregates results into clean outputs.

The result is a structured JSON artifact (plus a human‑readable summary) and comprehensive logs of the process.

---

 

## ✨ Key Features

- Literature‑Guided Pipeline: high‑signal retrieval before ideation, with sensible defaults and backoffs.
- Knowledge Graph Memory: lightweight `networkx` graph from topic (optionally from documents) to ground downstream reasoning.
- Robust Generation: planning + faceted decomposition + exploration variants + self‑critique with de‑duplication.
- Fast Screening: weighted, configurable selection + distinctness thresholds.
- Parallel Deep Review: reviewer, novelty, and proofreading in parallel with consolidated reports.
- Web UI: interactive process visualization, multi-session control, and live logs.
- Reproducible Outputs: JSON output with timing and costs; with multiple logs such as multi-round idea refinement and llms conversation could been utilized for future reference.


---

## 🧭 Pipeline


1. **Literature Search** → Academic paper retrieval (50 papers are good enough to generate mid-to-high-quality ideas)
2. **Knowledge Graph Construction** → Build topic-anchored knowledge graph from literature
3. **Idea Generation** → Multi-method idea generation (planning + faceted decomposition + GoT reasoning + variants + self-critique)
4. **Internal Selection** → LLM deduplication and filtering
5. **External Selection** → TF-IDF / Embedding-based similarity against literature
6. **Detailed Review** → Multi-agent evaluation (reviewer + novelty)
7. **Final Selection** → Top idea ranking
8. **Portfolio Analysis** → Summary and recommendations

---

## ⚙️ Installation

Requirements
- Python 3.8+
- Network access for the model API and Semantic Scholar

*Tip: use a virtual environment (venv or conda) to isolate dependencies.*

Install
```bash
pip install -e .
```

Configure your credentials in `configs/custom_pipeline_example.yaml` and rename to `agent_config.yaml`:

---

## 🖼️ Case Study: Web UI in Action

Here’s what the interactive Web UI looks like when running a research session:

![case_study_ui](casestudy.png)

---

## 🚀 Quick Start

* CLI

```bash
# help
python -m src --help

# full pipeline (ensure configs/agent_config.yaml is set)
python -m src --topic "Design scalable and robust algorithms for the k-truss breaking problem that bypass global trussness updates via localized, incremental, and approximation methods, enabling near-real-time interventions on large-scale graphs." --num_ideas 2 --debug
```



* Web UI

```bash
# process visualization UI
python -m src.ui_launcher --process-ui

# set UI host (default: localhost; use 0.0.0.0 for LAN)
python -m src.ui_launcher --process-ui --process-host 0.0.0.0

# set UI port (default: 7860)
python -m src.ui_launcher --process-ui --process-port 7861
```

---

## 📤 Outputs & 📜 Logs

- Results: `outputs/{topic}_{timestamp}.json` with the complete pipeline output.
- Run logs: `logs/session_YYYYMMDD_HHMMSS.log` (single file per run).
- LLM logs: `llm_logs/{topic}_{timestamp}.jsonl` (All interaction from agents per run with token & cost stats).
- Idea logs: `idea_logs/ideas_{timestamp}.json` (all generated ideas for each refinement stage).

---

## 🧯 Troubleshooting

- Always run as a module: `python -m src ...` (avoid `python src/main.py`).
- Ensure write permissions for `outputs/`, `logs/`, and `llm_logs/`.