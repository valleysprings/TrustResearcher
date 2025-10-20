<div align="center">

# 🔬 Autonomous Research Agent

Elegant, multi-stage research ideation — from literature search to refined, distinct, well‑reviewed ideas — with clear logs, reproducible outputs, and a minimal setup.

<br/>

<img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/Status-Active-34C759" alt="Status" />
<img src="https://img.shields.io/badge/Interface-CLI%20%26%20Web%20UI-8E8E93" alt="Interface" />

</div>

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
- Reproducible Outputs: JSON + summary.txt with timing and costs; single consolidated LLM log per run.

---

## 🏗️ Architecture

### 7-Pipeline System
The system uses a 7-pipeline architecture orchestrated by `ResearchPipelineOrchestrator`:

**Agents (7 specialized components):**
- **SemanticScholarAgent**: Literature search via Semantic Scholar API
- **IdeaGenerator**: Core idea generation with Graph-of-Thought reasoning
- **InternalSelector**: LLM-based idea deduplication and selection  
- **LiteratureSimilarityAgent**: TF-IDF similarity filtering against literature
- **ReviewerAgent**: Peer-review style evaluation
- **NoveltyAgent**: Novelty and significance assessment
- **Aggregator**: Result consolidation and portfolio analysis

**Pipeline Execution Order:**
1. **ValidationPipeline**: Pre-generation validation of system components
2. **LiteratureSearchPipeline**: Paper retrieval via Semantic Scholar API
3. **IdeaGenerationPipeline**: Idea generation with knowledge graph integration
4. **InternalSelectionPipeline**: LLM-based deduplication and initial filtering
5. **ExternalSelectionPipeline**: Literature similarity filtering using TF-IDF
6. **DetailedReviewPipeline**: Multi-agent review (reviewer + novelty + aggregator)
7. **FinalSelectionPipeline**: Final ranking and selection of top ideas
8. **PortfolioAnalysisPipeline**: Portfolio analysis and recommendations

### File Structure
```
autonomous-research-agent/
├── src/
│   ├── __main__.py                   # Main entry point
│   ├── main.py                       # Core application logic
│   ├── ui_launcher.py                # Independent UI system
│   ├── agents/
│   │   ├── base_agent.py             # Base agent interface
│   │   ├── aggregator.py             # Result aggregation
│   │   ├── idea_generator.py         # Core idea generation
│   │   ├── internal_selector.py      # LLM-based selection
│   │   ├── literature_similarity_agent.py  # TF-IDF similarity
│   │   ├── novelty_agent.py          # Novelty assessment
│   │   ├── reviewer_agent.py         # Peer review
│   │   ├── semantic_scholar_agent.py # Literature search
│   │   └── idea_gen/                 # Idea generation modules
│   │       ├── base_agent.py         # Base agent for idea gen
│   │       ├── graph_of_thought.py   # GoT reasoning
│   │       ├── faceted_decomposition.py  # Multi-faceted analysis
│   │       └── planning_module.py    # Strategic planning
│   ├── pipelines/
│   │   ├── research_pipeline_orchestrator.py  # Main orchestrator
│   │   ├── base_pipeline.py          # Pipeline interface
│   │   ├── validation_pipeline.py    # System validation
│   │   ├── literature_search_pipeline.py
│   │   ├── idea_generation_pipeline.py
│   │   ├── internal_selection_pipeline.py
│   │   ├── external_selection_pipeline.py
│   │   ├── detailed_review_pipeline.py
│   │   ├── final_selection_pipeline.py
│   │   └── portfolio_analysis_pipeline.py
│   ├── prompts/                      # All prompt templates
│   │   ├── interface_prompts.py      # Interface prompts
│   │   ├── literature_search/        # Literature search prompts
│   │   │   └── semantic_scholar_agent_prompts.py
│   │   ├── idea_generation/          # Idea generation prompts
│   │   │   ├── idea_generator_prompts.py
│   │   │   ├── faceted_decomposition_prompts.py
│   │   │   ├── kg_builder_prompts.py
│   │   │   └── planning_module_prompts.py
│   │   ├── selection/                # Selection prompts
│   │   │   └── idea_selector_prompts.py
│   │   └── detailed_review/          # Review prompts
│   │       ├── reviewer_agent_prompts.py
│   │       └── novelty_agent_prompts.py
│   ├── knowledge_graph/
│   │   ├── kg_builder.py             # Knowledge graph construction
│   │   └── graph_utils.py            # Graph utilities
│   └── utils/
│       ├── async_utils.py            # Async utilities
│       ├── config.py                 # Configuration management
│       ├── debug_logger.py           # Logging system
│       ├── llm_interface.py          # LLM client
│       ├── phase_timer.py            # Performance tracking
│       ├── pregen_validation.py      # Pre-generation validation
│       ├── session_manager.py        # Session management
│       ├── text_utils.py             # Text processing utilities
│       ├── token_cost_tracker.py     # Token and cost tracking
│       └── web_ui.py                 # Gradio interface
├── configs/
│   └── agent_config.yaml             # Configuration
└── outputs/, logs/, llm_logs/, idea_logs/, sessions/ (runtime)
```

### Pipeline Flow
1. **Literature Search** → Academic paper retrieval (50 papers are good enough to generate mid-to-high-quality ideas)
2. **Knowledge Graph Construction** → Build topic-anchored knowledge graph from literature
3. **Idea Generation** → Multi-method idea generation (planning + faceted decomposition + GoT reasoning + variants + self-critique)
4. **Internal Selection** → LLM deduplication and filtering
5. **External Selection** → TF-IDF / Embedding-based similarity against literature
6. **Detailed Review** → Multi-agent evaluation (reviewer + novelty)
7. **Final Selection** → Top idea ranking
8. **Portfolio Analysis** → Summary and recommendations


---

## 🧭 Pipeline

![pipeline](pipeline.png)

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

 
