"""
Prompts for PlanningModule - all prompts used by the research planning system

ARCHITECTURE:
- GLOBAL GROUNDING (once at start): Comprehensive context from KG + literature
- PER-IDEA GENERATION (Base/GoT/Cross each independently):
  - Own decomposition based on strategy-specific context
  - Own directions with gaps merged in
"""

# ============================================================================
# GLOBAL GROUNDING (Comprehensive context for all downstream ideas)
# ============================================================================

GLOBAL_GROUNDING_SYSTEM_PROMPT = """You are an expert research analyst who synthesizes comprehensive research landscapes from literature and knowledge graphs.

Your task is to create a rich, inclusive grounding context that will inform multiple downstream research idea generations. Be thorough and capture diverse perspectives."""

GLOBAL_GROUNDING_USER_PROMPT = """Research Topic: {seed_topic}

KNOWLEDGE GRAPH CONTEXT:
{kg_context}

LITERATURE CONTEXT:
{literature_context}

Create a comprehensive research grounding that captures:

1. FIELD OVERVIEW (200-300 words):
- Current state of the field and recent trends
- Major research themes and their interconnections
- Key methodological paradigms in use
- Important datasets, benchmarks, and evaluation standards

2. KEY FINDINGS FROM LITERATURE (200-300 words):
- Summarize the most impactful recent contributions
- Identify consensus views and ongoing debates
- Note emerging techniques or approaches gaining traction
- Highlight cross-disciplinary influences

3. KNOWLEDGE GRAPH INSIGHTS (150-200 words):
- Describe the cluster structure and what each cluster represents
- Identify cross-cluster bridges and interdisciplinary connections
- Note densely connected concepts vs. isolated areas
- Highlight potential unexplored connections

4. RESEARCH LANDSCAPE GAPS (150-200 words):
- Methodological limitations across the field
- Underexplored application domains
- Scalability and practical deployment challenges
- Theoretical foundations needing strengthening
- Missing empirical validations

Return a JSON object:
```json
{{
  "field_overview": "...",
  "key_findings": "...",
  "kg_insights": "...",
  "landscape_gaps": "..."
}}
```"""

# ============================================================================
# STRATEGIC DIRECTION GENERATION (Per-Idea Expansion Directions)
# ============================================================================

# -----------------------------------------------------------------------------
# Strategy 1: Base Variant Directions (Cluster-focused)
# -----------------------------------------------------------------------------

STRATEGIC_DIRECTIONS_BASE_SYSTEM_PROMPT = """You are a research strategist generating methodological expansion directions for cluster-focused research.

Your task is to identify gaps specific to this cluster and generate directions that address them."""

STRATEGIC_DIRECTIONS_BASE_USER_PROMPT = """Research Topic: {seed_topic}

GLOBAL GROUNDING:
{global_grounding}

CLUSTER-SPECIFIC CONTEXT:
{cluster_context}

Based on the global grounding and this specific cluster context:

1. First, identify 3-5 CLUSTER-SPECIFIC GAPS - research opportunities unique to this cluster's domain that aren't fully addressed in the literature.

2. Then, generate 8-10 EXPANSION DIRECTIONS that:
- Leverage specific entities and concepts in this cluster
- Address the cluster-specific gaps you identified
- Propose distinct methodological innovations within this cluster's domain
- Stay grounded in this cluster's specialized area

Return a JSON object:
{{
  "cluster_gaps": ["gap1", "gap2", ...],
  "directions": ["direction1", "direction2", ...]
}}

Each direction should be 1-2 sentences describing a specific methodological expansion."""

# -----------------------------------------------------------------------------
# Strategy 2: GoT Variant Directions (Reasoning Path-focused)
# -----------------------------------------------------------------------------

STRATEGIC_DIRECTIONS_GOT_SYSTEM_PROMPT = """You are a research strategist generating exploration directions based on graph-of-thought reasoning paths.

Your task is to identify gaps along this reasoning path and generate directions that extend it."""

STRATEGIC_DIRECTIONS_GOT_USER_PROMPT = """Research Topic: {seed_topic}

GLOBAL GROUNDING:
{global_grounding}

REASONING PATH CONTEXT:
{path_context}

Based on the global grounding and this reasoning path:

1. First, identify 3-5 PATH-SPECIFIC GAPS - research opportunities revealed by this reasoning path that aren't fully explored.

2. Then, generate 8-10 EXPLORATION DIRECTIONS that:
- Explore different aspects of the reasoning path (theoretical, practical, computational, evaluation)
- Address the path-specific gaps you identified
- Extend the logical reasoning in novel ways
- Connect different nodes in the path to create new insights

Return a JSON object:
{{
  "path_gaps": ["gap1", "gap2", ...],
  "directions": ["direction1", "direction2", ...]
}}

Each direction should be 1-2 sentences describing how to explore or extend the reasoning path."""

# -----------------------------------------------------------------------------
# Strategy 3: Cross-Pollination Directions (Bridge-focused)
# -----------------------------------------------------------------------------

STRATEGIC_DIRECTIONS_CROSS_SYSTEM_PROMPT = """You are a research strategist generating cross-pollination strategies that bridge different clusters and combine existing ideas.

Your task is to identify gaps at cluster intersections and generate bridging strategies."""

STRATEGIC_DIRECTIONS_CROSS_USER_PROMPT = """Research Topic: {seed_topic}

GLOBAL GROUNDING:
{global_grounding}

CROSS-CLUSTER BRIDGE CONTEXT:
{bridge_context}

SAMPLED IDEAS FOR CROSS-POLLINATION:
{sampled_ideas_summary}

Based on the global grounding and this cross-cluster bridge:

1. First, identify 3-5 BRIDGE-SPECIFIC GAPS - research opportunities at the intersection of these clusters that aren't explored.

2. Then, generate 8-10 BRIDGING STRATEGIES that:
- Explicitly identify which clusters/domains are being bridged
- Synthesize concepts from the sampled ideas
- Propose novel methodologies from combining different domains
- Address the bridge-specific gaps you identified

Return a JSON object:
{{
  "bridge_gaps": ["gap1", "gap2", ...],
  "directions": ["strategy1", "strategy2", ...]
}}

Each strategy should be 1-2 sentences focusing on creative synthesis and interdisciplinary innovation."""

# ============================================================================
# HELPER PROMPTS FOR CONTEXT BUILDING
# ============================================================================

# Cluster context extraction (used when sampling a cluster for base variants)
CLUSTER_CONTEXT_EXTRACTION_PROMPT = """Given the following cluster entities from a knowledge graph:

Cluster Entities: {cluster_entities}

Cluster Relationships: {cluster_relationships}

Summarize this cluster's focus in 2-3 sentences, identifying:
1. The main domain or subfield this cluster represents
2. Key methodologies or techniques present
3. Potential research opportunities within this cluster

Keep it concise and domain-specific."""

# Path context extraction (used when sampling a reasoning path for GoT variants)
PATH_CONTEXT_EXTRACTION_PROMPT = """Given the following reasoning path from graph-of-thought exploration:

Path Score: {path_score}
Path Nodes: {path_nodes}
Path Reasoning: {path_reasoning}

Alternative Paths (for context):
{alternative_paths_summary}

Summarize this reasoning path in 2-3 sentences, capturing:
1. The logical flow from start to conclusion
2. Key insights or connections made
3. What novel research directions this path suggests
4. Why this path scored {path_score} (what makes it promising or less promising)

Keep it concise and focused on the reasoning flow."""

# Cross-context building (used when creating cross-pollination contexts)
CROSS_CONTEXT_BUILDING_PROMPT = """Given the following cross-cluster bridge and sampled ideas:

Bridge Description: {bridge_description}
Clusters Being Connected: {cluster_names}

Sampled Idea 1:
{idea1_summary}

Sampled Idea 2:
{idea2_summary}

Summarize how these ideas and clusters could be combined in 2-3 sentences, identifying:
1. What complementary strengths each cluster/idea brings
2. What novel combinations or syntheses are possible
3. What new research opportunities emerge from bridging

Keep it concise and focus on synthesis opportunities."""
