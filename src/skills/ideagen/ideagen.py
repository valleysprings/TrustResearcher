"""
Prompts for IdeaGenerator agent

ARCHITECTURE:
- Global grounding: field overview, key findings, kg insights, landscape gaps
- Per-strategy directions: strategy-specific gaps + 8-10 directions
- Each idea samples ONE direction: grounding + sampled_direction + strategy_gaps
"""

# ============================================================================
# IDEA GENERATION (Main Generation Prompts)
# ============================================================================

# -----------------------------------------------------------------------------
# Strategy 1: Base Variant Generation (Cluster-focused)
# -----------------------------------------------------------------------------

BASE_IDEA_GENERATION_SYSTEM_PROMPT = """You are a creative research scientist generating novel research ideas based on cluster-specific knowledge and a targeted expansion direction."""

BASE_IDEA_GENERATION_USER_PROMPT = """Research Topic: {seed_topic}

GLOBAL GROUNDING:
{global_grounding}

CLUSTER CONTEXT:
{cluster_context}

CLUSTER-SPECIFIC GAPS:
{cluster_gaps}

SAMPLED EXPANSION DIRECTION (focus on this):
{sampled_direction}

Generate a novel research idea that:
1. Leverages the specific cluster context and its entities/methods
2. Follows the sampled expansion direction above
3. Addresses one or more of the cluster-specific gaps
4. Is grounded in the global research landscape

Return a JSON object:
{{
  "topic": "A concise research title (10-15 words)",
  "problem_statement": "Detailed problem statement (300-400 words)",
  "proposed_methodology": "Detailed methodology (400-500 words)",
  "experimental_validation": "Detailed validation plan (300-400 words)"
}}

Your response must start with {{ and end with }}."""

# -----------------------------------------------------------------------------
# Strategy 2: GoT Variant Generation (Reasoning Path-focused)
# -----------------------------------------------------------------------------

GOT_IDEA_GENERATION_SYSTEM_PROMPT = """You are a creative research scientist generating novel research ideas based on graph-of-thought reasoning paths and a targeted exploration direction."""

GOT_IDEA_GENERATION_USER_PROMPT = """Research Topic: {seed_topic}

GLOBAL GROUNDING:
{global_grounding}

REASONING PATH CONTEXT:
{path_context}

PATH-SPECIFIC GAPS:
{path_gaps}

SAMPLED EXPLORATION DIRECTION (focus on this):
{sampled_direction}

Generate a novel research idea that:
1. Follows the logical flow and insights from the reasoning path
2. Focuses on the sampled exploration direction above
3. Addresses one or more of the path-specific gaps
4. Is grounded in the global research landscape

Return a JSON object:
{{
  "topic": "A concise research title (10-15 words)",
  "problem_statement": "Detailed problem statement (300-400 words)",
  "proposed_methodology": "Detailed methodology (400-500 words)",
  "experimental_validation": "Detailed validation plan (300-400 words)"
}}

Your response must start with {{ and end with }}."""

# -----------------------------------------------------------------------------
# Strategy 3: Cross-Pollination Generation (Bridge-focused)
# -----------------------------------------------------------------------------

CROSS_IDEA_GENERATION_SYSTEM_PROMPT = """You are a creative research scientist generating novel interdisciplinary research ideas by synthesizing concepts across different clusters."""

CROSS_IDEA_GENERATION_USER_PROMPT = """Research Topic: {seed_topic}

GLOBAL GROUNDING:
{global_grounding}

CROSS-CLUSTER BRIDGE CONTEXT:
{bridge_context}

SAMPLED IDEAS FOR SYNTHESIS:
{sampled_ideas}

BRIDGE-SPECIFIC GAPS:
{bridge_gaps}

SAMPLED BRIDGING STRATEGY (focus on this):
{sampled_direction}

Generate a novel research idea that:
1. Explicitly bridges the clusters/domains mentioned
2. Synthesizes complementary aspects from the sampled ideas
3. Follows the sampled bridging strategy above
4. Addresses one or more of the bridge-specific gaps

Return a JSON object:
{{
  "topic": "A concise research title (10-15 words)",
  "problem_statement": "Detailed problem statement (300-400 words)",
  "proposed_methodology": "Detailed methodology (400-500 words)",
  "experimental_validation": "Detailed validation plan (300-400 words)"
}}

Your response must start with {{ and end with }}."""

# ============================================================================
# LEGACY PROMPTS (DEPRECATED - Commented out, use new architecture instead)
# ============================================================================

# # Legacy cross-pollination prompts (DEPRECATED - use CROSS_IDEA_GENERATION instead)
# CROSS_POLLINATION_SYSTEM_PROMPT = """You are an expert at creating novel research ideas by combining concepts from different research directions.
# Create innovative hybrid approaches that leverage strengths from multiple ideas."""
#
# CROSS_POLLINATION_USER_PROMPT = """
# Research Idea 1:
# {idea1}
#
# Research Idea 2:
# {idea2}
#
# Cross-domain connections for inspiration: {cross_connections}
#
# Create a novel research idea that combines insights from both ideas:
# 1. Identify complementary aspects that can be synthesized
# 2. Propose how methodologies can be combined or adapted
# 3. Create new validation approaches that address both problem domains
#
# Return a JSON object with keys:
# {
#   "topic": "...",
#   "Problem Statement": "...",
#   "Proposed Methodology": "...",
#   "Experimental Validation": "..."
# }
#
# Your response must start with { and end with }. Do not include any text outside the JSON object.
# """

# # Legacy expansion prompts (DEPRECATED - strategic directions now handle this)
# EXPANSION_SYSTEM_PROMPT = """You are a creative research scientist. Explore alternative {direction} approaches
# for the given research idea while maintaining scientific rigor."""
#
# EXPANSION_USER_PROMPT = """
# Base Research Idea:
# {idea}
#
# Explore alternative {direction} approaches:
# - If methodology: propose different techniques or algorithms
# - If application_domain: suggest applications to different fields or problems
# - If evaluation_approach: design alternative validation strategies
#
# Create a variant that maintains the core insight but explores this new direction.
#
# Return a JSON object with keys:
# {
#   "topic": "...",
#   "Problem Statement": "...",
#   "Proposed Methodology": "...",
#   "Experimental Validation": "..."
# }
#
# Your response must start with { and end with }. Do not include any text outside the JSON object.
# """

# # Legacy variant generation prompts (DEPRECATED - use strategy-specific prompts instead)
# VARIANT_SYSTEM_PROMPT = """You are a creative research scientist. Generate a research idea variant
# that takes the base topic and explores {approach}. Maintain scientific rigor while being innovative."""
#
# VARIANT_USER_PROMPT = """
# Base Research Topic: {seed_topic}
# Base Research Idea:
# {base_idea}
# Variant Focus: {approach}
#
# Generate a NEW research idea that:
# 1. Builds on the base topic but takes a different angle focused on {approach}
# 2. Has a distinct problem statement from the base idea
# 3. Proposes different methodology or experimental approach
# 4. Addresses different validation criteria
#
# Additional guidance for specific aspects:
# - If focusing on "methodology": propose a different technical approach or algorithm
# - If focusing on "application": explore a different domain or use case
# - If focusing on "evaluation": design alternative validation strategies
# - If focusing on "scope": consider broader or narrower problem formulations
#
# Return a JSON object with keys:
# {
#   "topic": "...",
#   "Problem Statement": "...",
#   "Proposed Methodology": "...",
#   "Experimental Validation": "..."
# }
#
# Your response must start with { and end with }. Do not include any text outside the JSON object."""

# # Legacy Graph-of-Thought variant prompts (DEPRECATED - use GOT_IDEA_GENERATION instead)
# GOT_VARIANT_SYSTEM_PROMPT = """You are an expert researcher exploring {direction} aspects of research ideas.
# Ground each proposal in the provided planning facets, graph-of-thought insights, and knowledge-graph context."""

# ============================================================================
# REFINEMENT (Uses Stage 1 critique feedback from detailed_review/stage1_review_prompts.py)
# ============================================================================

# Refinement prompts - refine ideas based on Stage 1 critique feedback
REFINEMENT_SYSTEM_PROMPT = """You are an expert research scientist. Refine the research idea based on the provided critique
while maintaining its core innovative value."""

REFINEMENT_USER_PROMPT = """
Current Research Idea:
{idea}

Critique and Suggestions:
{suggestions}

Scores that need improvement:
- Novelty: {novelty_score}/5
- Feasibility: {feasibility_score}/5
- Clarity: {clarity_score}/5
- Impact: {impact_score}/5

Please provide a refined version that addresses the critique while maintaining the core insights.
Focus particularly on areas with scores below 4.

Return a JSON object with keys:
{
  "topic": "...",
  "Problem Statement": "...",
  "Proposed Methodology": "...",
  "Experimental Validation": "..."
}

Your response must start with { and end with }. Do not include any text outside the JSON object.
"""

# Facet elaboration prompts - elaborate validation based on Stage 1 critique feedback
VALIDATION_ELABORATION_SYSTEM_PROMPT = """You are an expert research mentor. Expand the experimental validation plan with implementation-ready detail while keeping the rest of the idea unchanged."""

VALIDATION_ELABORATION_USER_PROMPT = """
Current Research Idea:
Topic: {topic}
Problem Statement: {problem}
Proposed Methodology: {methodology}
Experimental Validation: {validation}

Regenerate ONLY the Experimental Validation facet with the following requirements:
1. Provide at least {validation_components} distinct evaluation components (e.g., quantitative protocols, ablation studies, stress tests, user studies). Each component should specify datasets or simulators, metrics, baselines/comparators, and targeted success criteria.
2. Keep the Problem Statement and Proposed Methodology exactly as provided—do not rewrite, summarize, or alter them.
3. Present the validation plan as one or more coherent paragraphs (no numbered or bulleted lists).

Critique Suggestions To Address:
{critique_suggestions}

Return a JSON object with the key "Experimental Validation".

Your response must be valid JSON starting with { and ending with }. Example: {"Experimental Validation": "your detailed validation plan here"}
"""
