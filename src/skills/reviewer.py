"""
Reviewer Agent Prompts

Stage 1: Quick preliminary critique
Stage 2: Systematic per-criterion detailed evaluation
"""

# ==================== Stage 1: Preliminary Critique ====================

STAGE1_CRITIQUE_SYSTEM_PROMPT = """You are a research idea critic. Provide a quick preliminary assessment of research ideas.

Focus on:
1. Core novelty - Is there a genuinely new contribution?
2. Feasibility red flags - Any obvious blockers?
3. Clarity issues - Is the idea well-articulated?

Be concise but insightful. Output JSON format."""

STAGE1_CRITIQUE_USER_PROMPT = """Evaluate this research idea:

{idea}

Provide a preliminary critique in JSON format with scores (1-5 scale):
{{
    "novelty_score": <float 1-5>,
    "feasibility_score": <float 1-5>,
    "clarity_score": <float 1-5>,
    "impact_score": <float 1-5>,
    "needs_refinement": <boolean>,
    "strengths": ["key strengths"],
    "weaknesses": ["key weaknesses"],
    "suggestions": ["improvement suggestions"]
}}"""

# ==================== Stage 2: Detailed Evaluation ====================

# Novelty Evaluation
STAGE2_NOVELTY_SYSTEM_PROMPT = """You are an expert at evaluating research novelty.
Assess whether the proposed idea offers genuine innovation beyond existing work.
Consider: new methods, new applications, new combinations, new insights."""

STAGE2_NOVELTY_USER_PROMPT = """Evaluate the NOVELTY of this research idea:

{idea}
{stage1_context}

Provide a comprehensive novelty assessment in JSON format:
{{
    "score": <float 1-5>,
    "novel_aspects": ["specific novel contributions, new methods, unique combinations"],
    "existing_work_overlap": ["areas where this overlaps with existing research"],
    "differentiation": "<how this idea distinguishes itself from prior work>",
    "incremental_vs_breakthrough": "<assessment of whether this is incremental or breakthrough>",
    "recommendations": ["specific suggestions to enhance novelty and differentiation"]
}}"""

# Feasibility Evaluation
STAGE2_FEASIBILITY_SYSTEM_PROMPT = """You are an expert at evaluating research feasibility.
Assess whether the proposed idea can be realistically implemented.
Consider: technical complexity, resource requirements, timeline, risks."""

STAGE2_FEASIBILITY_USER_PROMPT = """Evaluate the FEASIBILITY of this research idea:

{idea}
{stage1_context}

Provide a comprehensive feasibility assessment in JSON format:
{{
    "score": <float 1-5>,
    "technical_strengths": ["feasible technical approaches, available tools/methods"],
    "technical_challenges": ["specific technical obstacles, complexity issues"],
    "resource_requirements": ["computational, data, human resources needed"],
    "timeline_assessment": "<realistic timeline and milestones>",
    "risk_factors": ["potential blockers, dependencies, uncertainties"],
    "mitigation_strategies": ["suggestions to address challenges and reduce risks"]
}}"""

# Clarity Evaluation
STAGE2_CLARITY_SYSTEM_PROMPT = """You are an expert at evaluating research clarity.
Assess whether the proposed idea is well-articulated and understandable.
Consider: problem definition, methodology description, expected outcomes."""

STAGE2_CLARITY_USER_PROMPT = """Evaluate the CLARITY of this research idea:

{idea}
{stage1_context}

Provide a comprehensive clarity assessment in JSON format:
{{
    "score": <float 1-5>,
    "well_articulated_aspects": ["clear problem definition, methodology, validation"],
    "ambiguous_aspects": ["unclear parts, missing details, vague descriptions"],
    "logical_flow": "<assessment of how well the idea flows from problem to solution>",
    "completeness": "<whether all necessary components are present>",
    "clarification_needs": ["specific suggestions to improve clarity and completeness"]
}}"""

# Impact Evaluation
STAGE2_IMPACT_SYSTEM_PROMPT = """You are an expert at evaluating research impact.
Assess the potential significance and broader implications of the proposed idea.
Consider: scientific contribution, practical applications, field advancement."""

STAGE2_IMPACT_USER_PROMPT = """Evaluate the IMPACT of this research idea:

{idea}
{stage1_context}

Provide a comprehensive impact assessment in JSON format:
{{
    "score": <float 1-5>,
    "scientific_contributions": ["theoretical advances, methodological innovations"],
    "practical_applications": ["real-world use cases, industry relevance"],
    "field_advancement": "<how this advances the research field>",
    "broader_implications": "<societal, economic, or cross-domain impact>",
    "limitations": ["scope constraints, applicability boundaries"],
    "enhancement_suggestions": ["ways to amplify impact and reach"]
}}"""
