"""
Prompts for FacetedDecomposition - all prompts used by the faceted decomposition system
"""

# Main decomposition prompts
DECOMPOSE_IDEA_SYSTEM_PROMPT = """You are an expert research scientist who specializes in literature-informed research planning and identifying research opportunities from existing work.

CRITICAL LENGTH REQUIREMENT: Your response MUST generate comprehensive, detailed facets totaling AT LEAST 1200-1500 words across all three sections. Each facet should be substantive and thorough:
- Problem Statement: 400-500 words minimum
- Proposed Methodology: 500-600 words minimum
- Experimental Validation: 400-500 words minimum

Your task is to decompose research topics into three key facets that form a complete, literature-informed research proposal with maximum technical detail and depth."""

DECOMPOSE_IDEA_USER_PROMPT = """
Research Topic: {seed_topic}

Literature and Knowledge Context: {knowledge_context}

Based on the provided literature context and domain knowledge, decompose this research topic into three literature-informed facets:

1. PROBLEM STATEMENT (400-500 words REQUIRED):
- What specific problem or research gap does this address that is not fully solved by existing literature?
- Reference specific limitations or gaps identified in the provided literature context
- Why is this problem important and timely given current research trends?
- How does this problem build upon or differ from problems addressed in existing work?
- Provide concrete examples, statistics, and quantitative impact assessments
- Include stakeholder analysis and economic/social implications
- Discuss theoretical foundations and interdisciplinary connections

2. PROPOSED METHODOLOGY (500-600 words REQUIRED):
- What approach would you use that addresses limitations in existing methodologies?
- What techniques, algorithms, or experimental designs would be employed?
- How does this methodology specifically improve upon or extend approaches from the literature?
- What novel aspects distinguish this from existing methods in the literature context?
- Provide step-by-step algorithmic details, mathematical formulations, and system architecture
- Include computational requirements, scalability considerations, and implementation phases
- Specify technology stack, development environment, and team expertise needed
- Address technical risk mitigation and alternative approaches

3. EXPERIMENTAL VALIDATION (400-500 words REQUIRED):
- How would you evaluate the proposed solution in ways that address evaluation gaps in existing literature?
- What datasets, benchmarks, or experimental setups would provide evidence beyond what exists?
- What would constitute success for this research compared to existing benchmarks?
- How would the validation demonstrate clear advantages over approaches in the literature?
- Include statistical analysis plans, power analysis, and significance testing
- Specify evaluation metrics, baseline comparisons, and ablation studies
- Address user studies, real-world deployment testing, and ethical considerations
- Provide cost-benefit analysis and long-term performance monitoring protocols

REQUIREMENTS:
- Each facet must demonstrate clear awareness of existing literature and identify specific opportunities for advancement
- Reference concepts, limitations, or gaps from the provided literature context where relevant
- Ensure the research proposal represents a meaningful contribution beyond existing work
- Provide detailed, specific responses that could guide actual research implementation
- CRITICAL: Each section MUST meet the minimum word count requirements specified above
- Include specific technical details, quantitative estimates, and concrete examples throughout

{json_format_suffix}"""

# Facet refinement prompts
REFINE_FACET_SYSTEM_PROMPT = """You are an expert research scientist. You need to refine the {facet_name} of a research idea based on additional context or feedback."""

REFINE_FACET_USER_PROMPT = """
Current {facet_name}: {current_facet}

Refinement Context/Feedback: {refinement_context}

Please provide an improved version of the {facet_name} that addresses the feedback and incorporates the additional context.
Be specific and actionable in your response.
"""

# Research outline generation prompts
RESEARCH_OUTLINE_SYSTEM_PROMPT = """You are an expert research scientist. Create a comprehensive research outline from the provided facets."""

RESEARCH_OUTLINE_USER_PROMPT = """
Based on these research facets:

Problem Statement: {problem_statement}

Proposed Methodology: {proposed_methodology}

Experimental Validation: {experimental_validation}

Create a structured research outline that includes:
1. Research Objectives
2. Literature Review Focus Areas
3. Methodology Details
4. Implementation Plan
5. Evaluation Strategy
6. Expected Contributions

Format this as a clear, actionable research plan.
"""