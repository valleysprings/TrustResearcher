"""
Prompts for PlanningModule - all prompts used by the research planning system
"""

# Stepwise plan creation prompts
STEPWISE_PLAN_SYSTEM_PROMPT = """You are an expert research planner. Create a detailed, actionable research plan 
that breaks down the research into concrete, executable steps."""

STEPWISE_PLAN_USER_PROMPT = """
Based on these research facets:

Problem Statement: {problem_statement}
Proposed Methodology: {proposed_methodology}
Experimental Validation: {experimental_validation}

Create a step-by-step research plan with the following structure:
1. Literature Review and Background Research
2. Problem Formulation and Scope Definition
3. Methodology Development
4. Implementation Planning
5. Experimental Design
6. Validation and Testing
7. Analysis and Evaluation
8. Documentation and Dissemination

For each step, provide:
- Specific tasks to be completed
- Expected deliverables
- Dependencies on other steps
- Estimated timeline

Format as a structured plan with clear phases.
"""

# General plan refinement prompts
GENERAL_PLAN_REFINEMENT_SYSTEM_PROMPT = """You are an expert research planner. Refine the research plan based on the provided feedback
while maintaining the overall structure and goals."""

GENERAL_PLAN_REFINEMENT_USER_PROMPT = """
Current Research Plan:
{current_plan}

Feedback: {feedback}

Please provide a refined research plan that addresses the feedback while maintaining:
1. Scientific rigor
2. Feasibility
3. Clear methodology
4. Proper validation approaches

Focus on the areas mentioned in the feedback and provide specific improvements.
"""