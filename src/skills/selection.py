"""
Prompts for the SelectionAgent

This module contains prompts and utilities used by the SelectionAgent for merging
similar research ideas.
"""

# LLM-based idea merging prompts
IDEA_MERGING_SYSTEM_PROMPT = """You are an expert research strategist specializing in combining similar research ideas into comprehensive, unified research proposals.

Your task is to merge two or more similar research ideas into a single, enhanced research idea that:
- Combines the strongest aspects of each input idea
- Eliminates redundancy while preserving unique contributions
- Creates a more comprehensive and robust research approach
- Maintains clarity and feasibility
- Enhances the overall research impact

MERGING PRINCIPLES:
1. Preserve the most compelling elements from each idea
2. Synthesize complementary approaches and methodologies
3. Expand the scope to encompass broader research questions
4. Integrate experimental validation approaches
5. Amplify potential impact by combining applications
6. Maintain scientific rigor and feasibility

RESPONSE FORMAT:
Provide a complete merged research idea with:
- Unified Topic (concise, descriptive title)
- Integrated Problem Statement
- Combined Methodology
- Comprehensive Experimental Validation
- Enhanced Potential Impact

Ensure the merged idea is coherent, well-integrated, and represents a genuine improvement over the individual components."""

IDEA_MERGING_USER_PROMPT = """Merge the following similar research ideas into a single, comprehensive research proposal. Combine their strengths while eliminating redundancy:

IDEAS TO MERGE:
{ideas_to_merge}

Create a unified research idea that integrates the best aspects of each input idea while being more comprehensive and impactful than any individual component."""


def format_ideas_for_merge(idea1, idea2) -> str:
    """
    Format two ideas for LLM merge prompt.

    Args:
        idea1: First ResearchIdea object
        idea2: Second ResearchIdea object

    Returns:
        Formatted string with both ideas
    """
    formatted = f"""
    IDEA 1:
    Topic: {idea1.topic}
    Problem Statement: {idea1.problem_statement}
    Proposed Methodology: {idea1.proposed_methodology}
    Experimental Validation: {idea1.experimental_validation}

    IDEA 2:
    Topic: {idea2.topic}
    Problem Statement: {idea2.problem_statement}
    Proposed Methodology: {idea2.proposed_methodology}
    Experimental Validation: {idea2.experimental_validation}
    """
    return formatted.strip()