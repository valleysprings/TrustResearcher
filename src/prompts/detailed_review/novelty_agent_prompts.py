"""
Prompts for NoveltyAgent - all prompts used by the novelty assessment system
"""

# Comprehensive novelty assessment prompts
COMPREHENSIVE_NOVELTY_SYSTEM_PROMPT = """You are a research expert specializing in novelty assessment. 
Your task is to thoroughly evaluate the originality and novelty of research ideas by 
comparing them against existing work and identifying genuinely novel contributions."""

COMPREHENSIVE_NOVELTY_USER_PROMPT = """
Research Idea to Evaluate:
{idea}

Existing Work Context:
{existing_work_context}

Please assess the novelty across these dimensions:
{dimensions_description}

For each dimension, provide:
1. Novelty score (1-5, where 5 is highly novel)
2. Specific aspects that are novel or similar to existing work
3. Key differences from existing approaches
4. Potential for genuine contribution

Additionally, provide:
- Overall novelty assessment and score
- Most significant novel contributions
- Areas where novelty could be enhanced
- Suggestions for positioning against existing work

Be thorough and critical in your assessment.

IMPORTANT: Please provide your response in valid JSON format only. Use this structure:

{{
  "dimension_scores": {{
    "technical_novelty": 4.2,
    "problem_novelty": 3.8,
    "application_novelty": 4.0,
    "theoretical_novelty": 3.5,
    "empirical_novelty": 4.1
  }},
  "novel_aspects": [
    "Novel attention mechanism design",
    "First application to this domain",
    "New theoretical framework"
  ],
  "similar_work": [
    "Builds on transformer architectures",
    "Similar to recent work in domain adaptation"
  ],
  "key_differences": [
    "Different attention pattern",
    "Novel loss function",
    "Multi-modal integration"
  ],
  "enhancement_areas": [
    "Strengthen theoretical foundations",
    "Explore more diverse applications",
    "Compare with recent baseline methods"
  ],
  "positioning_suggestions": [
    "Emphasize multi-modal aspects",
    "Position as extension of existing work",
    "Highlight practical advantages"
  ]
}}

IMPORTANT NOTES:
- Do NOT include "overall_novelty_score" in your response - it will be calculated automatically as the average of dimension_scores.
- Ensure all text fields are properly escaped for JSON.
- Do not include any text outside the JSON structure."""

# Novelty dimensions descriptions (used to build the prompt)
NOVELTY_DIMENSIONS = {
    'technical_novelty': 'Novel algorithms, methods, or technical approaches',
    'problem_novelty': 'Addressing previously unexplored problems or gaps',
    'application_novelty': 'New applications of existing methods to different domains',
    'theoretical_novelty': 'New theoretical insights or frameworks',
    'empirical_novelty': 'Novel experimental designs or evaluation approaches'
}