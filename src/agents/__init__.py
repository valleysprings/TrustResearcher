"""
Agents Package

This package contains all the AI agents used in the autonomous research system:
- IdeaGenAgent: Orchestrates research idea generation using specialized tools
- ReviewerAgent: Two-stage peer review (preliminary critique + systematic evaluation)
- RetrievalAgent: Searches for relevant papers using Semantic Scholar API
- SelectionAgent: Unified selector for internal (dedupe/merge) and external (literature) selection
"""

from .ideagen_agent import IdeaGenAgent
from .idea_gen.research_idea import ResearchIdea
from .reviewer_agent import ReviewerAgent
from .retrieval_agent import RetrievalAgent, Paper
from .selection_agent import SelectionAgent, SimilarityResult

__all__ = [
    'IdeaGenAgent',
    'ResearchIdea',
    'ReviewerAgent',
    'RetrievalAgent',
    'Paper',
    'SelectionAgent',
    'SimilarityResult'
]