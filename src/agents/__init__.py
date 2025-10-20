"""
Agents Package

This package contains all the AI agents used in the autonomous research system:
- IdeaGenerator: Generates research ideas using Graph-of-Thought reasoning
- ReviewerAgent: Peer-review style evaluation of ideas
- NoveltyAgent: Assesses novelty and significance of ideas
- Aggregator: Synthesizes feedback from multiple agents
- SemanticScholarAgent: Searches for relevant papers using Semantic Scholar API
- InternalSelector: Handles internal idea selection and merging
- ExternalSelector: Selects ideas based on distinctness from existing literature
"""

from .idea_generator import IdeaGenerator, ResearchIdea
from .reviewer_agent import ReviewerAgent
from .novelty_agent import NoveltyAgent
from .aggregator import Aggregator
from .semantic_scholar_agent import SemanticScholarAgent, Paper
from .external_selector import ExternalSelector, SimilarityResult
from .internal_selector import InternalSelector

__all__ = [
    'IdeaGenerator',
    'ResearchIdea',
    'ReviewerAgent',
    'NoveltyAgent',
    'Aggregator',
    'SemanticScholarAgent',
    'Paper',
    'ExternalSelector',
    'SimilarityResult',
    'InternalSelector'
]