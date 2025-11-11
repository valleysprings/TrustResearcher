"""
Pipeline Architecture for TrustResearcher

This module defines the core pipeline interfaces and implementations
for the multi-stage research idea generation and evaluation system.
"""

from .base_pipeline import BasePipeline, PipelineResult, PipelineContext
from .validation_pipeline import ValidationPipeline
from .literature_search_pipeline import LiteratureSearchPipeline
from .idea_generation_pipeline import IdeaGenerationPipeline
from .internal_selection_pipeline import InternalSelectionPipeline
from .external_selection_pipeline import ExternalSelectionPipeline
from .detailed_review_pipeline import DetailedReviewPipeline
from .final_selection_pipeline import FinalSelectionPipeline
from .portfolio_analysis_pipeline import PortfolioAnalysisPipeline
from .research_pipeline_orchestrator import ResearchPipelineOrchestrator

__all__ = [
    'BasePipeline',
    'PipelineResult',
    'PipelineContext',
    'ValidationPipeline',
    'LiteratureSearchPipeline',
    'IdeaGenerationPipeline',
    'InternalSelectionPipeline',
    'ExternalSelectionPipeline',
    'DetailedReviewPipeline',
    'FinalSelectionPipeline',
    'PortfolioAnalysisPipeline',
    'ResearchPipelineOrchestrator'
]