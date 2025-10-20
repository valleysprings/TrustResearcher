"""
Literature Search Pipeline

Handles searching for relevant academic papers using Semantic Scholar API
to provide literature context for research idea generation.
"""

from typing import Dict, Any, List
from .base_pipeline import BasePipeline, PipelineResult, PipelineContext
from ..agents.semantic_scholar_agent import SemanticScholarAgent


class LiteratureSearchPipeline(BasePipeline):
    """
    Pipeline for searching relevant academic literature
    """
    
    def __init__(self, semantic_scholar_agent: SemanticScholarAgent):
        super().__init__(
            name="Literature Search",
            description="Phase 1: Searching Relevant Literature"
        )
        self.semantic_scholar_agent = semantic_scholar_agent
        self.add_output("relevant_papers")
    
    async def execute(self, context: PipelineContext) -> PipelineResult:
        """
        Search for relevant papers using Semantic Scholar
        
        Args:
            context: Pipeline context containing topic and configuration
            
        Returns:
            PipelineResult: Found papers and search metadata
        """
        try:
            print("Phase 1: Searching Relevant Literature")
            print("-" * 40)
            
            # Get number of papers to search from config
            num_papers_to_search = context.config.get('semantic_scholar', {}).get('num_papers', 10)
            
            # Search for relevant papers
            relevant_papers = await self.semantic_scholar_agent.search_papers_by_topic(
                context.topic,
                num_papers=num_papers_to_search
            )
            
            # Update context
            context.relevant_papers = relevant_papers
            
            print(f"Found {len(relevant_papers)} relevant papers")
            
            # Display top papers
            if relevant_papers:
                print("Top papers found:")
                for i, paper in enumerate(relevant_papers[:3], 1):
                    print(f"  {i}. {paper.title} ({paper.year}) - Citations: {paper.citation_count}")
            
            # Prepare literature data for output
            literature_data = {
                'papers_found': len(relevant_papers),
                'search_query': context.topic,
                'num_papers_requested': num_papers_to_search,
                'top_papers': [
                    {
                        'title': paper.title,
                        'authors': paper.authors,
                        'year': paper.year,
                        'citation_count': paper.citation_count,
                        'url': paper.url,
                        'abstract': getattr(paper, 'abstract', '')[:200] + '...' if getattr(paper, 'abstract', '') else ''
                    }
                    for paper in relevant_papers  # All papers found
                ],
                'papers': relevant_papers  # Full paper objects for downstream use
            }
            
            return PipelineResult(
                success=True,
                data=literature_data,
                metadata={
                    'search_strategy': 'semantic_scholar_topic_search',
                    'papers_found': len(relevant_papers),
                    'search_parameters': {
                        'topic': context.topic,
                        'num_papers': num_papers_to_search
                    }
                }
            )
            
        except Exception as e:
            context.logger.log_error(f"Literature search failed: {str(e)}", "LiteratureSearchPipeline", e)
            
            # Return empty results but don't fail the pipeline
            return PipelineResult(
                success=False,
                data={
                    'papers_found': 0,
                    'search_query': context.topic,
                    'top_papers': [],
                    'papers': []
                },
                metadata={
                    'search_strategy': 'semantic_scholar_topic_search',
                    'papers_found': 0
                },
                error_message=f"Literature search error: {str(e)}"
            )
    
    def validate_dependencies(self, context: PipelineContext) -> bool:
        """
        Validate that we have the required configuration for literature search
        """
        semantic_scholar_config = context.config.get('semantic_scholar', {})
        
        if not semantic_scholar_config:
            context.logger.log_error("No semantic_scholar configuration found", "LiteratureSearchPipeline")
            return False
        
        if not semantic_scholar_config.get('api_key'):
            context.logger.log_warning("No Semantic Scholar API key found - searches may be limited", "LiteratureSearchPipeline")
        
        return True