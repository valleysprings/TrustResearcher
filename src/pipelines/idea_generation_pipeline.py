"""
Idea Generation Pipeline

Handles the core research idea generation using Graph-of-Thought reasoning,
knowledge graphs, and literature context to create novel research directions.
"""

from typing import Dict, Any, List
from .base_pipeline import BasePipeline, PipelineResult, PipelineContext
from ..agents.idea_generator import IdeaGenerator


class IdeaGenerationPipeline(BasePipeline):
    """
    Pipeline for generating research ideas using literature context
    """
    
    def __init__(self, idea_generator: IdeaGenerator):
        super().__init__(
            name="Idea Generation",
            description="Phase 2: Generating Research Ideas (Literature-Informed)"
        )
        self.idea_generator = idea_generator
        self.add_dependency("relevant_papers")
        self.add_output("generated_ideas")
    
    async def execute(self, context: PipelineContext) -> PipelineResult:
        """
        Generate research ideas using literature context and knowledge graphs
        
        Args:
            context: Pipeline context with literature and configuration
            
        Returns:
            PipelineResult: Generated ideas and metadata
        """
        try:
            print("Phase 2: Generating Research Ideas (Literature-Informed)")
            print("-" * 40)
            
            # Set literature context for informed planning
            if context.relevant_papers:
                self.idea_generator.set_literature_context(context.relevant_papers)
                context.logger.log_info(f"Set literature context with {len(context.relevant_papers)} papers")
            
            # Get overgeneration factor from config for robust filtering
            overgenerate_factor = context.config.get('idea_generation', {}).get('overgeneration_factor', 10)
            
            # Generate ideas with overgeneration
            ideas = await self.idea_generator.generate_ideas(
                seed_topic=context.topic,
                num_ideas=context.num_ideas,
                exploration_depth=context.config.get('exploration_depth', 2),
                overgenerate_factor=overgenerate_factor
            )
            
            # Update context
            context.ideas = ideas
            
            print(f"Generated {len(ideas)} literature-informed research ideas "
                  f"({overgenerate_factor}x overgeneration for robust filtering)")
            
            # Log top ideas for debugging
            context.logger.log_info(f"Generated {len(ideas)} ideas with topics: {[idea.topic for idea in ideas[:3]]}")
            
            # Prepare idea generation data
            idea_data = {
                'ideas_generated': len(ideas),
                'target_ideas': context.num_ideas,
                'overgeneration_factor': overgenerate_factor,
                'exploration_depth': context.config.get('exploration_depth', 2),
                'literature_informed': len(context.relevant_papers) > 0,
                'ideas_summary': [
                    {
                        'id': i+1,
                        'topic': str(idea.topic) if idea.topic else '',
                        'problem_statement': (str(idea.problem_statement)[:200] + '...'
                                            if len(str(idea.problem_statement)) > 200
                                            else str(idea.problem_statement))
                    }
                    for i, idea in enumerate(ideas)
                ],
                'ideas': ideas  # Full idea objects for downstream use
            }
            
            return PipelineResult(
                success=True,
                data=idea_data,
                metadata={
                    'generation_strategy': 'literature_informed_graph_of_thought',
                    'ideas_generated': len(ideas),
                    'overgeneration_factor': overgenerate_factor,
                    'literature_papers_used': len(context.relevant_papers)
                }
            )
            
        except Exception as e:
            context.logger.log_error(f"Idea generation failed: {str(e)}", "IdeaGenerationPipeline", e)
            return PipelineResult(
                success=False,
                data={
                    'ideas_generated': 0,
                    'ideas_summary': [],
                    'ideas': []
                },
                metadata={
                    'generation_strategy': 'literature_informed_graph_of_thought',
                    'ideas_generated': 0
                },
                error_message=f"Idea generation error: {str(e)}"
            )
    
    def validate_dependencies(self, context: PipelineContext) -> bool:
        """
        Validate that we have the required components for idea generation
        """
        if not context.topic:
            context.logger.log_error("No topic provided for idea generation", "IdeaGenerationPipeline")
            return False
        
        if context.num_ideas <= 0:
            context.logger.log_error("Invalid number of ideas requested", "IdeaGenerationPipeline")
            return False
        
        # Literature context is optional but recommended
        if not context.relevant_papers:
            context.logger.log_warning("No literature context available - ideas may be less informed", "IdeaGenerationPipeline")
        
        return True