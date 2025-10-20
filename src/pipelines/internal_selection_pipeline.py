"""
Internal Selection Pipeline

Handles internal selection and merging of generated ideas to eliminate 
redundancy and combine similar high-quality ideas using LLM-based merging.
"""

from typing import Dict, Any, List
from .base_pipeline import BasePipeline, PipelineResult, PipelineContext
from ..agents.internal_selector import InternalSelector


class InternalSelectionPipeline(BasePipeline):
    """
    Pipeline for internal idea selection with LLM-based iterative merging
    """
    
    def __init__(self, internal_selector: InternalSelector):
        super().__init__(
            name="Internal Selection",
            description="Phase 4: Internal Selection (Iterative LLM-based Merging)"
        )
        self.internal_selector = internal_selector
        self.add_dependency("generated_ideas")
        self.add_output("selected_ideas")
    
    async def execute(self, context: PipelineContext) -> PipelineResult:
        """
        Perform internal selection with LLM-based iterative merging
        
        Args:
            context: Pipeline context with generated ideas
            
        Returns:
            PipelineResult: Selected ideas and selection metadata
        """
        try:
            print("Phase 3: Internal Selection (Iterative LLM-based Merging)")
            print("-" * 40)
            
            if not context.ideas:
                return PipelineResult(
                    success=False,
                    data={'selected_ideas': [], 'selection_summary': []},
                    metadata={'selection_strategy': 'llm_iterative_merging'},
                    error_message="No ideas available for selection"
                )
            
            # Quick evaluate all generated ideas first
            evaluated_ideas = await self.internal_selector.quick_evaluate_ideas(context.ideas)
            
            # Get selection parameters from config
            intermediate_multiplier = context.config.get('idea_generation', {}).get('intermediate_selection_multiplier', 2)
            target_count = context.num_ideas * intermediate_multiplier
            merge_threshold = context.config.get('idea_generation', {}).get('merge_similarity_threshold', 0.7)
            
            # Perform LLM-based iterative merging selection
            selected_ideas = await self.internal_selector.select_ideas_with_llm_merging(
                evaluated_ideas,
                target_count=target_count,
                merge_threshold=merge_threshold
            )
            
            # Update context
            context.selected_ideas = selected_ideas
            
            print(f"Selected {len(selected_ideas)} ideas from {len(context.ideas)} generated ideas "
                  f"using LLM-based iterative merging (target: {target_count})")
            
            # Log selection details
            context.logger.log_info(f"Internal selection complete: {len(selected_ideas)} ideas: {[idea.topic for idea in selected_ideas]}")
            
            # Prepare selection data
            selection_data = {
                'ideas_evaluated': len(context.ideas),
                'ideas_selected': len(selected_ideas),
                'selection_ratio': len(selected_ideas) / len(context.ideas) if context.ideas else 0,
                'target_count': target_count,
                'merge_threshold': merge_threshold,
                'selection_summary': [
                    {
                        'id': i+1,
                        'topic': idea.topic,
                        'selected_reason': 'llm_iterative_merging_survivor'
                    }
                    for i, idea in enumerate(selected_ideas)
                ],
                'selected_ideas': selected_ideas  # Full idea objects for downstream use
            }
            
            return PipelineResult(
                success=True,
                data=selection_data,
                metadata={
                    'selection_strategy': 'llm_iterative_merging',
                    'ideas_evaluated': len(context.ideas),
                    'ideas_selected': len(selected_ideas),
                    'selection_parameters': {
                        'target_count': target_count,
                        'merge_threshold': merge_threshold,
                        'intermediate_multiplier': intermediate_multiplier
                    }
                }
            )
            
        except Exception as e:
            context.logger.log_error(f"Internal selection failed: {str(e)}", "InternalSelectionPipeline", e)
            return PipelineResult(
                success=False,
                data={
                    'ideas_evaluated': len(context.ideas) if context.ideas else 0,
                    'ideas_selected': 0,
                    'selection_summary': [],
                    'selected_ideas': []
                },
                metadata={'selection_strategy': 'llm_iterative_merging'},
                error_message=f"Internal selection error: {str(e)}"
            )
    
    def validate_dependencies(self, context: PipelineContext) -> bool:
        """
        Validate that we have ideas to select from
        """
        if not context.ideas:
            context.logger.log_error("No generated ideas available for internal selection", "InternalSelectionPipeline")
            return False
        
        if len(context.ideas) < context.num_ideas:
            context.logger.log_warning(
                f"Only {len(context.ideas)} ideas generated, less than requested {context.num_ideas}",
                "InternalSelectionPipeline"
            )
        
        return True