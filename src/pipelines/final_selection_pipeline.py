"""
Final Selection Pipeline

Performs final selection of the best research ideas using comparative
ranking when more refined ideas exist than requested.
"""

from typing import Dict, Any, List
from .base_pipeline import BasePipeline, PipelineResult, PipelineContext
from ..agents.internal_selector import InternalSelector
from ..agents.aggregator import Aggregator


class FinalSelectionPipeline(BasePipeline):
    """
    Pipeline for final selection of top research ideas
    """
    
    def __init__(self, internal_selector: InternalSelector, aggregator: Aggregator = None):
        super().__init__(
            name="Final Selection",
            description="Phase 6: Final Idea Selection"
        )
        self.internal_selector = internal_selector
        self.aggregator = aggregator
        self.add_dependency("refined_ideas")
        self.add_output("final_ideas")
    
    async def execute(self, context: PipelineContext) -> PipelineResult:
        """
        Perform final selection of best ideas using comparative ranking

        Args:
            context: Pipeline context with refined ideas

        Returns:
            PipelineResult: Final selected ideas and ranking results
        """
        try:
            print("Phase 6: Final Idea Selection")
            print("-" * 40)
            print(f"DEBUG: context.refined_ideas = {context.refined_ideas}")
            print(f"DEBUG: refined_ideas length = {len(context.refined_ideas) if context.refined_ideas else 'None'}")

            if not context.refined_ideas:
                print("DEBUG: No refined ideas available - returning early")
                return PipelineResult(
                    success=False,
                    data={'final_ideas': [], 'selection_summary': 'No refined ideas available'},
                    metadata={'selection_strategy': 'comparative_ranking'},
                    error_message="No refined ideas available for final selection"
                )
            
            # If we have aggregator, use adaptive score-based selection
            if self.aggregator:
                print("Using adaptive score-based selection")
                
                # Get synthesis for all refined ideas
                scored_ideas = []
                for idea in context.refined_ideas:
                    synthesis = self.aggregator.synthesize_feedback(idea.topic)
                    if 'error' not in synthesis:
                        score = synthesis.get('overall_score', 0)
                        scored_ideas.append((idea, score, synthesis))
                
                # Sort all ideas by score descending
                scored_ideas.sort(key=lambda x: x[1], reverse=True)
                
                # Filter for good ideas and above (>=3.5)
                good_ideas = [(idea, score, synthesis) for idea, score, synthesis in scored_ideas if score >= 3.5]
                
                print(f"Found {len(good_ideas)} 'good' ideas (>=3.5) out of {len(scored_ideas)} total ideas")
                print(f"User requested: {context.num_ideas} ideas")
                
                if len(good_ideas) >= context.num_ideas:
                    # Case 1: We have enough good ideas, output ALL good ideas
                    print(f"✅ Outputting ALL {len(good_ideas)} good ideas (≥ requested {context.num_ideas})")
                    final_ideas = [idea for idea, score, synthesis in good_ideas]
                    selection_method = 'all_good_ideas'
                    selection_summary = f"Selected all {len(good_ideas)} good ideas (score ≥ 3.5)"
                    
                elif len(good_ideas) > 0:
                    # Case 2: We have some good ideas but not enough, supplement with highest scoring
                    print(f"⚠️  Only {len(good_ideas)} good ideas available (< requested {context.num_ideas})")
                    print(f"Supplementing with highest scoring ideas to reach {context.num_ideas} total")
                    final_ideas = [idea for idea, score, synthesis in scored_ideas[:context.num_ideas]]
                    selection_method = 'good_plus_highest'
                    selection_summary = f"Selected {len(good_ideas)} good ideas + {context.num_ideas - len(good_ideas)} highest scoring to reach {context.num_ideas} total"
                    
                else:
                    # Case 3: No good ideas, take top requested number
                    print(f"❌ No good ideas found, selecting top {context.num_ideas} by score")
                    final_ideas = [idea for idea, score, synthesis in scored_ideas[:context.num_ideas]]
                    selection_method = 'top_by_score'
                    selection_summary = f"No good ideas found, selected top {context.num_ideas} by score"
                
                # Display selected ideas
                print("Selected ideas:")
                for i, idea in enumerate(final_ideas, 1):
                    # Find the score for this idea
                    idea_score = next((score for selected_idea, score, _ in scored_ideas if selected_idea == idea), 0)
                    quality = "Good" if idea_score >= 3.5 else "Moderate" if idea_score >= 2.5 else "Weak"
                    print(f"  {i}. {idea.topic} (Score: {idea_score:.2f}, Quality: {quality})")
                
                # Update context
                context.final_ideas = final_ideas
                
                selection_data = {
                    'ideas_considered': len(context.refined_ideas),
                    'final_ideas_count': len(final_ideas),
                    'selection_method': selection_method,
                    'good_ideas_found': len(good_ideas),
                    'score_summary': [{'topic': idea.topic, 'score': next((score for selected_idea, score, _ in scored_ideas if selected_idea == idea), 0)} for idea in final_ideas],
                    'final_ideas': final_ideas,
                    'selection_summary': selection_summary
                }
                
                return PipelineResult(
                    success=True,
                    data=selection_data,
                    metadata={
                        'selection_strategy': 'adaptive_score_based',
                        'ranking_used': False,
                        'ideas_considered': len(context.refined_ideas),
                        'final_selection_count': len(final_ideas),
                        'good_ideas_threshold': 3.5,
                        'good_ideas_found': len(good_ideas)
                    }
                )
            
            # If we have fewer or equal refined ideas than requested, use all of them
            if len(context.refined_ideas) <= context.num_ideas:
                print(f"All {len(context.refined_ideas)} refined ideas will be included in final results")
                final_ideas = context.refined_ideas
                
                # Update context
                context.final_ideas = final_ideas
                
                selection_data = {
                    'ideas_considered': len(context.refined_ideas),
                    'final_ideas_count': len(final_ideas),
                    'selection_method': 'all_refined_ideas_included',
                    'final_ideas': final_ideas,
                    'selection_summary': f"All {len(final_ideas)} refined ideas included (no ranking needed)"
                }
                
                return PipelineResult(
                    success=True,
                    data=selection_data,
                    metadata={
                        'selection_strategy': 'include_all_refined',
                        'ranking_used': False,
                        'ideas_considered': len(context.refined_ideas)
                    }
                )
            
            # Perform comparative ranking for final selection
            print(f"Selecting top {context.num_ideas} ideas from {len(context.refined_ideas)} candidates using comparative ranking")
            
            context.logger.log_info(f"Starting comparative ranking for {len(context.refined_ideas)} refined ideas")
            
            # Use comparative ranking for final selection
            comparative_results = await self.internal_selector.rank_ideas_comparatively(context.refined_ideas)
            
            # Check if ranking returned valid results
            if not comparative_results or len(comparative_results) == 0:
                context.logger.log_error("Comparative ranking returned empty results", "FinalSelectionPipeline")
                return PipelineResult(
                    success=False,
                    data={'final_ideas': [], 'selection_summary': 'Comparative ranking failed'},
                    metadata={'selection_strategy': 'comparative_ranking'},
                    error_message="Comparative ranking returned no results"
                )
            
            # Sort by rank and take top requested number
            final_ideas = [idea for idea, ranking in comparative_results[:context.num_ideas] if ranking is not None]
            
            # Update context
            context.final_ideas = final_ideas
            
            print(f"Final selection completed - Chosen {len(final_ideas)} best ideas")
            context.logger.log_info(f"Final selection: {len(final_ideas)} ideas selected from {len(context.refined_ideas)}")
            
            # Prepare ranking summary
            ranking_summary = []
            for i, (idea, ranking) in enumerate(comparative_results[:context.num_ideas]):
                if ranking is not None:
                    ranking_summary.append({
                        'rank': i + 1,
                        'topic': idea.topic,
                        'ranking_score': ranking.get('score', 0),
                        'ranking_criteria': ranking.get('criteria', {})
                    })
            
            selection_data = {
                'ideas_considered': len(context.refined_ideas),
                'final_ideas_count': len(final_ideas),
                'selection_method': 'comparative_ranking',
                'ranking_summary': ranking_summary,
                'final_ideas': final_ideas,  # Full idea objects for downstream use
                'selection_summary': f"Selected top {len(final_ideas)} ideas from {len(context.refined_ideas)} using comparative ranking"
            }
            
            return PipelineResult(
                success=True,
                data=selection_data,
                metadata={
                    'selection_strategy': 'comparative_ranking',
                    'ranking_used': True,
                    'ideas_considered': len(context.refined_ideas),
                    'final_selection_count': len(final_ideas)
                }
            )
            
        except Exception as e:
            context.logger.log_error(f"Final selection failed: {str(e)}", "FinalSelectionPipeline", e)
            return PipelineResult(
                success=False,
                data={
                    'ideas_considered': len(context.refined_ideas) if context.refined_ideas else 0,
                    'final_ideas_count': 0,
                    'final_ideas': [],
                    'selection_summary': 'Final selection failed'
                },
                metadata={'selection_strategy': 'comparative_ranking'},
                error_message=f"Final selection error: {str(e)}"
            )
    
    def validate_dependencies(self, context: PipelineContext) -> bool:
        """
        Validate that we have refined ideas for final selection
        """
        print(f"DEBUG: validate_dependencies called, context.refined_ideas = {context.refined_ideas}")
        print(f"DEBUG: refined_ideas length = {len(context.refined_ideas) if context.refined_ideas else 'None'}")
        if not context.refined_ideas:
            context.logger.log_error("No refined ideas available for final selection", "FinalSelectionPipeline")
            print("DEBUG: validate_dependencies returning False - no refined ideas")
            return False

        print("DEBUG: validate_dependencies returning True")
        return True