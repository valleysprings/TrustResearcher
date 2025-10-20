"""
External Selection Pipeline

Analyzes the similarity of internally selected ideas against existing literature
to filter out ideas that are too similar to published work.
"""

from typing import Dict, Any, List
from .base_pipeline import BasePipeline, PipelineResult, PipelineContext
from ..agents.external_selector import ExternalSelector


class ExternalSelectionPipeline(BasePipeline):
    """
    Pipeline for external selection - filtering ideas against existing literature
    """

    def __init__(self, external_selector: ExternalSelector):
        super().__init__(
            name="External Selection",
            description="Phase 5: External Selection (Literature Similarity Filter)"
        )
        self.external_selector = external_selector
        self.add_dependency("selected_ideas")
        self.add_dependency("relevant_papers")
        self.add_output("filtered_ideas")
        self.add_output("literature_similarity_analysis")
    
    async def execute(self, context: PipelineContext) -> PipelineResult:
        """
        Analyze external distinctness of selected ideas against literature
        
        Args:
            context: Pipeline context with selected ideas and relevant papers
            
        Returns:
            PipelineResult: Externally distinct ideas and analysis results
        """
        try:
            print("Phase 5: External Selection (Literature Similarity Filter)")
            print("-" * 40)
            
            if not context.selected_ideas:
                return PipelineResult(
                    success=False,
                    data={'filtered_ideas': [], 'literature_similarity_report': {}},
                    metadata={'analysis_type': 'literature_similarity'},
                    error_message="No selected ideas available for external selection"
                )
            
            if not context.relevant_papers:
                context.logger.log_warning("No literature available for external selection - all ideas will pass", "ExternalSelectionPipeline")
                # If no papers, all ideas are considered externally approved
                context.filtered_ideas = context.selected_ideas
                return PipelineResult(
                    success=True,
                    data={
                        'filtered_ideas': context.selected_ideas,
                        'literature_similarity_report': {'warning': 'No literature available for comparison'},
                        'analysis_summary': 'All ideas passed external selection (no literature for comparison)'
                    },
                    metadata={'analysis_type': 'literature_similarity', 'papers_available': 0}
                )
            
            # Analyze literature similarity of selected ideas
            similarity_results = self.external_selector.batch_analyze_distinctness(
                context.selected_ideas,
                context.relevant_papers
            )

            # Filter ideas to keep only sufficiently different from literature
            filtered_ideas, rejected_ideas = self.external_selector.filter_distinct_ideas(
                context.selected_ideas,
                context.relevant_papers
            )

            # Generate literature similarity report
            literature_similarity_report = self.external_selector.generate_distinctness_report(similarity_results)
            
            # Update context
            context.filtered_ideas = filtered_ideas
            context.literature_similarity_results = literature_similarity_report  # Store the JSON-serializable report
            
            print(f"{len(filtered_ideas)}/{len(context.selected_ideas)} internally selected ideas are sufficiently different from existing literature")
            
            if len(filtered_ideas) < len(context.selected_ideas):
                rejected_count = len(context.selected_ideas) - len(filtered_ideas)
                print(f"{rejected_count} ideas were too similar to existing work")
                
                if len(filtered_ideas) == 0:
                    print("No externally approved ideas found. Consider adjusting the topic or similarity thresholds.")
                    context.logger.log_warning("No externally approved ideas found after filtering", "ExternalSelectionPipeline")
                    # Fallback to original selection
                    filtered_ideas = context.selected_ideas
                    context.filtered_ideas = filtered_ideas
            
            # Log literature similarity results
            context.logger.log_info(f"External selection: {len(filtered_ideas)}/{len(context.selected_ideas)} ideas are sufficiently different from literature")
            
            # Prepare literature similarity analysis data
            similarity_analysis_data = {
                'ideas_analyzed': len(context.selected_ideas),
                'approved_ideas_count': len(filtered_ideas),
                'rejected_ideas_count': len(context.selected_ideas) - len(filtered_ideas),
                'approval_ratio': len(filtered_ideas) / len(context.selected_ideas) if context.selected_ideas else 0,
                'papers_used_for_comparison': len(context.relevant_papers),
                'literature_similarity_report': literature_similarity_report,
                'filtered_ideas': filtered_ideas,  # Full idea objects for downstream use
                'rejected_ideas': rejected_ideas,
                'analysis_summary': f"{len(filtered_ideas)}/{len(context.selected_ideas)} ideas passed external selection filter"
            }
            
            return PipelineResult(
                success=True,
                data=similarity_analysis_data,
                metadata={
                    'analysis_type': 'literature_similarity',
                    'similarity_thresholds': self.external_selector.config,
                    'papers_used': len(context.relevant_papers),
                    'approval_ratio': len(filtered_ideas) / len(context.selected_ideas) if context.selected_ideas else 0
                }
            )
            
        except Exception as e:
            context.logger.log_error(f"External selection failed: {str(e)}", "ExternalSelectionPipeline", e)
            return PipelineResult(
                success=False,
                data={
                    'ideas_analyzed': len(context.selected_ideas) if context.selected_ideas else 0,
                    'approved_ideas_count': 0,
                    'literature_similarity_report': {},
                    'filtered_ideas': [],
                    'analysis_summary': 'External selection failed'
                },
                metadata={'analysis_type': 'literature_similarity'},
                error_message=f"External selection error: {str(e)}"
            )
    
    def validate_dependencies(self, context: PipelineContext) -> bool:
        """
        Validate that we have selected ideas for analysis
        """
        if not context.selected_ideas:
            context.logger.log_error("No selected ideas available for external selection", "ExternalSelectionPipeline")
            return False
        
        # Papers are optional but recommended
        if not context.relevant_papers:
            context.logger.log_warning("No literature papers available - external selection will be limited", "ExternalSelectionPipeline")
        
        return True