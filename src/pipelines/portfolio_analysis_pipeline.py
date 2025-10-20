"""
Portfolio Analysis Pipeline

Conducts final portfolio analysis and generates recommendations
for the selected research ideas.
"""

from typing import Dict, Any, List
from .base_pipeline import BasePipeline, PipelineResult, PipelineContext
from ..agents.aggregator import Aggregator


class PortfolioAnalysisPipeline(BasePipeline):
    """
    Pipeline for portfolio analysis and final recommendations
    """
    
    def __init__(self, aggregator: Aggregator):
        super().__init__(
            name="Portfolio Analysis",
            description="Phase 7: Portfolio Analysis and Recommendations"
        )
        self.aggregator = aggregator
        self.add_dependency("final_ideas")
        self.add_output("portfolio_analysis")
        self.add_output("final_results")
    
    async def execute(self, context: PipelineContext) -> PipelineResult:
        """
        Perform portfolio analysis and generate final recommendations
        
        Args:
            context: Pipeline context with final ideas
            
        Returns:
            PipelineResult: Portfolio analysis and final formatted results
        """
        try:
            print("Phase 7: Portfolio Analysis and Recommendations")
            print("-" * 40)
            
            if not context.final_ideas:
                return PipelineResult(
                    success=False,
                    data={'portfolio_analysis': {}, 'final_results': {}},
                    metadata={'analysis_type': 'portfolio_recommendation'},
                    error_message="No final ideas available for portfolio analysis"
                )
            
            # Get portfolio recommendations
            portfolio_rec = self.aggregator.get_portfolio_recommendation()
            context.logger.log_component_state("aggregator", {"portfolio_recommendation": portfolio_rec})
            
            # Update context
            context.portfolio_analysis = portfolio_rec
            
            # Display portfolio analysis
            print(f"Portfolio Quality: {portfolio_rec.get('portfolio_quality', 'unknown').title()}")
            print(f"Recommended Action: {portfolio_rec.get('recommended_action', 'No recommendation')}")
            
            if 'top_3_ideas' in portfolio_rec:
                print("Top 3 Ideas:")
                for j, idea_topic in enumerate(portfolio_rec['top_3_ideas'], 1):
                    print(f"  {j}. {idea_topic}")
            
            # Display detailed results
            print(f"\nFinal Research Ideas")
            print("=" * 60)
            
            # Prepare ideas data for JSON output
            ideas_json_data = []
            ideas_display_data = []
            
            for i, idea in enumerate(context.final_ideas, 1):
                print(f"\nIDEA {i}:")
                print(idea)
                
                # Get synthesis from aggregator
                synthesis = self.aggregator.synthesize_feedback(idea.topic)
                if 'error' not in synthesis:
                    print("\n=== Review Feedback ===")
                    review_feedback = synthesis['review_feedback']
                    for key, value in review_feedback.items():
                        print(f"{key}: {value}")

                    print("\n=== Novelty Check ===")
                    novelty_check = synthesis['novelty_check']
                    for key, value in novelty_check.items():
                        print(f"{key}: {value}")
                
                # Prepare JSON data
                idea_json = {
                    "id": i,
                    "topic": idea.topic,
                    "problem_statement": idea.problem_statement,
                    "proposed_methodology": idea.proposed_methodology,
                    "experimental_validation": idea.experimental_validation,
                    "review_feedback": synthesis.get('review_feedback', {}) if 'error' not in synthesis else {},
                    "novelty_check": synthesis.get('novelty_check', {}) if 'error' not in synthesis else {}
                }
                ideas_json_data.append(idea_json)
                
                # Prepare display data
                ideas_display_data.append({
                    'idea': idea,
                    'synthesis': synthesis,
                    'display_data': idea_json
                })
                
                print("-" * 40)
            
            # Generate comprehensive JSON output
            json_output = {
                "research_ideas": ideas_json_data,
                "portfolio_analysis": portfolio_rec,
                "literature_search": {
                    "relevant_papers_found": len(context.relevant_papers),
                    "top_papers": [
                        {
                            "title": paper.title,
                            "authors": paper.authors,
                            "year": paper.year,
                            "citation_count": paper.citation_count,
                            "url": paper.url
                        }
                        for paper in context.relevant_papers  # All papers found
                    ],
                },
                "literature_similarity_analysis": context.literature_similarity_results or {},
                "generation_metadata": {
                    "seed_topic": context.topic,
                    "requested_ideas": context.num_ideas,
                    "generated_ideas": len(context.ideas) if context.ideas else 0,
                    "selected_for_review": len(context.selected_ideas) if context.selected_ideas else 0,
                    "literature_filtered_ideas": len(context.filtered_ideas) if context.filtered_ideas else 0,
                    "final_processed_ideas": len(context.final_ideas),
                    "execution_time_seconds": round(context.timer.get_total_time(), 2),
                    "timestamp": context.logger.session_id
                }
            }
            
            # Display JSON output
            print(f"\nJSON Output:")
            print("=" * 60)
            import json
            print(json.dumps(json_output, indent=2, ensure_ascii=False))
            
            # Prepare portfolio analysis data
            portfolio_data = {
                'portfolio_recommendation': portfolio_rec,
                'final_ideas_analyzed': len(context.final_ideas),
                'ideas_display_data': ideas_display_data,
                'json_output': json_output,
                'analysis_summary': f"Portfolio analysis completed for {len(context.final_ideas)} final ideas"
            }
            
            return PipelineResult(
                success=True,
                data=portfolio_data,
                metadata={
                    'analysis_type': 'portfolio_recommendation',
                    'ideas_analyzed': len(context.final_ideas),
                    'portfolio_quality': portfolio_rec.get('portfolio_quality', 'unknown')
                }
            )
            
        except Exception as e:
            context.logger.log_error(f"Portfolio analysis failed: {str(e)}", "PortfolioAnalysisPipeline", e)
            return PipelineResult(
                success=False,
                data={
                    'portfolio_recommendation': {},
                    'final_ideas_analyzed': len(context.final_ideas) if context.final_ideas else 0,
                    'json_output': {},
                    'analysis_summary': 'Portfolio analysis failed'
                },
                metadata={'analysis_type': 'portfolio_recommendation'},
                error_message=f"Portfolio analysis error: {str(e)}"
            )
    
    def validate_dependencies(self, context: PipelineContext) -> bool:
        """
        Validate that we have final ideas for portfolio analysis
        """
        if not context.final_ideas:
            context.logger.log_error("No final ideas available for portfolio analysis", "PortfolioAnalysisPipeline")
            return False
        
        return True