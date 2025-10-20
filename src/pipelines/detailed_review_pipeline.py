"""
Detailed Review Pipeline

Conducts comprehensive review and critique of distinct ideas using
multiple specialized agents working in parallel.
"""

import asyncio
from typing import Dict, Any, List
from .base_pipeline import BasePipeline, PipelineResult, PipelineContext
from ..agents.reviewer_agent import ReviewerAgent
from ..agents.novelty_agent import NoveltyAgent
from ..agents.aggregator import Aggregator
from ..agents.idea_generator import ResearchIdea


class DetailedReviewPipeline(BasePipeline):
    """
    Pipeline for detailed review and critique of ideas using multiple agents
    """
    
    def __init__(self, reviewer_agent: ReviewerAgent, novelty_agent: NoveltyAgent,
                 aggregator: Aggregator):
        super().__init__(
            name="Detailed Review",
            description="Phase 5: Detailed Review and Critique"
        )
        self.reviewer_agent = reviewer_agent
        self.novelty_agent = novelty_agent
        self.aggregator = aggregator
        
        self.add_dependency("filtered_ideas")
        self.add_output("refined_ideas")
        self.add_output("review_results")
    
    async def execute(self, context: PipelineContext) -> PipelineResult:
        """
        Perform detailed review and critique of literature-filtered ideas
        
        Args:
            context: Pipeline context with filtered ideas
            
        Returns:
            PipelineResult: Refined ideas and review results
        """
        try:
            print("Phase 6: Detailed Review and Critique")
            print("-" * 40)
            
            if not context.filtered_ideas:
                return PipelineResult(
                    success=False,
                    data={'refined_ideas': [], 'review_summary': []},
                    metadata={'review_strategy': 'multi_agent_parallel'},
                    error_message="No filtered ideas available for detailed review"
                )
            
            ideas_to_process = context.filtered_ideas
            
            # Process all ideas in parallel
            print(f"Processing {len(ideas_to_process)} ideas in parallel...")
            context.logger.log_info(f"Starting parallel processing of {len(ideas_to_process)} ideas")
            
            # Create tasks for all ideas
            processing_tasks = [
                self._process_single_idea(idea, i+1, context)
                for i, idea in enumerate(ideas_to_process)
            ]
            
            # Execute all idea processing tasks concurrently
            refined_ideas_results = await asyncio.gather(*processing_tasks, return_exceptions=True)
            
            # Filter out exceptions and collect successful results
            refined_ideas = []
            review_results = []
            
            for i, result in enumerate(refined_ideas_results):
                if isinstance(result, Exception):
                    context.logger.log_error(f"Failed to process idea {i+1}: {result}", "DetailedReviewPipeline")
                    print(f"Failed to process idea {i+1}: {result}")
                else:
                    refined_idea, review_result = result
                    refined_ideas.append(refined_idea)
                    review_results.append(review_result)
            
            # Update context
            context.refined_ideas = refined_ideas
            
            context.logger.log_info(f"Parallel processing completed - {len(refined_ideas)}/{len(ideas_to_process)} ideas processed successfully")
            
            # Prepare review data
            review_data = {
                'ideas_processed': len(ideas_to_process),
                'ideas_refined': len(refined_ideas),
                'success_rate': len(refined_ideas) / len(ideas_to_process) if ideas_to_process else 0,
                'review_results': review_results,
                'refined_ideas': refined_ideas,  # Full refined idea objects
                'processing_summary': f"{len(refined_ideas)}/{len(ideas_to_process)} ideas successfully processed"
            }
            
            return PipelineResult(
                success=True,
                data=review_data,
                metadata={
                    'review_strategy': 'dual_agent_parallel',
                    'agents_used': ['ReviewerAgent', 'NoveltyAgent'],
                    'ideas_processed': len(ideas_to_process),
                    'success_rate': len(refined_ideas) / len(ideas_to_process) if ideas_to_process else 0
                }
            )
            
        except Exception as e:
            context.logger.log_error(f"Detailed review failed: {str(e)}", "DetailedReviewPipeline", e)
            return PipelineResult(
                success=False,
                data={
                    'ideas_processed': len(context.filtered_ideas) if context.filtered_ideas else 0,
                    'ideas_refined': 0,
                    'review_results': [],
                    'refined_ideas': [],
                    'processing_summary': 'Detailed review failed'
                },
                metadata={'review_strategy': 'multi_agent_parallel'},
                error_message=f"Detailed review error: {str(e)}"
            )
    
    async def _process_single_idea(self, idea: ResearchIdea, idea_index: int, context: PipelineContext) -> tuple:
        """
        Process a single idea through all review agents in parallel
        
        Args:
            idea: Research idea to process
            idea_index: Index of the idea for logging
            context: Pipeline context
            
        Returns:
            tuple: (refined_idea, review_result)
        """
        print(f"\nProcessing Idea {idea_index}: {idea.topic}")
        context.logger.log_info(f"Processing idea {idea_index}: {idea.topic}")
        
        idea_start_time = asyncio.get_event_loop().time()
        
        try:
            # Run review and novelty check in parallel
            context.logger.log_info(f"Starting parallel processing for idea {idea_index}", "parallel_processor")

            # Execute both agent operations concurrently
            review_task = self.reviewer_agent.review(idea)
            novelty_task = self.novelty_agent.check_novelty(idea)

            # Wait for both tasks to complete
            review_feedback, novelty_check = await asyncio.gather(
                review_task, novelty_task,
                return_exceptions=True
            )
            
            # Handle any exceptions
            if isinstance(review_feedback, Exception):
                context.logger.log_error(f"Review failed for idea {idea_index}: {review_feedback}", "reviewer_agent")
                review_feedback = {'overall_score': 0, 'error': str(review_feedback)}
            
            if isinstance(novelty_check, Exception):
                context.logger.log_error(f"Novelty check failed for idea {idea_index}: {novelty_check}", "novelty_agent")
                novelty_check = {'overall_novelty_score': 0, 'error': str(novelty_check)}

            # No refinement needed - use original idea
            refined_idea = idea
            
            # Extract scores and log results
            review_score = review_feedback.get('overall_score', 0)
            novelty_score = novelty_check.get('overall_novelty_score', 0)
            
            context.logger.log_info(f"Parallel processing completed for idea {idea_index} - Review: {review_score}, Novelty: {novelty_score}", "parallel_processor")
            print(f"  Review Score: {review_score:.2f}/5")
            print(f"  Novelty Score: {novelty_score:.2f}/5")
            
            # Aggregate feedback
            context.logger.log_info(f"Aggregating feedback for idea {idea_index}", "aggregator")
            self.aggregator.collect_feedback(idea, review_feedback, novelty_check)
            
            idea_time = asyncio.get_event_loop().time() - idea_start_time
            context.logger.log_performance_metric("main_system", f"idea_{idea_index}_processing_time", idea_time, "seconds")
            
            print(f"  Idea reviewed and analyzed ({idea_time:.2f}s)")
            
            # Prepare review result
            review_result = {
                'idea_id': idea_index,
                'topic': idea.topic,
                'review_score': review_score,
                'novelty_score': novelty_score,
                'processing_time': idea_time,
                'review_feedback': review_feedback,
                'novelty_check': novelty_check,
                'refined': False  # No refinement performed
            }
            
            return refined_idea, review_result
            
        except Exception as e:
            context.logger.log_error(f"Error processing idea {idea_index}: {str(e)}", "DetailedReviewPipeline", e)
            raise e
    
    def validate_dependencies(self, context: PipelineContext) -> bool:
        """
        Validate that we have filtered ideas for detailed review
        """
        if not context.filtered_ideas:
            context.logger.log_error("No filtered ideas available for detailed review", "DetailedReviewPipeline")
            return False
        
        return True