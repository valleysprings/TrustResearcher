#!/usr/bin/env python3
"""
TrustResearcher Main Module

Consolidated orchestration of the research idea generation pipeline.
All pipeline stages are executed directly from this module.
"""

import sys
import asyncio
import argparse
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass

from .agents.ideagen_agent import IdeaGenAgent
from .agents.reviewer_agent import ReviewerAgent
from .agents.retrieval_agent import RetrievalAgent
from .agents.selection_agent import SelectionAgent
from .knowledge_graph.kg_ops import KGOps
from .utils.config import load_config
from .utils.debug_logger import init_debug_logger, DebugLogger
from .utils.phase_timer import PhaseTimer
from .utils.pregen_validation import run_pregen_validation
from .utils.token_cost_tracker import get_current_token_count


@dataclass
class ResearchContext:
    """Shared context for research pipeline execution"""
    topic: str
    num_ideas: int
    config: Dict[str, Any]
    logger: DebugLogger
    timer: PhaseTimer

    # Pipeline data
    relevant_papers: List = None
    ideas: List = None
    selected_ideas: List = None
    filtered_ideas: List = None
    refined_ideas: List = None
    final_ideas: List = None
    literature_similarity_results: Dict = None
    portfolio_analysis: Dict = None

    def __post_init__(self):
        """Initialize empty lists"""
        if self.relevant_papers is None:
            self.relevant_papers = []
        if self.ideas is None:
            self.ideas = []
        if self.selected_ideas is None:
            self.selected_ideas = []
        if self.filtered_ideas is None:
            self.filtered_ideas = []
        if self.refined_ideas is None:
            self.refined_ideas = []
        if self.final_ideas is None:
            self.final_ideas = []


class ResearchOrchestrator:
    """Main orchestrator for research pipeline execution"""

    def __init__(self, config: Dict[str, Any], logger: DebugLogger, timer: PhaseTimer):
        self.config = config
        self.logger = logger
        self.timer = timer
        self.agents = {}
        self._setup_complete = False

    async def setup_agents(self) -> bool:
        """Initialize all research agents"""
        try:
            self.logger.log_info("Initializing research agents")

            llm_config = self.config['llm']

            # Initialize knowledge graph builder
            kg = KGOps(
                config=self.config,
                llm_config=llm_config,
                logger=self.logger
            )

            # Initialize agents
            self.agents = {
                'retrieval': RetrievalAgent(
                    api_key=self.config['semantic_scholar'].get('api_key'),
                    retrieval_config=self.config['semantic_scholar'],
                    logger=self.logger,
                    llm_config=llm_config
                ),
                'selection': SelectionAgent(
                    selection_config=self.config['external_selector'],
                    logger=self.logger,
                    llm_config=llm_config
                ),
                'ideagen': IdeaGenAgent(
                    kg,
                    idea_generation_config=self.config['idea_generation'],
                    planning_config=self.config['planning_module'],
                    logger=self.logger,
                    llm_config=llm_config
                ),
                'reviewer': ReviewerAgent(
                    reviewer_config=self.config['reviewer'],
                    logger=self.logger,
                    llm_config=llm_config
                )
            }

            # Setup cost tracking
            self._setup_cost_tracking()

            self.logger.log_info("All agents initialized successfully")
            print("All agents initialized")

            self._setup_complete = True
            return True

        except Exception as e:
            self.logger.log_error(f"Failed to setup agents: {str(e)}", "ResearchOrchestrator", e)
            return False

    def _setup_cost_tracking(self):
        """Setup cost tracking from agent LLM interfaces"""
        try:
            cost_tracker = None
            for _, agent in self.agents.items():
                if hasattr(agent, 'llm') and agent.llm and hasattr(agent.llm, 'cost_tracker'):
                    cost_tracker = agent.llm.cost_tracker
                    break

            if cost_tracker:
                self.timer.cost_tracker = cost_tracker
                self.logger.log_info("Cost tracking setup complete")
            else:
                self.logger.log_warning("No cost tracker found in agents")
        except Exception as e:
            self.logger.log_error(f"Failed to setup cost tracking: {e}", "ResearchOrchestrator")

    async def run_validation(self, ctx: ResearchContext) -> bool:
        """Phase 0: Validate external services"""
        try:
            print("Phase 0: Validating External Services")
            print("-" * 40)

            with ctx.timer.time_phase("validation", "System Validation"):
                success, validation_results = run_pregen_validation(ctx.config, ctx.logger)

                total_tests = len(validation_results)
                passed_tests = sum(1 for r in validation_results if r.success)

                if success:
                    print(f"All external services validated ({passed_tests}/{total_tests} tests passed)")
                else:
                    print(f"External service validation issues ({passed_tests}/{total_tests} tests passed)")
                    failed_tests = [r for r in validation_results if not r.success]
                    print(f"\nFailed tests ({len(failed_tests)}):")
                    for failure in failed_tests:
                        criticality = "CRITICAL" if failure.service in ['LLM', 'Configuration'] else "NON-CRITICAL"
                        print(f"   [{criticality}] {failure.service} - {failure.test_name}: {failure.message}")

                    critical_failures = [r for r in validation_results if not r.success and
                                       r.service in ['LLM', 'Configuration']]
                    if critical_failures:
                        print(f"\n⚠️  {len(critical_failures)} critical service failure(s) detected - workflow may fail")
                        return False

            return True

        except Exception as e:
            ctx.logger.log_error(f"Validation failed: {str(e)}", "validation", e)
            return False

    async def run_retrieval(self, ctx: ResearchContext) -> bool:
        """Phase 1: Search relevant literature"""
        try:
            print("\nPhase 1: Searching Relevant Literature")
            print("-" * 40)

            with ctx.timer.time_phase("retrieval", "Literature Search"):
                num_papers = ctx.config['semantic_scholar']['num_papers']
                ctx.relevant_papers = await self.agents['retrieval'].search_papers_by_topic(
                    ctx.topic,
                    num_papers=num_papers
                )

                print(f"Found {len(ctx.relevant_papers)} relevant papers")

                if ctx.relevant_papers:
                    print("Top papers found:")
                    for i, paper in enumerate(ctx.relevant_papers[:3], 1):
                        print(f"  {i}. {paper.title} ({paper.year}) - Citations: {paper.citation_count}")

            return True

        except Exception as e:
            ctx.logger.log_error(f"Literature search failed: {str(e)}", "retrieval", e)
            ctx.logger.log_warning("Continuing without literature context")
            return True  # Non-critical failure

    async def run_ideation(self, ctx: ResearchContext) -> bool:
        """Phase 2: Generate research ideas"""
        try:
            print("\nPhase 2: Generating Research Ideas (Literature-Informed)")
            print("-" * 40)

            with ctx.timer.time_phase("ideation", "Idea Generation"):
                overgenerate_factor = ctx.config['idea_generation']['overgeneration_factor']

                ctx.ideas = await self.agents['ideagen'].generate_ideas(
                    seed_topic=ctx.topic,
                    num_ideas=ctx.num_ideas,
                    literature_context=ctx.relevant_papers
                )

                print(f"Generated {len(ctx.ideas)} literature-informed research ideas "
                      f"({overgenerate_factor}x overgeneration for robust filtering)")

                ctx.logger.log_info(f"Generated {len(ctx.ideas)} ideas with topics: {[idea.topic for idea in ctx.ideas[:3]]}")

            return True

        except Exception as e:
            ctx.logger.log_error(f"Idea generation failed: {str(e)}", "ideation", e)
            return False

    async def run_selection(self, ctx: ResearchContext) -> bool:
        """Phase 3: Preliminary idea selection"""
        try:
            print("\nPhase 3: Preliminary Idea Selection")
            print("-" * 40)

            with ctx.timer.time_phase("selection", "Idea Selection"):
                # External selection: filter against literature FIRST
                print("\n[External] Comparing ideas against literature...")
                ctx.selected_ideas, results = self.agents['selection'].filter_against_literature(
                    ctx.ideas,
                    ctx.relevant_papers or []
                )

                # Generate report
                ctx.literature_similarity_results = self.agents['selection'].generate_selection_report(results)
                print(f"[External] {len(ctx.selected_ideas)}/{len(ctx.ideas)} ideas sufficiently novel")

                # Internal selection: dedupe/merge similar ideas SECOND
                print("\n[Internal] Merging similar ideas...")
                config = ctx.config['idea_generation']
                target_count = ctx.num_ideas * config['intermediate_selection_multiplier']

                ctx.filtered_ideas = await self.agents['selection'].select_diverse_ideas(
                    ctx.selected_ideas,
                    target_count=int(target_count),
                    merge_similar=True
                )
                print(f"[Internal] {len(ctx.filtered_ideas)}/{len(ctx.selected_ideas)} ideas after merge")

            return True

        except Exception as e:
            ctx.logger.log_error(f"Selection failed: {str(e)}", "selection", e)
            return False

    async def run_review(self, ctx: ResearchContext) -> bool:
        """Phase 4: Detailed review and critique"""
        try:
            print("\nPhase 4: Detailed Review and Critique")
            print("-" * 40)

            if not ctx.filtered_ideas:
                ctx.logger.log_error("No filtered ideas available for review", "review")
                return False

            with ctx.timer.time_phase("review", "Detailed Review"):
                print(f"Processing {len(ctx.filtered_ideas)} ideas in parallel...")

                # Process all ideas in parallel
                tasks = [
                    self._review_single_idea(idea, i+1, ctx)
                    for i, idea in enumerate(ctx.filtered_ideas)
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Collect successful results
                ctx.refined_ideas = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        ctx.logger.log_error(f"Failed to process idea {i+1}: {result}", "review")
                    else:
                        ctx.refined_ideas.append(result)

                ctx.logger.log_info(f"Review completed - {len(ctx.refined_ideas)}/{len(ctx.filtered_ideas)} ideas processed")

            return True

        except Exception as e:
            ctx.logger.log_error(f"Review failed: {str(e)}", "review", e)
            return False

    async def _review_single_idea(self, idea, idea_index: int, ctx: ResearchContext):
        """Process a single idea through two-stage review"""
        print(f"\nProcessing Idea {idea_index}: {idea.topic}")
        ctx.logger.log_info(f"Processing idea {idea_index}: {idea.topic}")

        start_time = asyncio.get_event_loop().time()
        tokens_before = get_current_token_count()

        try:
            review_result = await self.agents['reviewer'].review(idea)

            if isinstance(review_result, Exception):
                ctx.logger.log_error(f"Review failed for idea {idea_index}: {review_result}", "reviewer")
                review_result = {'overall_assessment': {'overall_score': None}, 'error': str(review_result)}

            # Extract overall assessment
            overall_assessment = review_result.get('overall_assessment', {})
            review_score = overall_assessment.get('overall_score')

            # Extract Stage 1 info
            stage1 = review_result.get('stage1_preliminary', {})
            stage1_score = stage1.get('overall_score') if stage1 else None

            # Extract Stage 2 info
            stage2 = review_result.get('stage2_detailed', {})
            stage2_score = stage2.get('overall_detailed_score') if stage2 else None

            tokens_after = get_current_token_count()
            tokens_used = tokens_after - tokens_before

            # Log detailed review info
            ctx.logger.log_info(f"Review completed for idea {idea_index} - Overall: {review_score}, Stage1: {stage1_score}, Stage2: {stage2_score}")

            # Print review summary
            if review_score is not None:
                print(f"  Overall Score: {review_score:.2f}/5")
            else:
                print(f"  Overall Score: N/A")

            if stage1_score is not None:
                print(f"  Stage 1 (Preliminary): {stage1_score:.2f}/5")
            if stage2_score is not None:
                print(f"  Stage 2 (Detailed): {stage2_score:.2f}/5")

            # Show criterion scores if available
            if stage2 and 'criterion_reviews' in stage2:
                criterion_scores = []
                for criterion, review in stage2['criterion_reviews'].items():
                    score = review.get('score')
                    if score is not None:
                        criterion_scores.append(f"{criterion.capitalize()}: {score:.2f}")
                if criterion_scores:
                    print(f"  Criteria: {', '.join(criterion_scores)}")

            # Attach review feedback to idea
            if not hasattr(idea, 'review_feedback'):
                idea.review_feedback = []

            idea.review_feedback.append({
                'type': 'two_stage_review',
                'overall_score': review_score,
                'stage1_score': stage1_score,
                'stage2_score': stage2_score,
                'overall_assessment': overall_assessment,
                'stage1_preliminary': stage1,
                'stage2_detailed': stage2
            })

            return idea

        except Exception as e:
            ctx.logger.log_error(f"Error processing idea {idea_index}: {str(e)}", "review", e)
            raise e

    async def run_final_selection(self, ctx: ResearchContext) -> bool:
        """Phase 5: Final idea selection"""
        try:
            print("\nPhase 5: Final Idea Selection")
            print("-" * 40)

            if not ctx.refined_ideas:
                ctx.logger.log_error("No refined ideas available for final selection", "final_selection")
                return False

            with ctx.timer.time_phase("final_selection", "Final Selection"):
                # Score ideas from review feedback
                scored_ideas = self._score_ideas(ctx.refined_ideas)

                # Separate ideas with scores from those without
                ideas_with_scores = [(idea, score) for idea, score in scored_ideas if score is not None]
                ideas_without_scores = [(idea, score) for idea, score in scored_ideas if score is None]

                # Sort ideas with scores
                ideas_with_scores.sort(key=lambda x: x[1], reverse=True)

                # Combine: scored ideas first, then unscored
                scored_ideas = ideas_with_scores + ideas_without_scores

                # Filter for good ideas (using minor_revise threshold from config)
                good_threshold = ctx.config['reviewer']['aggregation']['thresholds']['minor_revise']
                good_ideas = [(idea, score) for idea, score in ideas_with_scores if score >= good_threshold]

                print(f"Found {len(good_ideas)} 'good' ideas (>={good_threshold}) out of {len(ideas_with_scores)} scored ideas")
                if ideas_without_scores:
                    print(f"Warning: {len(ideas_without_scores)} ideas have no scores")
                print(f"User requested: {ctx.num_ideas} ideas")

                # Simple selection: output all good ideas if enough, otherwise top num_ideas
                if len(good_ideas) >= ctx.num_ideas:
                    ctx.final_ideas = [idea for idea, _ in good_ideas]
                    print(f"Selected ALL {len(good_ideas)} good ideas (threshold met)")
                else:
                    ctx.final_ideas = [idea for idea, _ in scored_ideas[:ctx.num_ideas]]
                    print(f"Selected top {ctx.num_ideas} ideas (only {len(good_ideas)} met threshold)")

                # Display selected ideas
                print("Selected ideas:")
                thresholds = ctx.config['reviewer']['aggregation']['thresholds']
                for i, idea in enumerate(ctx.final_ideas, 1):
                    idea_score = next((s for ide, s in scored_ideas if ide == idea), None)
                    if idea_score is not None:
                        quality = "Good" if idea_score >= thresholds['minor_revise'] else "Moderate" if idea_score >= thresholds['major_revise'] else "Weak"
                        print(f"  {i}. {idea.topic} (Score: {idea_score:.2f}, Quality: {quality})")
                    else:
                        print(f"  {i}. {idea.topic} (Score: N/A, Quality: Unknown)")

            return True

        except Exception as e:
            ctx.logger.log_error(f"Final selection failed: {str(e)}", "final_selection", e)
            return False

    def _score_ideas(self, ideas: List) -> List[tuple]:
        """Extract scores from ideas' review feedback"""
        scored = []
        for idea in ideas:
            score = self._get_idea_score(idea)
            scored.append((idea, score))
        return scored

    def _get_idea_score(self, idea) -> Optional[float]:
        """Extract overall score from idea's review feedback"""
        if not hasattr(idea, 'review_feedback') or not idea.review_feedback:
            return None

        for feedback in idea.review_feedback:
            if isinstance(feedback, dict):
                if feedback.get('type') == 'two_stage_review':
                    score = feedback.get('overall_score')
                    return float(score) if score is not None else None
                if 'overall_score' in feedback:
                    score = feedback.get('overall_score')
                    return float(score) if score is not None else None

        return None

    def _extract_review_data(self, idea) -> Dict[str, Any]:
        """Extract review data from idea's attached feedback (Stage 2 only for output)"""
        result = {
            'review_feedback': {},
            'overall_score': None,
            'stage1_preliminary': None,
            'stage2_detailed': None
        }

        if not hasattr(idea, 'review_feedback') or not idea.review_feedback:
            return result

        for feedback in idea.review_feedback:
            if isinstance(feedback, dict):
                if feedback.get('type') == 'two_stage_review':
                    score = feedback.get('overall_score')
                    result['overall_score'] = float(score) if score is not None else None

                    # Store both stages internally (for debugging/analysis)
                    result['stage1_preliminary'] = feedback.get('stage1_preliminary')
                    result['stage2_detailed'] = feedback.get('stage2_detailed')

                    # Output only Stage 2 aggregated feedback in final JSON
                    result['review_feedback'] = {
                        'overall_score': result['overall_score'],
                        'key_strengths': feedback.get('key_strengths', []),
                        'key_weaknesses': feedback.get('key_weaknesses', []),
                        'actionable_suggestions': feedback.get('actionable_suggestions', [])
                    }
                    break

        return result

    def _generate_final_output(self, ctx: ResearchContext) -> Dict[str, Any]:
        """Generate final JSON output from research context"""
        # Build ideas JSON data
        ideas_json_data = []
        for i, idea in enumerate(ctx.final_ideas, 1):
            review_data = self._extract_review_data(idea)
            ideas_json_data.append({
                "id": i,
                "topic": idea.topic,
                "problem_statement": idea.problem_statement,
                "proposed_methodology": idea.proposed_methodology,
                "experimental_validation": idea.experimental_validation,
                "review_feedback": review_data.get('review_feedback', {}),
                "overall_score": review_data.get('overall_score', 0)
            })

        # Generate comprehensive JSON output
        return self._generate_json_output(ideas_json_data, ctx)

    def _generate_json_output(self, ideas_json_data: List, ctx: ResearchContext) -> Dict:
        """Generate comprehensive JSON output"""
        return {
            "research_ideas": ideas_json_data,
            "literature_search": {
                "relevant_papers_found": len(ctx.relevant_papers) if ctx.relevant_papers else 0,
                "top_papers": [
                    {
                        "title": paper.title,
                        "authors": paper.authors,
                        "year": paper.year,
                        "citation_count": paper.citation_count,
                        "url": paper.url
                    }
                    for paper in (ctx.relevant_papers or [])
                ],
            },
            "literature_similarity_analysis": ctx.literature_similarity_results or {},
            "generation_metadata": {
                "seed_topic": ctx.topic,
                "requested_ideas": ctx.num_ideas,
                "generated_ideas": len(ctx.ideas) if ctx.ideas else 0,
                "selected_for_review": len(ctx.selected_ideas) if ctx.selected_ideas else 0,
                "literature_filtered_ideas": len(ctx.filtered_ideas) if ctx.filtered_ideas else 0,
                "final_processed_ideas": len(ctx.final_ideas),
                "execution_time_seconds": round(ctx.timer.get_total_time(), 2),
                "timestamp": ctx.logger.session_id
            }
        }

    async def execute(self, topic: str, num_ideas: int, run_validation: bool = True) -> Dict[str, Any]:
        """Execute the complete research pipeline"""
        try:
            # Initialize context
            ctx = ResearchContext(
                topic=topic,
                num_ideas=num_ideas,
                config=self.config,
                logger=self.logger,
                timer=self.timer
            )

            self.logger.log_info(f"Starting research pipeline for topic: {topic}")

            # Phase 0: Validation (optional)
            if run_validation:
                if not await self.run_validation(ctx):
                    return {'success': False, 'error': 'Validation failed'}

            # Phase 1: Literature retrieval
            if not await self.run_retrieval(ctx):
                self.logger.log_warning("Literature retrieval failed, continuing without literature")

            # Phase 2: Idea generation
            if not await self.run_ideation(ctx):
                return {'success': False, 'error': 'Idea generation failed'}

            # Phase 3: Preliminary selection
            if not await self.run_selection(ctx):
                return {'success': False, 'error': 'Idea selection failed'}

            # Phase 4: Detailed review
            if not await self.run_review(ctx):
                return {'success': False, 'error': 'Review failed'}

            # Phase 5: Final selection
            if not await self.run_final_selection(ctx):
                return {'success': False, 'error': 'Final selection failed'}

            # Generate and save output
            json_output = self._generate_final_output(ctx)
            output_path = self._generate_output_path(topic, ctx.logger.session_id)
            self._save_results(output_path, json_output)

            self.logger.log_info("Research pipeline completed successfully")

            return {
                'success': True,
                'output_path': str(output_path),
                'results': json_output,
                'context': ctx
            }

        except Exception as e:
            self.logger.log_error(f"Pipeline execution failed: {str(e)}", "execute", e)
            return {'success': False, 'error': str(e)}

    def _generate_output_path(self, topic: str, session_id: str) -> Path:
        """Generate output file path"""
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)

        # Use only timestamp as filename (topic is already in the JSON content)
        filename = f"{session_id}.json"

        return outputs_dir / filename

    def _save_results(self, output_path: Path, results: Dict[str, Any]):
        """Save results to JSON file"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.log_info(f"Saving results to {output_path}")

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            self.logger.log_info(f"Results successfully saved to {output_path}")
            print(f"Results saved to {output_path}")

        except Exception as e:
            self.logger.log_error(f"Failed to save results: {e}", "save_results", e)

    async def cleanup(self):
        """Clean up all agents and resources"""
        if not self._setup_complete:
            return

        self.logger.log_info("Starting cleanup of all agents and resources")

        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'llm') and hasattr(agent.llm, 'close_session'):
                try:
                    await agent.llm.close_session()
                    self.logger.log_info(f"Closed session for {agent_name}")
                except Exception as e:
                    self.logger.log_warning(f"Failed to close session for {agent_name}: {e}")

        # Wait for all pending async tasks to complete (event-driven)
        current_task = asyncio.current_task()
        pending_tasks = [task for task in asyncio.all_tasks() if task is not current_task]
        if pending_tasks:
            self.logger.log_info(f"Waiting for {len(pending_tasks)} pending tasks to complete")
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        self.logger.log_info("Cleanup completed")


async def main():
    """Main entry point for TrustResearcher"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='TrustResearcher - Consolidated Pipeline')
    parser.add_argument('--topic', type=str, required=False,
                       help='Seed topic for research idea generation')
    parser.add_argument('--num_ideas', type=int, default=3,
                       help='Number of research ideas to generate')
    parser.add_argument('--config', type=str, default='configs/',
                       help='Configuration directory path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save results')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with comprehensive logging')
    parser.add_argument('--validate', action='store_true',
                       help='Run component validation before main execution')
    parser.add_argument('--pregen-check', action='store_true',
                       help='Run pre-generation validation tests and exit')
    parser.add_argument('--skip-pregen', action='store_true',
                       help='Skip pre-generation validation (not recommended)')

    args = parser.parse_args()

    # Handle pregen-check mode (validation only)
    if args.pregen_check:
        from .utils.pregen_validation import print_validation_report

        print("Running Pre-Generation Validation Tests")
        print("=" * 50)

        config = load_config(args.config)
        logger = init_debug_logger(debug_mode=True, topic="validation_check")

        success, validation_results = run_pregen_validation(config, logger)
        print_validation_report(validation_results)

        sys.exit(0 if success else 1)

    # Validate required arguments
    if not args.topic:
        print("Error: --topic is required for research idea generation")
        parser.print_help()
        sys.exit(1)

    # Initialize debug logger and timing
    logger = init_debug_logger(debug_mode=args.debug, topic=args.topic)
    timer = PhaseTimer(logger)

    # Reset global token statistics
    from .utils.token_cost_tracker import reset_global_session_stats
    reset_global_session_stats()

    # Print startup information
    print("TrustResearcher - Consolidated Pipeline Architecture")
    print("=" * 60)
    if args.debug:
        print("DEBUG MODE ENABLED - Comprehensive logging active")
        print(f"Logs saved to: logs/{{session,idea,kg,llm}}/")

    print(f"Topic: {args.topic}")
    print(f"Generating {args.num_ideas} research ideas")
    print("=" * 60)

    logger.log_info(f"Starting research pipeline for topic: {args.topic}")

    try:
        # Load configuration
        logger.log_info("Loading configuration")
        config = load_config(args.config)
        logger.log_info(f"Configuration loaded with {len(config)} sections")
        print("Configuration loaded")

        # Initialize orchestrator
        logger.log_info("Initializing research orchestrator")
        orchestrator = ResearchOrchestrator(config, logger, timer)

        # Setup agents
        print("Initializing research agents...")
        setup_success = await orchestrator.setup_agents()
        if not setup_success:
            print("Failed to initialize research agents")
            sys.exit(1)

        # Execute pipeline
        print("\nStarting Research Pipeline")
        print("=" * 60)

        result = await orchestrator.execute(
            topic=args.topic,
            num_ideas=args.num_ideas,
            run_validation=not args.skip_pregen
        )

        if result['success']:
            # Print summary
            total_time = timer.get_total_time()
            timer.print_performance_summary()

            session_summary = logger.get_session_summary()
            print(f"\nSession Summary")
            print("-" * 40)
            print(f"Total Execution Time: {total_time:.2f} seconds")
            print(f"LLM Conversations: {sum(logger.llm_counters.values())}")
            print(f"Output File: {result['output_path']}")

            if args.debug:
                print(f"\nDebug Information:")
                print(f"Session ID: {session_summary['session_id']}")
                print(f"Main Log File: {session_summary['log_files']['main_log']}")
                print(f"LLM Conversation File: {session_summary['log_files']['llm_log']}")

            logger.finalize_session()
            logger.log_info(f"Research pipeline completed successfully in {total_time:.2f} seconds")
            print(f"\nTrustResearcher completed successfully!")

        else:
            print(f"\nPipeline execution failed: {result['error']}")
            logger.log_error(f"Pipeline execution failed: {result['error']}")
            sys.exit(1)

        # Cleanup
        await orchestrator.cleanup()
        await asyncio.sleep(0.2)  # Give sessions time to close properly

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        logger.log_info("Execution interrupted by user")
        try:
            if 'orchestrator' in locals():
                await orchestrator.cleanup()
                await asyncio.sleep(0.2)  # Give sessions time to close properly
        except:
            pass
        sys.exit(1)

    except Exception as e:
        logger.log_error(f"Fatal error in main execution: {str(e)}", "main_system", e)
        print(f"Error: {str(e)}")

        try:
            if 'orchestrator' in locals():
                await orchestrator.cleanup()
                await asyncio.sleep(0.2)  # Give sessions time to close properly
        except:
            pass

        if args.debug:
            import traceback
            traceback.print_exc()

        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

