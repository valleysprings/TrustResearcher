"""
Research Pipeline Orchestrator

Main orchestrator that coordinates the execution of all research pipelines
in the correct order with proper dependency management and error handling.
"""

import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path

from .base_pipeline import BasePipeline, PipelineResult, PipelineContext, PipelineExecutor
from .validation_pipeline import ValidationPipeline
from .literature_search_pipeline import LiteratureSearchPipeline
from .idea_generation_pipeline import IdeaGenerationPipeline
from .internal_selection_pipeline import InternalSelectionPipeline
from .external_selection_pipeline import ExternalSelectionPipeline
from .detailed_review_pipeline import DetailedReviewPipeline
from .final_selection_pipeline import FinalSelectionPipeline
from .portfolio_analysis_pipeline import PortfolioAnalysisPipeline

from ..agents.idea_generator import IdeaGenerator
from ..agents.reviewer_agent import ReviewerAgent
from ..agents.novelty_agent import NoveltyAgent
from ..agents.aggregator import Aggregator
from ..agents.semantic_scholar_agent import SemanticScholarAgent
from ..agents.external_selector import ExternalSelector
from ..agents.internal_selector import InternalSelector
from ..utils.debug_logger import DebugLogger
from ..utils.phase_timer import PhaseTimer


class ResearchPipelineOrchestrator:
    """
    Orchestrator for the complete research pipeline execution
    """
    
    def __init__(self, config: Dict[str, Any], logger: DebugLogger, timer: PhaseTimer):
        """
        Initialize the research pipeline orchestrator
        
        Args:
            config: System configuration dictionary
            logger: Debug logger instance
            timer: Phase timer for performance tracking
        """
        self.config = config
        self.logger = logger
        self.timer = timer
        
        # Will be initialized in setup_agents
        self.agents = {}
        self.pipelines = []
        self._setup_complete = False
    
    async def setup_agents(self) -> bool:
        """
        Initialize all agents required for the research pipeline
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            self.logger.log_info("Initializing agents for pipeline orchestrator")

            # Get LLM configuration
            llm_config = self.config.get('llm', {})
            
            # Initialize knowledge graph builder first
            from ..knowledge_graph.kg_builder import KGBuilder
            kg_builder = KGBuilder(self.config.get('knowledge_graph', {}), llm_config=llm_config, logger=self.logger)
            
            # Initialize all agents
            self.agents = {
                'semantic_scholar_agent': SemanticScholarAgent(
                    api_key=self.config.get('semantic_scholar', {}).get('api_key'),
                    config=self.config.get('semantic_scholar', {}),
                    logger=self.logger,
                    llm_config=llm_config
                ),
                'external_selector': ExternalSelector(
                    config=self.config.get('external_selector', {}),
                    logger=self.logger
                ),
                'internal_selector': InternalSelector(
                    config=self.config.get('internal_selector', {}),
                    logger=self.logger,
                    llm_config=llm_config
                ),
                'idea_generator': IdeaGenerator(
                    kg_builder,
                    self.config,  # Pass full config instead of just idea_generator section
                    logger=self.logger,
                    llm_config=llm_config
                ),
                'reviewer_agent': ReviewerAgent(
                    self.config.get('reviewer_agent', {}),
                    logger=self.logger,
                    llm_config=llm_config
                ),
                'novelty_agent': NoveltyAgent(
                    self.config.get('novelty_agent', {}),
                    logger=self.logger,
                    llm_config=llm_config
                ),
                'aggregator': Aggregator()
            }
            
            self.logger.log_info("All agents initialized successfully")
            print("All agents initialized")

            # Collect cost trackers from agents with LLM interfaces
            self._setup_cost_tracking()

            self._setup_complete = True
            return True
            
        except Exception as e:
            self.logger.log_error(f"Failed to setup agents: {str(e)}", "ResearchPipelineOrchestrator", e)
            return False

    def _setup_cost_tracking(self):
        """Setup cost tracking by finding any agent with a cost tracker"""
        try:
            # Find the first agent with an LLM interface to get a cost tracker
            # Since all agents use the same global tracking now, any tracker will work
            cost_tracker = None

            for agent_name, agent in self.agents.items():
                if hasattr(agent, 'llm') and agent.llm and hasattr(agent.llm, 'cost_tracker'):
                    cost_tracker = agent.llm.cost_tracker
                    break

            if cost_tracker:
                # Pass the cost tracker to the timer
                self.timer.cost_tracker = cost_tracker
                self.logger.log_info("Cost tracking setup complete", "ResearchPipelineOrchestrator")
            else:
                self.logger.log_warning("No cost tracker found in agents", "ResearchPipelineOrchestrator")

        except Exception as e:
            self.logger.log_error(f"Failed to setup cost tracking: {e}", "ResearchPipelineOrchestrator")

    
    def setup_pipelines(self) -> List[BasePipeline]:
        """
        Setup and configure all research pipelines
        
        Returns:
            List[BasePipeline]: List of configured pipelines in execution order
        """
        if not self._setup_complete:
            raise RuntimeError("Agents must be setup before configuring pipelines")
        
        # Configure pipelines in execution order
        self.pipelines = [
            ValidationPipeline(),
            
            LiteratureSearchPipeline(
                self.agents['semantic_scholar_agent']
            ),
            
            IdeaGenerationPipeline(
                self.agents['idea_generator']
            ),
            
            InternalSelectionPipeline(
                self.agents['internal_selector']
            ),
            
            ExternalSelectionPipeline(
                self.agents['external_selector']
            ),
            
            DetailedReviewPipeline(
                self.agents['reviewer_agent'],
                self.agents['novelty_agent'],
                self.agents['aggregator']
            ),
            
            FinalSelectionPipeline(
                self.agents['internal_selector'],
                self.agents['aggregator']
            ),
            
            PortfolioAnalysisPipeline(
                self.agents['aggregator']
            )
        ]
        
        return self.pipelines
    
    async def execute_research_pipeline(
        self, 
        topic: str, 
        num_ideas: int,
        run_validation: bool = True,
        skip_phases: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete research pipeline
        
        Args:
            topic: Research topic to explore
            num_ideas: Number of final ideas to generate
            run_validation: Whether to run validation pipeline
            skip_phases: List of phase names to skip (optional)
            
        Returns:
            Dict[str, Any]: Complete pipeline execution results
        """
        if not self._setup_complete:
            await self.setup_agents()
        
        if not self.pipelines:
            self.setup_pipelines()
        
        # Initialize pipeline context
        context = PipelineContext(
            topic=topic,
            num_ideas=num_ideas,
            config=self.config,
            logger=self.logger,
            timer=self.timer
        )
        
        # Track pipeline results
        pipeline_results = {}
        execution_summary = {
            'total_pipelines': len(self.pipelines),
            'executed_pipelines': 0,
            'successful_pipelines': 0,
            'failed_pipelines': 0,
            'skipped_pipelines': 0
        }
        
        skip_phases = skip_phases or []
        
        try:
            self.logger.log_info(f"Starting research pipeline execution for topic: {topic}")
            
            # Execute pipelines in sequence
            for i, pipeline in enumerate(self.pipelines):
                pipeline_name = pipeline.name.lower().replace(' ', '_')
                
                # Skip validation if requested
                if not run_validation and isinstance(pipeline, ValidationPipeline):
                    self.logger.log_info(f"Skipping validation pipeline as requested")
                    execution_summary['skipped_pipelines'] += 1
                    continue
                
                # Skip specific phases if requested
                if pipeline_name in skip_phases:
                    self.logger.log_info(f"Skipping pipeline {pipeline.name} as requested")
                    execution_summary['skipped_pipelines'] += 1
                    continue
                
                # Execute pipeline
                self.logger.log_info(f"Executing pipeline {i+1}/{len(self.pipelines)}: {pipeline.name}")
                
                result = await PipelineExecutor.execute_pipeline(
                    pipeline, 
                    context, 
                    f"phase{i+1}_{pipeline_name}"
                )
                
                pipeline_results[pipeline_name] = result
                execution_summary['executed_pipelines'] += 1
                
                if result.success:
                    execution_summary['successful_pipelines'] += 1
                    self.logger.log_info(f"Pipeline {pipeline.name} completed successfully")
                else:
                    execution_summary['failed_pipelines'] += 1
                    self.logger.log_error(f"Pipeline {pipeline.name} failed: {result.error_message}")
                    
                    # Handle critical failures
                    if isinstance(pipeline, ValidationPipeline) and not result.success:
                        # Validation failures are handled within the pipeline
                        pass
                    elif isinstance(pipeline, LiteratureSearchPipeline) and not result.success:
                        # Continue without literature if search fails
                        self.logger.log_warning("Continuing without literature context")
                    elif isinstance(pipeline, IdeaGenerationPipeline) and not result.success:
                        # Critical failure - cannot continue without ideas
                        raise RuntimeError(f"Critical pipeline failure: {pipeline.name}")
            
            # Generate final output
            final_output = await self._generate_final_output(context, pipeline_results)
            
            # Save results
            output_path = self._generate_output_filename(topic, context.logger.session_id)
            await self._save_results(output_path, final_output, context)
            
            self.logger.log_info("Research pipeline execution completed successfully")
            
            return {
                'success': True,
                'results': final_output,
                'output_path': str(output_path),
                'pipeline_results': pipeline_results,
                'execution_summary': execution_summary,
                'context': context
            }
            
        except Exception as e:
            self.logger.log_error(f"Research pipeline execution failed: {str(e)}", "ResearchPipelineOrchestrator", e)
            
            return {
                'success': False,
                'error': str(e),
                'pipeline_results': pipeline_results,
                'execution_summary': execution_summary,
                'context': context
            }
    
    async def _generate_final_output(self, context: PipelineContext, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate the final consolidated output from all pipeline results
        
        Args:
            context: Pipeline execution context
            pipeline_results: Results from all executed pipelines
            
        Returns:
            Dict[str, Any]: Final consolidated output
        """
        # Get portfolio analysis results if available
        portfolio_result = pipeline_results.get('portfolio_analysis')
        portfolio_data = portfolio_result.data if portfolio_result and portfolio_result.success else {}
        
        return portfolio_data.get('json_output', {
            'research_ideas': [],
            'portfolio_analysis': {},
            'literature_search': {'relevant_papers_found': 0, 'top_papers': []},
            'distinctness_analysis': {},
            'generation_metadata': {
                'seed_topic': context.topic,
                'requested_ideas': context.num_ideas,
                'execution_time_seconds': round(context.timer.get_total_time(), 2),
                'timestamp': context.logger.session_id
            }
        })
    
    def _generate_output_filename(self, topic: str, session_id: str, custom_output: str = None) -> Path:
        """
        Generate organized output filename with timestamp and topic
        
        Args:
            topic: Research topic
            session_id: Session identifier
            custom_output: Custom output filename (optional)
            
        Returns:
            Path: Output file path
        """
        # Create outputs directory
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        
        # Clean topic name for filename
        clean_topic = "".join(c if c.isalnum() else "_" for c in topic.lower())[:30]
        
        if custom_output:
            # If user specified output, use it but put it in outputs folder
            custom_name = Path(custom_output).stem
            filename = f"{clean_topic}_{session_id}_{custom_name}.json"
        else:
            # Generate standard filename
            filename = f"{clean_topic}_{session_id}.json"
        
        return outputs_dir / filename
    
    async def _save_results(self, output_path: Path, results: Dict[str, Any], context: PipelineContext):
        """
        Save the results to file
        
        Args:
            output_path: Path to save results
            results: Results dictionary to save
            context: Pipeline context
        """
        import json
        
        try:
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.logger.log_info(f"Saving results to {output_path}")
            
            # Save JSON results
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.log_info(f"Results successfully saved to {output_path}")
            
            print(f"Results saved to {output_path}")
        
        except Exception as e:
            self.logger.log_error(f"Failed to save results: {e}", "ResearchPipelineOrchestrator", e)
    
    async def cleanup(self):
        """
        Clean up all agents and resources
        """
        if not self._setup_complete:
            return
        
        self.logger.log_info("Starting cleanup of all agents and resources")
        
        # Clean up all LLM interface sessions
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'llm') and hasattr(agent.llm, 'close_session'):
                try:
                    await agent.llm.close_session()
                    self.logger.log_info(f"Closed session for {agent_name}")
                except Exception as e:
                    self.logger.log_warning(f"Failed to close session for {agent_name}: {e}")
        
        # Longer delay to ensure all async operations and connections fully close
        await asyncio.sleep(0.2)

        # Force garbage collection to ensure destructor calls
        import gc
        gc.collect()
        
        self.logger.log_info("Cleanup completed")
