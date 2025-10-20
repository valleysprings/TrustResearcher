"""
Base Pipeline Interface

Defines the core pipeline architecture for the research agent system.
All pipelines inherit from BasePipeline and implement the execute method.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import asyncio

from ..utils.debug_logger import DebugLogger
from ..utils.phase_timer import PhaseTimer


@dataclass
class PipelineResult:
    """
    Standard result format for all pipelines
    """
    success: bool
    data: Any
    metadata: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Add timestamp
        self.metadata['timestamp'] = datetime.now().isoformat()


@dataclass
class PipelineContext:
    """
    Shared context passed between pipelines
    """
    topic: str
    num_ideas: int
    config: Dict[str, Any]
    logger: DebugLogger
    timer: PhaseTimer
    
    # Data that accumulates across pipelines
    relevant_papers: List = None
    ideas: List = None
    selected_ideas: List = None
    filtered_ideas: List = None  # Ideas that pass external literature similarity filter
    refined_ideas: List = None
    final_ideas: List = None
    
    # Analysis results
    literature_similarity_results: Dict = None  # Results from external literature similarity analysis
    portfolio_analysis: Dict = None
    
    def __post_init__(self):
        """Initialize empty lists if not provided"""
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


class BasePipeline(ABC):
    """
    Base class for all research pipeline stages
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._dependencies = []
        self._outputs = []
    
    @abstractmethod
    async def execute(self, context: PipelineContext) -> PipelineResult:
        """
        Execute the pipeline stage
        
        Args:
            context: Shared pipeline context containing data and configuration
            
        Returns:
            PipelineResult: Result of pipeline execution
        """
        pass
    
    def add_dependency(self, pipeline_name: str):
        """Add a pipeline dependency"""
        if pipeline_name not in self._dependencies:
            self._dependencies.append(pipeline_name)
    
    def add_output(self, output_name: str):
        """Add an output that this pipeline produces"""
        if output_name not in self._outputs:
            self._outputs.append(output_name)
    
    @property
    def dependencies(self) -> List[str]:
        """Get pipeline dependencies"""
        return self._dependencies.copy()
    
    @property
    def outputs(self) -> List[str]:
        """Get pipeline outputs"""
        return self._outputs.copy()
    
    def validate_dependencies(self, context: PipelineContext) -> bool:
        """
        Validate that all dependencies are satisfied in the context
        
        Args:
            context: Pipeline context to validate
            
        Returns:
            bool: True if all dependencies are satisfied
        """
        # This is a basic implementation - subclasses can override for specific validation
        return True
    
    async def pre_execute(self, context: PipelineContext) -> bool:
        """
        Pre-execution hook for validation and setup
        
        Args:
            context: Pipeline context
            
        Returns:
            bool: True if ready to execute, False to skip
        """
        if not self.validate_dependencies(context):
            context.logger.log_error(
                f"Pipeline {self.name} dependencies not satisfied",
                self.__class__.__name__
            )
            return False
        
        context.logger.log_info(f"Starting pipeline: {self.name}", self.__class__.__name__)
        return True
    
    async def post_execute(self, context: PipelineContext, result: PipelineResult):
        """
        Post-execution hook for cleanup and logging
        
        Args:
            context: Pipeline context
            result: Pipeline execution result
        """
        status = "SUCCESS" if result.success else "FAILED"
        context.logger.log_info(
            f"Pipeline {self.name} {status} in {result.execution_time:.2f}s",
            self.__class__.__name__
        )
        
        if not result.success and result.error_message:
            context.logger.log_error(
                f"Pipeline {self.name} error: {result.error_message}",
                self.__class__.__name__
            )


class PipelineExecutor:
    """
    Utility class for executing pipelines with timing and error handling
    """
    
    @staticmethod
    async def execute_pipeline(
        pipeline: BasePipeline,
        context: PipelineContext,
        phase_name: str = None
    ) -> PipelineResult:
        """
        Execute a pipeline with full timing and error handling
        
        Args:
            pipeline: Pipeline to execute
            context: Pipeline context
            phase_name: Optional phase name for timing (defaults to pipeline name)
            
        Returns:
            PipelineResult: Result of pipeline execution
        """
        phase_name = phase_name or pipeline.name.lower().replace(' ', '_')
        
        try:
            # Pre-execution validation
            if not await pipeline.pre_execute(context):
                return PipelineResult(
                    success=False,
                    data=None,
                    metadata={'pipeline': pipeline.name},
                    error_message="Pre-execution validation failed"
                )
            
            # Execute with timing
            with context.timer.time_phase(phase_name, pipeline.description):
                start_time = asyncio.get_event_loop().time()
                result = await pipeline.execute(context)
                execution_time = asyncio.get_event_loop().time() - start_time
                
                # Set execution time
                result.execution_time = execution_time
                result.metadata['pipeline'] = pipeline.name
                result.metadata['phase'] = phase_name
            
            # Post-execution cleanup
            await pipeline.post_execute(context, result)
            
            return result
            
        except Exception as e:
            error_msg = f"Pipeline {pipeline.name} failed with exception: {str(e)}"
            context.logger.log_error(error_msg, pipeline.__class__.__name__, e)
            
            return PipelineResult(
                success=False,
                data=None,
                metadata={'pipeline': pipeline.name, 'phase': phase_name},
                error_message=error_msg
            )