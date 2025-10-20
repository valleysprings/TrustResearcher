"""
Validation Pipeline

Handles pre-generation validation of external services and system components.
This pipeline validates LLM endpoints, APIs, and configurations before
executing expensive research operations.
"""

from typing import Dict, Any
from .base_pipeline import BasePipeline, PipelineResult, PipelineContext
from ..utils.pregen_validation import run_pregen_validation
from ..utils.debug_logger import ComponentValidator


class ValidationPipeline(BasePipeline):
    """
    Pipeline for validating system components and external services
    """
    
    def __init__(self):
        super().__init__(
            name="System Validation",
            description="Validate external services and system components"
        )
        self.add_output("validation_results")
    
    async def execute(self, context: PipelineContext) -> PipelineResult:
        """
        Execute validation of system components
        
        Args:
            context: Pipeline context containing configuration
            
        Returns:
            PipelineResult: Validation results and component status
        """
        try:
            # Run pre-generation validation
            print("Phase 0: Validating External Services")
            print("-" * 40)
            
            success, validation_results = run_pregen_validation(context.config, context.logger)
            
            # Show validation summary
            total_tests = len(validation_results)
            passed_tests = sum(1 for r in validation_results if r.success)
            
            if success:
                print(f"All external services validated ({passed_tests}/{total_tests} tests passed)")
            else:
                print(f"External service validation issues ({passed_tests}/{total_tests} tests passed)")
                
                # Identify critical failures
                critical_failures = [r for r in validation_results if not r.success and 
                                   r.service in ['LLM', 'Configuration']]
                
                if critical_failures:
                    print("\nCritical service failures detected:")
                    for failure in critical_failures:
                        print(f"   • {failure.service} - {failure.test_name}: {failure.message}")
            
            # Prepare detailed validation data
            validation_data = {
                'overall_success': success,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'results': [
                    {
                        'service': r.service,
                        'test_name': r.test_name,
                        'success': r.success,
                        'message': r.message,
                        'is_critical': r.service in ['LLM', 'Configuration']
                    }
                    for r in validation_results
                ],
                'critical_failures': len([r for r in validation_results if not r.success and 
                                        r.service in ['LLM', 'Configuration']])
            }
            
            return PipelineResult(
                success=success,
                data=validation_data,
                metadata={
                    'validation_type': 'pre_generation',
                    'services_tested': list(set(r.service for r in validation_results))
                }
            )
            
        except Exception as e:
            context.logger.log_error(f"Validation pipeline failed: {str(e)}", "ValidationPipeline", e)
            return PipelineResult(
                success=False,
                data=None,
                metadata={'validation_type': 'pre_generation'},
                error_message=f"Validation pipeline error: {str(e)}"
            )


class ComponentValidationPipeline(BasePipeline):
    """
    Pipeline for validating system components (agents, knowledge graph, etc.)
    """
    
    def __init__(self):
        super().__init__(
            name="Component Validation", 
            description="Validate system components and agents"
        )
        self.add_dependency("agents_initialized")
        self.add_output("component_validation_results")
    
    async def execute(self, context: PipelineContext) -> PipelineResult:
        """
        Execute component validation
        
        Args:
            context: Pipeline context with initialized agents
            
        Returns:
            PipelineResult: Component validation results
        """
        try:
            print("Running Component Validation")
            print("-" * 40)
            
            validator = ComponentValidator(context.logger)
            validation_results = {}
            
            # This would need to be passed in context or retrieved differently
            # For now, we'll return a placeholder structure
            
            return PipelineResult(
                success=True,
                data={
                    'components_validated': [],
                    'validation_summary': 'Component validation would be implemented with agent instances'
                },
                metadata={'validation_type': 'component_validation'}
            )
            
        except Exception as e:
            return PipelineResult(
                success=False,
                data=None,
                metadata={'validation_type': 'component_validation'},
                error_message=f"Component validation error: {str(e)}"
            )