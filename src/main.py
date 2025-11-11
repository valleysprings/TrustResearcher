#!/usr/bin/env python3
"""
TrustResearcher Main Module

This implements a comprehensive research pipeline with the following methodology:
- Orchestrated 7-phase pipeline architecture
- Knowledge Graph Generation for external memory
- Self-Planning & Critique with Graph-of-Thought reasoning
- Multi-agent system with specialized roles
- Literature-informed idea generation
- Distinctness analysis and portfolio optimization
"""

import sys
import asyncio
import argparse
from typing import Dict, Any

from .pipelines.research_pipeline_orchestrator import ResearchPipelineOrchestrator
from .utils.config import load_config
from .utils.debug_logger import init_debug_logger
from .utils.phase_timer import PhaseTimer




async def main():
    """Main function that orchestrates the TrustResearcher using the pipeline architecture"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='TrustResearcher - Orchestrated Pipeline')
    parser.add_argument('--topic', type=str, required=False, 
                       help='Seed topic for research idea generation')
    parser.add_argument('--num_ideas', type=int, default=3,
                       help='Number of research ideas to generate')
    parser.add_argument('--config', type=str, default='configs/agent_config.yaml',
                       help='Configuration file path')
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
        from .utils.pregen_validation import run_pregen_validation, print_validation_report
        
        print("Running Pre-Generation Validation Tests")
        print("=" * 50)
        
        # Load config for validation
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

    # Reset global token statistics for new session
    from .utils.token_cost_tracker import reset_global_session_stats
    reset_global_session_stats()
    
    # Print startup information
    print("TrustResearcher - Orchestrated Pipeline Architecture")
    print("=" * 60)
    if args.debug:
        print("DEBUG MODE ENABLED - Comprehensive logging active")
        print(f"Logs saved to: logs/{{session,idea,kg,llm}}/")

    print(f"Topic: {args.topic}")
    print(f"Generating {args.num_ideas} research ideas")
    print("=" * 60)
    
    logger.log_info(f"Starting orchestrated research pipeline for topic: {args.topic}")
    
    try:
        # Load configuration
        logger.log_info("Loading configuration")
        config = load_config(args.config)
        logger.log_info(f"Configuration loaded with {len(config)} sections")
        print("Configuration loaded")
        
        # Initialize pipeline orchestrator
        logger.log_info("Initializing research pipeline orchestrator")
        orchestrator = ResearchPipelineOrchestrator(config, logger, timer)
        
        # Setup agents
        print("Initializing research agents...")
        setup_success = await orchestrator.setup_agents()
        if not setup_success:
            print("Failed to initialize research agents")
            sys.exit(1)
        
        # Execute the complete research pipeline
        print("\nStarting Orchestrated Research Pipeline")
        print("=" * 60)
        
        execution_result = await orchestrator.execute_research_pipeline(
            topic=args.topic,
            num_ideas=args.num_ideas,
            run_validation=not args.skip_pregen,
            skip_phases=[]
        )
        
        if execution_result['success']:
            # Print execution summary
            summary = execution_result['execution_summary']
            print(f"\nPipeline Execution Summary")
            print("-" * 40)
            print(f"Total Pipelines: {summary['total_pipelines']}")
            print(f"Executed: {summary['executed_pipelines']}")
            print(f"Successful: {summary['successful_pipelines']}")
            print(f"Failed: {summary['failed_pipelines']}")
            print(f"Skipped: {summary['skipped_pipelines']}")
            
            # Print performance summary
            total_time = timer.get_total_time()
            timer.print_performance_summary()
            
            # Print session summary
            session_summary = logger.get_session_summary()
            print(f"\nSession Summary")
            print("-" * 40)
            print(f"Total Execution Time: {total_time:.2f} seconds")
            print(f"LLM Conversations: {sum(logger.llm_counters.values())}")
            print(f"Output File: {execution_result['output_path']}")
            
            if args.debug:
                print(f"\nDebug Information:")
                print(f"Session ID: {session_summary['session_id']}")
                print(f"Main Log File: {session_summary['log_files']['main_log']}")
                print(f"LLM Conversation File: {session_summary['log_files']['llm_log']}")
            
            logger.finalize_session()
            logger.log_info(f"Orchestrated research pipeline completed successfully in {total_time:.2f} seconds")
            print(f"\nTrustResearcher completed successfully!")
            
        else:
            print(f"\nPipeline execution failed: {execution_result['error']}")
            logger.log_error(f"Pipeline execution failed: {execution_result['error']}")
            sys.exit(1)
        
        # Cleanup
        await orchestrator.cleanup()

        # Give aiohttp sessions time to fully close
        await asyncio.sleep(0.1)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        logger.log_info("Execution interrupted by user")
        try:
            if 'orchestrator' in locals():
                await orchestrator.cleanup()
                await asyncio.sleep(0.1)  # Give sessions time to close
        except:
            pass
        sys.exit(1)
        
    except Exception as e:
        logger.log_error(f"Fatal error in main execution: {str(e)}", "main_system", e)
        print(f"Error: {str(e)}")
        
        # Cleanup on error
        try:
            if 'orchestrator' in locals():
                await orchestrator.cleanup()
                await asyncio.sleep(0.1)  # Give sessions time to close
        except:
            pass
        
        if args.debug:
            import traceback
            traceback.print_exc()
        
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
