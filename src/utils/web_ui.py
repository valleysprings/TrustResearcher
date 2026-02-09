#!/usr/bin/env python3
"""
Web UI for TrustResearcher Process Visualization

Provides real-time visualization of the research agent's execution phases,
idea generation progress, and final results through a clean Gradio interface.
"""

import json
import asyncio
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import gradio as gr
import pandas as pd

from ..main import ResearchOrchestrator, ResearchContext
from .config import load_config
from .debug_logger import init_debug_logger
from .phase_timer import PhaseTimer


class ProcessVisualizerUI:
    """Complete process visualization UI for TrustResearcher"""

    def __init__(self, config: Dict = None, port: int = 7860, share: bool = False, host: str = 'localhost'):
        """Initialize the process visualization UI"""
        self.config = config or {}
        self.port = port
        self.share = share
        self.host = host or 'localhost'
        self.interface = None
        self.base_dir = Path.cwd()

        # Pipeline state
        self.current_orchestrator = None
        self.current_context = None
        self.is_running = False
        self.logs = []
        self.phase_status = {}
        self.timing_info = {}
        self.token_info = {}

    def create_interface(self):
        """Create the complete Gradio interface"""
        # Custom CSS for better styling
        self.custom_css = """
        * {
            font-family: 'Times New Roman', Times, serif !important;
        }
        .gradio-container {
            max-width: 100% !important;
            padding: 1.5rem 3rem !important;
        }
        body {
            font-size: 16px !important;
        }
        """

        with gr.Blocks(title="TrustResearcher") as interface:
            gr.Markdown("# 🔬 TrustResearcher")

            with gr.Row():
                with gr.Column(scale=1):
                    # Input section
                    gr.Markdown("## 📝 Input Configuration")
                    topic_input = gr.Textbox(
                        label="Research Topic",
                        placeholder="Enter your research topic...",
                        lines=2
                    )
                    num_ideas_input = gr.Number(
                        label="Number of Ideas",
                        value=3,
                        minimum=1,
                        maximum=10,
                        step=1
                    )

                    start_btn = gr.Button("🚀 Start Research Pipeline", variant="primary", size="lg")
                    stop_btn = gr.Button("⏹️ Stop", variant="stop", size="sm")

                    # Status section
                    gr.Markdown("## 📊 Pipeline Status")
                    status_display = gr.Textbox(
                        label="Current Status",
                        value="Ready to start",
                        interactive=False,
                        lines=2
                    )

                    # Phase progress
                    phase_progress = gr.Textbox(
                        label="Phase Progress",
                        value="",
                        interactive=False,
                        lines=8
                    )

                with gr.Column(scale=2):
                    # Results section
                    gr.Markdown("## 📋 Results")

                    with gr.Tabs():
                        with gr.Tab("Ideas"):
                            ideas_display = gr.JSON(
                                label="Generated Research Ideas"
                            )

                        with gr.Tab("Literature"):
                            literature_display = gr.JSON(
                                label="Relevant Papers"
                            )

                        with gr.Tab("Metadata"):
                            metadata_display = gr.JSON(
                                label="Generation Metadata"
                            )

                        with gr.Tab("Logs"):
                            logs_display = gr.Textbox(
                                label="Activity Logs",
                                lines=20,
                                interactive=False
                            )

            # Timer for auto-refresh
            timer = gr.Timer(value=2)

            # Event handlers
            start_btn.click(
                fn=self.start_pipeline,
                inputs=[topic_input, num_ideas_input],
                outputs=[status_display, phase_progress, ideas_display, literature_display, metadata_display, logs_display]
            )

            stop_btn.click(
                fn=self.stop_pipeline,
                outputs=[status_display]
            )

            # Auto-refresh status every 2 seconds using Timer
            timer.tick(
                fn=self.get_status,
                outputs=[status_display, phase_progress, ideas_display, literature_display, metadata_display, logs_display]
            )

        self.interface = interface
        return interface

    def start_pipeline(self, topic: str, num_ideas: int):
        """Start the research pipeline"""
        if self.is_running:
            return (
                "Pipeline is already running",
                self._format_phase_status(),
                None, None, None,
                "\n".join(self.logs)
            )

        if not topic or not topic.strip():
            return (
                "Error: Please enter a research topic",
                "",
                None, None, None,
                "Error: Topic is required"
            )

        # Reset state
        self.logs = []
        self.phase_status = {}
        self.is_running = True

        # Add initial log
        self._add_log(f"Starting pipeline for topic: {topic}")
        self._add_log(f"Generating {int(num_ideas)} ideas")

        # Run pipeline in background thread
        thread = threading.Thread(
            target=self._run_pipeline_sync,
            args=(topic, int(num_ideas))
        )
        thread.daemon = True
        thread.start()

        return (
            "Pipeline started...",
            self._format_phase_status(),
            None, None, None,
            "\n".join(self.logs)
        )

    def stop_pipeline(self):
        """Stop the research pipeline"""
        self.is_running = False
        self._add_log("Pipeline stopped by user")
        return "Pipeline stopped"

    def get_status(self):
        """Get current pipeline status for UI updates"""
        status_text = "Running..." if self.is_running else "Ready" if not self.logs else "Completed"

        # Get results if available
        ideas_json = None
        literature_json = None
        metadata_json = None

        if self.current_context and hasattr(self.current_context, 'final_ideas'):
            if self.current_context.final_ideas:
                ideas_json = [
                    {
                        "topic": idea.topic,
                        "problem_statement": getattr(idea, 'problem_statement', ''),
                        "proposed_methodology": getattr(idea, 'proposed_methodology', ''),
                    }
                    for idea in self.current_context.final_ideas
                ]

            if self.current_context.relevant_papers:
                literature_json = [
                    {
                        "title": paper.title,
                        "authors": getattr(paper, 'authors', []),
                        "year": getattr(paper, 'year', ''),
                        "citation_count": getattr(paper, 'citation_count', 'NA')
                    }
                    for paper in self.current_context.relevant_papers
                ]

            metadata_json = {
                "topic": self.current_context.topic,
                "num_ideas_requested": self.current_context.num_ideas,
                "ideas_generated": len(self.current_context.ideas) if self.current_context.ideas else 0,
                "ideas_selected": len(self.current_context.selected_ideas) if self.current_context.selected_ideas else 0,
                "final_ideas": len(self.current_context.final_ideas) if self.current_context.final_ideas else 0,
                "timing": self.timing_info,
                "tokens": self.token_info
            }

        return (
            status_text,
            self._format_phase_status(),
            ideas_json,
            literature_json,
            metadata_json,
            "\n".join(self.logs[-50:])  # Last 50 log entries
        )

    def _add_log(self, message: str):
        """Add a log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)

    def _format_phase_status(self) -> str:
        """Format phase status for display"""
        if not self.phase_status:
            return "No phases started yet"

        status_lines = []
        phase_order = ["validation", "retrieval", "ideation", "selection", "review", "final_selection"]

        for phase in phase_order:
            if phase in self.phase_status:
                status = self.phase_status[phase]
                emoji = "✅" if status == "completed" else "🔄" if status == "running" else "⏸️"
                status_lines.append(f"{emoji} {phase.replace('_', ' ').title()}: {status}")

        return "\n".join(status_lines)

    def _run_pipeline_sync(self, topic: str, num_ideas: int):
        """Run the pipeline synchronously in a background thread"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run the async pipeline (always run validation)
            loop.run_until_complete(self._run_pipeline_async(topic, num_ideas))

        except Exception as e:
            self._add_log(f"Error: {str(e)}")
            self.phase_status["error"] = str(e)
        finally:
            self.is_running = False
            loop.close()

    async def _run_pipeline_async(self, topic: str, num_ideas: int):
        """Run the research pipeline asynchronously"""
        try:
            # Initialize logger and timer
            logger = init_debug_logger(debug_mode=False, topic=topic)
            timer = PhaseTimer(logger)

            self._add_log("Initializing research orchestrator...")

            # Initialize orchestrator
            orchestrator = ResearchOrchestrator(self.config, logger, timer)
            self.current_orchestrator = orchestrator

            # Setup agents
            self._add_log("Setting up research agents...")
            self.phase_status["setup"] = "running"

            setup_success = await orchestrator.setup_agents()
            if not setup_success:
                self._add_log("Failed to initialize research agents")
                self.phase_status["setup"] = "failed"
                return

            self.phase_status["setup"] = "completed"
            self._add_log("All agents initialized successfully")

            # Execute pipeline with phase tracking
            result = await self._execute_with_tracking(orchestrator, topic, num_ideas)

            if result['success']:
                self._add_log("Pipeline completed successfully!")
                self._add_log(f"Results saved to: {result['output_path']}")
            else:
                self._add_log(f"Pipeline failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            self._add_log(f"Pipeline error: {str(e)}")
            import traceback
            self._add_log(traceback.format_exc())

    async def _execute_with_tracking(self, orchestrator: ResearchOrchestrator,
                                     topic: str, num_ideas: int) -> Dict[str, Any]:
        """Execute pipeline with phase tracking"""
        try:
            # Initialize context
            ctx = ResearchContext(
                topic=topic,
                num_ideas=num_ideas,
                config=self.config,
                logger=orchestrator.logger,
                timer=orchestrator.timer
            )
            self.current_context = ctx

            # Phase 0: Validation (always run)
            self._add_log("Phase 0: Validating external services...")
            self.phase_status["validation"] = "running"
            if not await orchestrator.run_validation(ctx):
                self.phase_status["validation"] = "failed"
                return {'success': False, 'error': 'Validation failed'}
            self.phase_status["validation"] = "completed"
            self._add_log("Validation completed")

            # Phase 1: Literature retrieval
            self._add_log("Phase 1: Searching relevant literature...")
            self.phase_status["retrieval"] = "running"
            await orchestrator.run_retrieval(ctx)
            self.phase_status["retrieval"] = "completed"
            self._add_log(f"Found {len(ctx.relevant_papers)} relevant papers")

            # Phase 2: Idea generation
            self._add_log("Phase 2: Generating research ideas...")
            self.phase_status["ideation"] = "running"
            if not await orchestrator.run_ideation(ctx):
                self.phase_status["ideation"] = "failed"
                return {'success': False, 'error': 'Idea generation failed'}
            self.phase_status["ideation"] = "completed"
            self._add_log(f"Generated {len(ctx.ideas)} ideas")

            # Phase 3: Preliminary selection
            self._add_log("Phase 3: Preliminary idea selection...")
            self.phase_status["selection"] = "running"
            if not await orchestrator.run_selection(ctx):
                self.phase_status["selection"] = "failed"
                return {'success': False, 'error': 'Selection failed'}
            self.phase_status["selection"] = "completed"
            self._add_log(f"Selected {len(ctx.filtered_ideas)} ideas for review")

            # Phase 4: Detailed review
            self._add_log("Phase 4: Detailed review and critique...")
            self.phase_status["review"] = "running"
            if not await orchestrator.run_review(ctx):
                self.phase_status["review"] = "failed"
                return {'success': False, 'error': 'Review failed'}
            self.phase_status["review"] = "completed"
            self._add_log(f"Reviewed {len(ctx.refined_ideas)} ideas")

            # Phase 5: Final selection
            self._add_log("Phase 5: Final idea selection...")
            self.phase_status["final_selection"] = "running"
            if not await orchestrator.run_final_selection(ctx):
                self.phase_status["final_selection"] = "failed"
                return {'success': False, 'error': 'Final selection failed'}
            self.phase_status["final_selection"] = "completed"
            self._add_log(f"Selected {len(ctx.final_ideas)} final ideas")

            # Get timing and cost information
            total_time = orchestrator.timer.get_total_time()

            # Get token usage from cost tracker
            token_info = {}
            if orchestrator.timer.cost_tracker:
                cost_tracker = orchestrator.timer.cost_tracker
                token_info = {
                    'total_tokens': cost_tracker.total_tokens,
                    'total_cost': cost_tracker.total_cost,
                    'prompt_tokens': cost_tracker.prompt_tokens,
                    'completion_tokens': cost_tracker.completion_tokens
                }

            # Store timing and token info for UI access
            self.timing_info = {
                'total_time': total_time,
                'phases': orchestrator.timer.phase_times if hasattr(orchestrator.timer, 'phase_times') else {}
            }
            self.token_info = token_info

            # Log summary
            self._add_log("=" * 50)
            self._add_log(f"Pipeline completed in {total_time:.2f} seconds")
            if token_info:
                self._add_log(f"Total tokens: {token_info['total_tokens']:,}")
                self._add_log(f"Total cost: ${token_info['total_cost']:.4f}")
            self._add_log("=" * 50)

            # Generate and save output
            json_output = orchestrator._generate_final_output(ctx)
            output_path = orchestrator._generate_output_path(topic, orchestrator.logger.session_id)
            orchestrator._save_results(output_path, json_output)

            return {
                'success': True,
                'output_path': str(output_path),
                'results': json_output,
                'context': ctx,
                'timing': {
                    'total_time': total_time,
                    'phases': orchestrator.timer.phase_times if hasattr(orchestrator.timer, 'phase_times') else {}
                },
                'tokens': token_info
            }

        except Exception as e:
            self._add_log(f"Execution error: {str(e)}")
            return {'success': False, 'error': str(e)}

    def launch(self):
        """Launch the Gradio interface"""
        if self.interface is None:
            self.create_interface()

        # Find available port
        available_port = self.port
        import socket

        def is_port_available(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((self.host, port))
                    return True
                except OSError:
                    return False

        for test_port in range(self.port, self.port + 10):
            if is_port_available(test_port):
                available_port = test_port
                break

        if available_port != self.port:
            print(f"⚠️  Port {self.port} occupied, using port {available_port} instead")

        print(f"🌐 Starting UI at: http://{self.host}:{available_port}")

        self.interface.launch(
            server_name=self.host,
            server_port=available_port,
            share=self.share,
            show_error=True,
            inbrowser=True,
            theme=gr.themes.Soft(),
            css=self.custom_css
        )


# Global UI instance for easy access
_ui_instance = None

def get_ui_instance() -> ProcessVisualizerUI:
    """Get or create the global UI instance"""
    global _ui_instance
    if _ui_instance is None:
        _ui_instance = ProcessVisualizerUI()
    return _ui_instance

def start_web_ui(host='localhost', port=7860, share=False) -> ProcessVisualizerUI:
    """Start the Gradio web UI server"""
    global _ui_instance
    _ui_instance = ProcessVisualizerUI(port=port, share=share, host=host)
    _ui_instance.launch()
    return _ui_instance
