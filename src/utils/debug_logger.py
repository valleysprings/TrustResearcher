#!/usr/bin/env python3
"""
Debug Logger for TrustResearcher

Provides comprehensive logging in a single consolidated file per session with line numbers
"""

import logging
import json
import os
import sys
import inspect
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class DebugLogger:
    """Enhanced logger with single file per session and line number tracking"""
    
    def __init__(self, debug_mode: bool = False, log_dir: str = "logs", topic: str = None):
        self.debug_mode = debug_mode
        self.base_log_dir = Path(log_dir)
        self.topic = topic

        # Create unified logs structure: logs/{session,idea,kg,llm}
        self.session_log_dir = self.base_log_dir / "session"
        self.llm_log_dir = self.base_log_dir / "llm"
        self.idea_log_dir = self.base_log_dir / "idea"
        self.kg_log_dir = self.base_log_dir / "kg"

        # Create all subdirectories
        for directory in [self.session_log_dir, self.llm_log_dir, self.idea_log_dir, self.kg_log_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Session timestamp - unified format
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Log files - all use timestamp prefix only
        self.main_log_file = self.session_log_dir / f"{self.session_id}.log"
        self.llm_log_file = self.llm_log_dir / f"{self.session_id}.jsonl"
        
        # Setup main logger
        self.logger = logging.getLogger(f"research_agent_{self.session_id}")
        self.logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(self.main_log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler for important messages
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
        
        # Create simple formatter with file location
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        if debug_mode:
            self.logger.addHandler(console_handler)
        
        # LLM conversation counters
        self.llm_counters = {}
        self.performance_metrics = {}
        self.component_states = {}
        
        # Log session start
        self.logger.info(f"=== NEW SESSION STARTED ===")
        self.logger.info(f"Session ID: {self.session_id}")
        self.logger.info(f"Debug Mode: {debug_mode}")
        self.logger.info(f"Log File: {self.main_log_file}")
    
    def _get_caller_info(self) -> tuple:
        """Get caller function and line number from actual source, not debug logger"""
        frame = inspect.currentframe()
        try:
            # Go up the stack to find the actual caller (skip debug logger frames)
            caller_frame = frame.f_back.f_back
            
            # Skip debug logger frames to get to real calling code
            while (caller_frame and 
                   ('debug_logger.py' in caller_frame.f_code.co_filename or
                    caller_frame.f_code.co_name.startswith('log_'))):
                caller_frame = caller_frame.f_back
            
            if caller_frame:
                # Get relative path from project root
                full_path = caller_frame.f_code.co_filename
                # Find 'src/' in the path and use everything from src/ onwards
                if '/src/' in full_path:
                    relative_path = 'src/' + full_path.split('/src/')[-1]
                else:
                    relative_path = os.path.basename(full_path)
                
                line_number = caller_frame.f_lineno
                function_name = caller_frame.f_code.co_name
                return relative_path, line_number, function_name
            else:
                return "unknown.py", 0, "unknown"
        finally:
            del frame
    
    def log_info(self, message: str, component: str = "main_system"):
        """Log informational message"""
        filename, line_no, func_name = self._get_caller_info()
        self.logger.info(f"[{filename}:{line_no}] {message}")
    
    def log_debug(self, message: str, component: str = "main_system"):
        """Log debug message"""
        if self.debug_mode:
            filename, line_no, func_name = self._get_caller_info()
            self.logger.debug(f"[{filename}:{line_no}] {message}")
    
    def log_error(self, message: str, component: str = "main_system", exception: Exception = None):
        """Log error message"""
        filename, line_no, func_name = self._get_caller_info()
        error_msg = f"[{filename}:{line_no}] {message}"
        if exception:
            error_msg += f" - Exception: {str(exception)}"
        
        self.logger.error(error_msg)
    
    def log_warning(self, message: str, component: str = "main_system"):
        """Log warning message"""
        filename, line_no, func_name = self._get_caller_info()
        self.logger.warning(f"[{filename}:{line_no}] {message}")
    
    def log_llm_conversation(self, agent_name: str, system_prompt: str, user_prompt: str, 
                           response: str, metadata: Dict[str, Any] = None):
        """Log LLM conversation to single shared file"""
        # Initialize counter for this agent if not exists
        if agent_name not in self.llm_counters:
            self.llm_counters[agent_name] = 0
        
        self.llm_counters[agent_name] += 1
        counter = self.llm_counters[agent_name]
        
        # Get caller info
        filename, line_no, func_name = self._get_caller_info()
        
        # Create conversation log entry
        conversation = {
            "session_id": self.session_id,
            "topic": self.topic,
            "agent_name": agent_name,
            "conversation_number": counter,
            "timestamp": datetime.now().isoformat(),
            "caller_info": {
                "file": filename,
                "line": line_no,
                "function": func_name
            },
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response": response,
            "metadata": metadata or {},
            "token_counts": {
                "system_prompt_tokens": len(system_prompt.split()),
                "user_prompt_tokens": len(user_prompt.split()),
                "response_tokens": len(response.split())
            },
            # Include detailed cost information if available in metadata
            "cost_info": metadata.get("cost_info") if metadata else None
        }
        
        # Write to single shared LLM log file
        with open(self.llm_log_file, 'a') as f:
            json.dump(conversation, f, ensure_ascii=False)
            f.write('\n')
        
        # Also log summary to main log
        cost_summary = ""
        if conversation.get("cost_info"):
            cost_info = conversation["cost_info"]
            cost_summary = f", Cost: ${cost_info['costs_usd']['total_cost']:.6f}"
        
        self.log_info(f"LLM Conversation #{counter} - Agent: {agent_name}, Tokens: {conversation['token_counts']['response_tokens']} response{cost_summary}", "llm_interface")
    
    def log_performance_metric(self, component: str, metric_name: str, value: float, unit: str = ""):
        """Log performance metrics"""
        if component not in self.performance_metrics:
            self.performance_metrics[component] = {}
        
        self.performance_metrics[component][metric_name] = {
            "value": value,
            "unit": unit,
            "timestamp": datetime.now().isoformat()
        }
        
        self.log_info(f"Performance - {metric_name}: {value} {unit}", component)
    
    def log_component_state(self, component: str, state: Dict[str, Any]):
        """Log component state"""
        self.component_states[component] = {
            "state": state,
            "timestamp": datetime.now().isoformat()
        }

        self.log_debug(f"Component state: {json.dumps(state, default=str)}", component)

    def save_ideas(self, ideas: list, seed_topic: str, num_ideas: int, overgenerate_factor: int):
        """Save all generated ideas to logs/idea/ directory.

        Args:
            ideas: List of ResearchIdea objects
            seed_topic: The research topic
            num_ideas: Target number of ideas requested
            overgenerate_factor: Overgeneration multiplier used
        """
        if not ideas:
            self.log_warning("No ideas to save", "ideagen")
            return

        try:
            # Use session timestamp for filename
            log_file = self.idea_log_dir / f"{self.session_id}.json"

            # Prepare ideas data with all refinement rounds
            ideas_data = []
            for idea in ideas:
                idea_dict = {
                    "topic": idea.topic,
                    "source": getattr(idea, 'source', 'unknown'),
                    "method": getattr(idea, 'method', 'unknown'),
                    "refinement_rounds": []
                }

                # Add initial facets
                idea_dict["refinement_rounds"].append({
                    "round": 1,
                    "label": "initial",
                    "reason": "Initial plan output",
                    "facets": {
                        "topic": idea.topic,
                        "Problem Statement": idea.problem_statement,
                        "Proposed Methodology": idea.proposed_methodology,
                        "Experimental Validation": idea.experimental_validation
                    }
                })

                # Add refinement rounds if available
                if hasattr(idea, 'review_feedback') and idea.review_feedback:
                    for idx, feedback in enumerate(idea.review_feedback, start=2):
                        if isinstance(feedback, dict):
                            round_data = {
                                "round": idx,
                                "label": feedback.get('type', 'refinement'),
                                "reason": feedback.get('reason', 'Applied refinement'),
                                "facets": {
                                    "topic": idea.topic,
                                    "Problem Statement": idea.problem_statement,
                                    "Proposed Methodology": idea.proposed_methodology,
                                    "Experimental Validation": idea.experimental_validation
                                }
                            }

                            # Add critique data if available
                            if feedback.get('type') == 'self_critique':
                                round_data["critique"] = {
                                    "overall_score": feedback.get('overall_score', 0),
                                    "novelty_score": feedback.get('novelty_score', 0),
                                    "feasibility_score": feedback.get('feasibility_score', 0),
                                    "clarity_score": feedback.get('clarity_score', 0),
                                    "impact_score": feedback.get('impact_score', 0),
                                    "needs_refinement": feedback.get('needs_refinement', False),
                                    "strengths": feedback.get('strengths', []),
                                    "weaknesses": feedback.get('weaknesses', []),
                                    "suggestions": feedback.get('suggestions', [])
                                }

                            idea_dict["refinement_rounds"].append(round_data)

                ideas_data.append(idea_dict)

            # Prepare complete log data
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "seed_topic": seed_topic,
                "target_generation_count": num_ideas,
                "actual_generation_count": len(ideas),
                "overgenerate_factor": overgenerate_factor,
                "ideas": ideas_data
            }

            # Save to file
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)

            self.log_info(f"Saved {len(ideas)} ideas to {log_file}", "ideagen")

        except Exception as e:
            self.log_error(f"Failed to save ideas log: {e}", "ideagen", e)

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "log_files": {
                "main_log": str(self.main_log_file),
                "llm_log": str(self.llm_log_file)
            },
            "log_structure": {
                "session": str(self.session_log_dir),
                "llm": str(self.llm_log_dir),
                "idea": str(self.idea_log_dir),
                "kg": str(self.kg_log_dir)
            },
            "llm_conversation_counts": self.llm_counters.copy(),
            "performance_metrics": self.performance_metrics.copy(),
            "component_states": {k: v["state"] for k, v in self.component_states.items()}
        }

    def finalize_session(self):
        """Finalize session and write summary"""
        summary = self.get_session_summary()

        # Write session summary - timestamp prefix only
        summary_file = self.session_log_dir / f"{self.session_id}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Session finalized - Summary: {summary_file}")
        self.logger.info(f"=== SESSION ENDED ===")


class ComponentValidator:
    """Component validator for testing system functionality"""
    
    def __init__(self, logger: DebugLogger):
        self.logger = logger
    
    async def validate_knowledge_graph(self, kg_builder) -> bool:
        """Validate knowledge graph builder"""
        try:
            # Test basic functionality - build_from_seed_topic is async
            await kg_builder.build_from_seed_topic("test topic")
            return True
        except Exception as e:
            self.logger.log_error(f"KG validation failed: {e}", "validator")
            return False
    
    async def validate_llm_interface(self, llm_interface) -> bool:
        """Validate LLM interface"""
        try:
            response = await llm_interface.generate_with_system_prompt("Test", "Hello", max_tokens=10, caller="validator")
            return len(response) > 0
        except Exception as e:
            self.logger.log_error(f"LLM validation failed: {e}", "validator")
            return False
    
    def validate_agent(self, agent, agent_name: str) -> bool:
        """Validate agent has required methods"""
        try:
            required_methods = ['gather_information', 'generate_ideas', 'critique_ideas', 'refine_ideas']
            
            for method_name in required_methods:
                if not hasattr(agent, method_name):
                    self.logger.log_error(f"Agent {agent_name} missing method: {method_name}", "validator")
                    return False
                
                # Test if method is callable
                method = getattr(agent, method_name)
                if not callable(method):
                    self.logger.log_error(f"Agent {agent_name} method {method_name} is not callable", "validator")
                    return False
                    
                self.logger.log_info(f"Validation - has_{method_name}: PASSED", "validator")
            
            return True
        except Exception as e:
            self.logger.log_error(f"Agent validation failed for {agent_name}: {e}", "validator")
            return False


def init_debug_logger(debug_mode: bool = False, topic: str = None) -> DebugLogger:
    """Initialize debug logger with topic"""
    return DebugLogger(debug_mode=debug_mode, topic=topic)