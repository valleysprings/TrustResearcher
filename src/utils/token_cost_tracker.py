#!/usr/bin/env python3
"""
Token Cost Tracker for TrustResearcher

Provides precise token counting and cost calculation for LLM invocations using tokencost library.
Integrates with the existing logging system to track token usage and costs per agent and session.
"""

import warnings
import os
import sys
import contextlib
from typing import List, Dict, Any, Union, Optional
from decimal import Decimal
from datetime import datetime
import json
from io import StringIO

# Comprehensive warning suppression for tiktoken model update messages
# Must be done BEFORE importing tokencost/tiktoken
warnings.filterwarnings("ignore", message=".*may update over time.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Returning num tokens assuming.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*gpt-4.*may update over time.*", category=UserWarning)

# Also suppress at environment level for tiktoken
os.environ.setdefault('TIKTOKEN_CACHE_DIR', '/tmp/tiktoken_cache')

# Context manager to suppress stderr output
@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output"""
    old_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        yield
    finally:
        sys.stderr = old_stderr

# Import tiktoken with both warnings and stderr suppressed
with warnings.catch_warnings(), suppress_stderr():
    warnings.simplefilter("ignore", UserWarning)
    import tiktoken
    import tokencost

# Global token tracking across all agents
_global_session_stats = {
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_prompt_cost": Decimal('0.0'),
    "total_completion_cost": Decimal('0.0'),
    "agent_stats": {},
    "conversation_count": 0
}


class TokenCostTracker:
    """Track token usage and costs for LLM invocations"""
    
    def __init__(self, model_name: str, logger=None):
        """
        Initialize token cost tracker
        
        Args:
            model_name: Name of the model used for cost calculation
            logger: Debug logger instance for logging costs and metrics
        """
        self.model_name = model_name
        self.logger = logger
        
        # Validate model is supported by tokencost
        if model_name not in tokencost.TOKEN_COSTS:
            # Try to find closest match or use default
            available_models = list(tokencost.TOKEN_COSTS.keys())
            if logger:
                logger.log_warning(f"Model '{model_name}' not found in tokencost. Available models: {available_models[:10]}...")
            
            # Use gpt-4 as fallback for cost calculation
            self.cost_model = "gpt-4" 
            if logger:
                logger.log_info(f"Using '{self.cost_model}' for cost calculation as fallback", "token_tracker")
        else:
            self.cost_model = model_name
        
        # Session tracking
        self.session_stats = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_prompt_cost": Decimal('0.0'),
            "total_completion_cost": Decimal('0.0'),
            "agent_stats": {},
            "conversation_count": 0
        }
    
    def count_tokens(self, text: Union[str, List[Dict]], is_messages: bool = False) -> int:
        """
        Count tokens in text or message list
        
        Args:
            text: String or list of message dictionaries
            is_messages: True if text is a list of message dictionaries
            
        Returns:
            Number of tokens
        """
        try:
            # Suppress all tiktoken related warnings and stderr output for cleaner output
            with warnings.catch_warnings(), suppress_stderr():
                warnings.filterwarnings("ignore", message=".*may update over time.*")
                warnings.filterwarnings("ignore", message=".*Returning num tokens assuming.*")
                warnings.filterwarnings("ignore", category=UserWarning, module="tiktoken")
                if is_messages and isinstance(text, list):
                    try:
                        return tokencost.count_message_tokens(text, self.cost_model)
                    except KeyError:
                        return tokencost.count_message_tokens(text, "gpt-4o")
                else:
                    return tokencost.count_string_tokens(str(text), self.cost_model)
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error counting tokens: {e}", "token_tracker")
            # Fallback to word count estimation (1 token ≈ 0.75 words)
            word_count = len(str(text).split())
            return int(word_count / 0.75)
    
    def calculate_prompt_cost(self, prompt: Union[str, List[Dict]]) -> Decimal:
        """
        Calculate cost for prompt
        
        Args:
            prompt: String prompt or list of message dictionaries
            
        Returns:
            Cost in USD as Decimal
        """
        try:
            with warnings.catch_warnings(), suppress_stderr():
                warnings.filterwarnings("ignore", message=".*may update over time.*")
                warnings.filterwarnings("ignore", message=".*Returning num tokens assuming.*")
                warnings.filterwarnings("ignore", category=UserWarning, module="tiktoken")
                try:
                    return tokencost.calculate_prompt_cost(prompt, self.cost_model)
                except KeyError:
                    return tokencost.calculate_prompt_cost(prompt, "gpt-4o")
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error calculating prompt cost: {e}", "token_tracker")
            return Decimal('0.0')
    
    def calculate_completion_cost(self, completion: str) -> Decimal:
        """
        Calculate cost for completion/response
        
        Args:
            completion: Response text from the model
            
        Returns:
            Cost in USD as Decimal
        """
        try:
            with warnings.catch_warnings(), suppress_stderr():
                warnings.filterwarnings("ignore", message=".*may update over time.*")
                warnings.filterwarnings("ignore", message=".*Returning num tokens assuming.*")
                warnings.filterwarnings("ignore", category=UserWarning, module="tiktoken")
                return tokencost.calculate_completion_cost(completion, self.cost_model)
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error calculating completion cost: {e}", "token_tracker")
            return Decimal('0.0')
    
    def calculate_total_cost(self, messages: List[Dict], response: str) -> Dict[str, Any]:
        """
        Calculate total cost for a conversation (prompt + response)
        
        Args:
            messages: List of message dictionaries for the prompt
            response: Response text from the model
            
        Returns:
            Dictionary with detailed token and cost information
        """
        try:
            # Count tokens
            prompt_tokens = self.count_tokens(messages, is_messages=True)
            response_tokens = self.count_tokens(response)
            total_tokens = prompt_tokens + response_tokens
            
            # Calculate costs
            prompt_cost = self.calculate_prompt_cost(messages)
            response_cost = self.calculate_completion_cost(response)
            total_cost = prompt_cost + response_cost
            
            return {
                "model_used": self.model_name,
                "cost_model": self.cost_model,
                "tokens": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": response_tokens,
                    "total_tokens": total_tokens
                },
                "costs_usd": {
                    "prompt_cost": float(prompt_cost),
                    "completion_cost": float(response_cost), 
                    "total_cost": float(total_cost)
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error calculating total cost: {e}", "token_tracker")
            return {
                "model_used": self.model_name,
                "cost_model": self.cost_model,
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "costs_usd": {"prompt_cost": 0.0, "completion_cost": 0.0, "total_cost": 0.0},
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def track_conversation(self, agent_name: str, messages: List[Dict], response: str) -> Dict[str, Any]:
        """
        Track a complete conversation with cost calculation and session statistics

        Args:
            agent_name: Name of the agent making the request
            messages: List of message dictionaries for the prompt
            response: Response text from the model

        Returns:
            Dictionary with cost information and updated session stats
        """
        global _global_session_stats

        # Calculate costs for this conversation
        cost_info = self.calculate_total_cost(messages, response)

        # Update global session statistics
        _global_session_stats["total_prompt_tokens"] += cost_info["tokens"]["prompt_tokens"]
        _global_session_stats["total_completion_tokens"] += cost_info["tokens"]["completion_tokens"]
        _global_session_stats["total_prompt_cost"] += Decimal(str(cost_info["costs_usd"]["prompt_cost"]))
        _global_session_stats["total_completion_cost"] += Decimal(str(cost_info["costs_usd"]["completion_cost"]))
        _global_session_stats["conversation_count"] += 1

        # Update agent-specific statistics in global tracker
        if agent_name not in _global_session_stats["agent_stats"]:
            _global_session_stats["agent_stats"][agent_name] = {
                "conversations": 0,
                "total_tokens": 0,
                "total_cost": Decimal('0.0')
            }

        agent_stats = _global_session_stats["agent_stats"][agent_name]
        agent_stats["conversations"] += 1
        agent_stats["total_tokens"] += cost_info["tokens"]["total_tokens"]
        agent_stats["total_cost"] += Decimal(str(cost_info["costs_usd"]["total_cost"]))

        # Also update local session stats for backward compatibility
        self.session_stats["total_prompt_tokens"] += cost_info["tokens"]["prompt_tokens"]
        self.session_stats["total_completion_tokens"] += cost_info["tokens"]["completion_tokens"]
        self.session_stats["total_prompt_cost"] += Decimal(str(cost_info["costs_usd"]["prompt_cost"]))
        self.session_stats["total_completion_cost"] += Decimal(str(cost_info["costs_usd"]["completion_cost"]))
        self.session_stats["conversation_count"] += 1

        if agent_name not in self.session_stats["agent_stats"]:
            self.session_stats["agent_stats"][agent_name] = {
                "conversations": 0,
                "total_tokens": 0,
                "total_cost": Decimal('0.0')
            }

        local_agent_stats = self.session_stats["agent_stats"][agent_name]
        local_agent_stats["conversations"] += 1
        local_agent_stats["total_tokens"] += cost_info["tokens"]["total_tokens"]
        local_agent_stats["total_cost"] += Decimal(str(cost_info["costs_usd"]["total_cost"]))

        # Log the costs
        if self.logger:
            self.logger.log_info(
                f"Token Cost - Agent: {agent_name}, Tokens: {cost_info['tokens']['total_tokens']}, "
                f"Cost: ${cost_info['costs_usd']['total_cost']:.6f}",
                "token_tracker"
            )

        # Add agent name to cost info
        cost_info["agent_name"] = agent_name

        return cost_info
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive session cost summary from global statistics

        Returns:
            Dictionary with session cost statistics
        """
        global _global_session_stats

        total_cost = _global_session_stats["total_prompt_cost"] + _global_session_stats["total_completion_cost"]
        total_tokens = _global_session_stats["total_prompt_tokens"] + _global_session_stats["total_completion_tokens"]

        # Convert agent stats to serializable format
        agent_summary = {}
        for agent, stats in _global_session_stats["agent_stats"].items():
            agent_summary[agent] = {
                "conversations": stats["conversations"],
                "total_tokens": stats["total_tokens"],
                "total_cost_usd": float(stats["total_cost"]),
                "avg_tokens_per_conversation": stats["total_tokens"] / stats["conversations"] if stats["conversations"] > 0 else 0,
                "avg_cost_per_conversation": float(stats["total_cost"]) / stats["conversations"] if stats["conversations"] > 0 else 0.0
            }

        return {
            "model_used": self.model_name,
            "cost_model": self.cost_model,
            "session_totals": {
                "conversations": _global_session_stats["conversation_count"],
                "total_tokens": total_tokens,
                "prompt_tokens": _global_session_stats["total_prompt_tokens"],
                "completion_tokens": _global_session_stats["total_completion_tokens"],
                "total_cost_usd": float(total_cost),
                "prompt_cost_usd": float(_global_session_stats["total_prompt_cost"]),
                "completion_cost_usd": float(_global_session_stats["total_completion_cost"]),
                "avg_tokens_per_conversation": total_tokens / _global_session_stats["conversation_count"] if _global_session_stats["conversation_count"] > 0 else 0,
                "avg_cost_per_conversation": float(total_cost) / _global_session_stats["conversation_count"] if _global_session_stats["conversation_count"] > 0 else 0.0
            },
            "agent_breakdown": agent_summary,
            "timestamp": datetime.now().isoformat()
        }
    
    def log_session_summary(self):
        """Log final session cost summary"""
        summary = self.get_session_summary()
        
        if self.logger:
            session_totals = summary["session_totals"]
            self.logger.log_info(
                f"SESSION COST SUMMARY - Total: ${session_totals['total_cost_usd']:.6f}, "
                f"Tokens: {session_totals['total_tokens']}, "
                f"Conversations: {session_totals['conversations']}", 
                "token_tracker"
            )
            
            # Log per-agent breakdown
            for agent, stats in summary["agent_breakdown"].items():
                self.logger.log_info(
                    f"Agent '{agent}' - Cost: ${stats['total_cost_usd']:.6f}, "
                    f"Tokens: {stats['total_tokens']}, "
                    f"Conversations: {stats['conversations']}", 
                    "token_tracker"
                )


def get_available_models() -> List[str]:
    """Get list of models supported by tokencost"""
    return list(tokencost.TOKEN_COSTS.keys())


def is_model_supported(model_name: str) -> bool:
    """Check if a model is supported by tokencost"""
    return model_name in tokencost.TOKEN_COSTS


def reset_global_session_stats():
    """Reset global session statistics for a new session"""
    global _global_session_stats
    _global_session_stats.update({
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_prompt_cost": Decimal('0.0'),
        "total_completion_cost": Decimal('0.0'),
        "agent_stats": {},
        "conversation_count": 0
    })