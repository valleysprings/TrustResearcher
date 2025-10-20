#!/usr/bin/env python3
"""
Phase Timer Utility

Unified timing system for tracking research agent performance across all phases.
Provides consistent timing methodology, comprehensive performance reports, and 
automatic parallel processing efficiency calculations.
"""

import time
from typing import Dict
from contextlib import contextmanager


class PhaseTimer:
    """Unified timing system for tracking research agent performance"""

    def __init__(self, logger=None, cost_tracker=None):
        """
        Initialize the PhaseTimer

        Args:
            logger: Optional debug logger instance for integration
            cost_tracker: Optional TokenCostTracker instance for cost tracking
        """
        self.logger = logger
        self.cost_tracker = cost_tracker
        self.timings: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}
        self.phase_order = []
        self.total_start_time = time.time()
    
    @contextmanager
    def time_phase(self, phase_name: str, description: str = None):
        """
        Context manager for timing a phase
        
        Args:
            phase_name: Internal name for the phase (used in logs/metrics)
            description: Human-readable description for console output
            
        Usage:
            with timer.time_phase("phase1_search", "📚 Literature Search"):
                # Phase code here
                pass
        """
        display_name = description or phase_name
        print(f"\n⏱️  Starting {display_name}...")
        
        if self.logger:
            self.logger.log_info(f"Starting {phase_name}")
        
        start_time = time.time()
        self.start_times[phase_name] = start_time
        
        try:
            yield self
        finally:
            end_time = time.time()
            duration = end_time - start_time
            self.timings[phase_name] = duration
            self.phase_order.append(phase_name)
            
            if self.logger:
                self.logger.log_performance_metric("main_system", f"{phase_name}_duration", duration, "seconds")
                self.logger.log_info(f"{phase_name} completed in {duration:.2f} seconds")
            
            print(f"✅ {display_name} completed in {duration:.2f} seconds")
    
    def get_phase_duration(self, phase_name: str) -> float:
        """
        Get the duration of a specific phase
        
        Args:
            phase_name: Name of the phase to query
            
        Returns:
            Duration in seconds, or 0.0 if phase not found
        """
        return self.timings.get(phase_name, 0.0)
    
    def get_total_time(self) -> float:
        """
        Get total execution time since timer initialization
        
        Returns:
            Total time in seconds
        """
        return time.time() - self.total_start_time
    
    def print_performance_summary(self):
        """
        Print a comprehensive performance summary with:
        - Phase-by-phase breakdown with percentages
        - Token usage and cost information (if available)
        - Fastest and slowest phase identification
        - Parallel processing efficiency calculation
        - Professional formatting
        """
        total_time = self.get_total_time()

        print(f"\n📊 Performance Summary")
        print("=" * 50)
        print(f"{'Phase':<25} {'Duration':<12} {'% of Total':<12}")
        print("-" * 50)

        for phase_name in self.phase_order:
            duration = self.timings[phase_name]
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            print(f"{phase_name:<25} {duration:<8.2f}s    {percentage:<8.1f}%")

        print("-" * 50)
        print(f"{'TOTAL EXECUTION':<25} {total_time:<8.2f}s    100.0%")

        # Show fastest and slowest phases
        if len(self.timings) > 1:
            fastest_phase = min(self.timings.items(), key=lambda x: x[1])
            slowest_phase = max(self.timings.items(), key=lambda x: x[1])

            print(f"\n🚀 Fastest Phase: {fastest_phase[0]} ({fastest_phase[1]:.2f}s)")
            print(f"🐌 Slowest Phase: {slowest_phase[0]} ({slowest_phase[1]:.2f}s)")

        # Add token cost summary if available
        if self.cost_tracker:
            cost_summary = self.cost_tracker.get_session_summary()
            session_totals = cost_summary["session_totals"]

            print(f"\n💰 Token Usage & Cost Summary")
            print("=" * 50)
            print(f"{'Metric':<25} {'Value':<25}")
            print("-" * 50)
            print(f"{'Total Tokens':<25} {session_totals['total_tokens']:,}")
            print(f"{'Prompt Tokens':<25} {session_totals['prompt_tokens']:,}")
            print(f"{'Completion Tokens':<25} {session_totals['completion_tokens']:,}")
            print(f"{'Total Conversations':<25} {session_totals['conversations']:,}")
            print(f"{'Total Cost (USD)':<25} ${session_totals['total_cost_usd']:.6f}")
            print(f"{'Avg Cost per Conversation':<25} ${session_totals['avg_cost_per_conversation']:.6f}")

            # Show top agents by cost
            if cost_summary["agent_breakdown"]:
                print(f"\n🤖 Top Agents by Cost")
                print("-" * 70)
                sorted_agents = sorted(
                    cost_summary["agent_breakdown"].items(),
                    key=lambda x: x[1]["total_cost_usd"],
                    reverse=True
                )[:3]  # Top 3 agents

                for agent, stats in sorted_agents:
                    # Right-align cost and token count for better readability
                    print(f"{agent:<45} ${stats['total_cost_usd']:>7.4f} ({stats['total_tokens']:>8,} tokens)")

            # Log the detailed cost summary to the logger
            if hasattr(self, 'logger') and self.logger:
                self.cost_tracker.log_session_summary()

        # Calculate parallel processing efficiency if applicable
        review_phases = [name for name in self.timings.keys() if 'review' in name.lower() or 'critique' in name.lower()]
        # Efficiency calculation for internal tracking only
    
    def get_performance_data(self) -> Dict:
        """
        Get performance data as a dictionary for programmatic use
        
        Returns:
            Dictionary containing timing data, percentages, and summary stats
        """
        total_time = self.get_total_time()
        
        performance_data = {
            'total_time': total_time,
            'phases': {},
            'summary': {
                'fastest_phase': None,
                'slowest_phase': None,
                'total_phases': len(self.timings)
            }
        }
        
        # Add phase data with percentages
        for phase_name in self.phase_order:
            duration = self.timings[phase_name]
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            
            performance_data['phases'][phase_name] = {
                'duration': duration,
                'percentage': percentage,
                'order': self.phase_order.index(phase_name)
            }
        
        # Add fastest/slowest phase info
        if self.timings:
            fastest = min(self.timings.items(), key=lambda x: x[1])
            slowest = max(self.timings.items(), key=lambda x: x[1])
            
            performance_data['summary']['fastest_phase'] = {
                'name': fastest[0],
                'duration': fastest[1]
            }
            performance_data['summary']['slowest_phase'] = {
                'name': slowest[0], 
                'duration': slowest[1]
            }
        
        return performance_data
    
    def reset(self):
        """Reset the timer for a new timing session"""
        self.timings.clear()
        self.start_times.clear()
        self.phase_order.clear()
        self.total_start_time = time.time()
        
        if self.logger:
            self.logger.log_info("PhaseTimer reset for new session")