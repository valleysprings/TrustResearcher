"""
Idea Generator Agent

Generates research ideas using literature-informed planning and graph-of-thought reasoning.
Supports cross-pollination, variant generation, and iterative refinement of ideas.
"""

import asyncio
import json
import re
import os
import ast
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional, Any, Tuple
from .base_agent import BaseAgent
from ..knowledge_graph.kg_builder import KGBuilder
from .idea_gen.planning_module import PlanningModule
from .idea_gen.graph_of_thought import GraphOfThought
from ..utils.llm_interface import LLMInterface
from ..utils.async_utils import limit_async_func_call, retry_with_timeout, async_batch_processor
from ..utils.text_utils import parse_json_response, safe_json_parse, create_json_prompt_suffix
from ..prompts.idea_generation.idea_generator_prompts import (
    CROSS_POLLINATION_SYSTEM_PROMPT, CROSS_POLLINATION_USER_PROMPT,
    EXPANSION_SYSTEM_PROMPT, EXPANSION_USER_PROMPT,
    REFINEMENT_SYSTEM_PROMPT, REFINEMENT_USER_PROMPT,
    SELF_CRITIQUE_SYSTEM_PROMPT, SELF_CRITIQUE_USER_PROMPT,
    VARIANT_SYSTEM_PROMPT, VARIANT_USER_PROMPT,
    GOT_VARIANT_SYSTEM_PROMPT,
    VALIDATION_ELABORATION_SYSTEM_PROMPT, VALIDATION_ELABORATION_USER_PROMPT,
)


class ResearchIdea:
    """Represents a complete research idea with all facets"""
    def __init__(self, topic: str, facets: Dict[str, str], source: str = "unknown", method: str = "unknown"):
        self.topic = topic
        self.facets = facets  # Problem Statement, Methodology, Validation
        self.source = source  # Which agent generated this (e.g., "idea_generator", "cross_pollination", "kg_extraction")
        self.method = method  # Which specific method (e.g., "got_reasoning", "variant_generation", "literature_expansion")
        self.refinement_history = []
        self.review_feedback = []
        self.novelty_score = None
        self.reasoning_context = {}
        
    def add_feedback(self, feedback: Dict):
        """Add review feedback to the idea"""
        self.review_feedback.append(feedback)
    
    def refine(self, new_facets: Dict[str, str], reasoning: str = ""):
        """Refine the idea facets"""
        self.refinement_history.append({
            'old_facets': self.facets.copy(),
            'new_facets': new_facets,
            'reasoning': reasoning
        })
        self.facets = new_facets
        
    def to_dict(self):
        result = {
            'topic': self.topic,
            'facets': self.facets,
            'source': self.source,
            'method': self.method,
            'review_feedback': self.review_feedback,
            'novelty_score': self.novelty_score,
            'reasoning_context': self.reasoning_context,
        }

        # Handle literature_context if it exists
        if hasattr(self, 'literature_context') and self.literature_context:
            result['literature_context'] = {}
            for key, value in self.literature_context.items():
                if key == 'referenced_papers' and isinstance(value, list):
                    # Convert Paper objects to dictionaries
                    result['literature_context'][key] = [
                        paper.to_dict() if hasattr(paper, 'to_dict') else paper
                        for paper in value
                    ]
                else:
                    result['literature_context'][key] = value

        return result
        
    def __str__(self):
        return f"""
            Research Idea: {self.topic}

            Problem Statement: {self.facets.get('Problem Statement', 'Not defined')}

            Proposed Methodology: {self.facets.get('Proposed Methodology', 'Not defined')}

            Experimental Validation: {self.facets.get('Experimental Validation', 'Not defined')}

            Source: {self.source} | Method: {self.method}
        """.strip()
    
    @property
    def problem_statement(self) -> str:
        """Get the problem statement from facets"""
        value = self.facets.get('Problem Statement', '')
        return str(value) if value else ''

    @property
    def proposed_methodology(self) -> str:
        """Get the proposed methodology from facets"""
        value = self.facets.get('Proposed Methodology', '')
        return str(value) if value else ''

    @property
    def experimental_validation(self) -> str:
        """Get the experimental validation from facets"""
        value = self.facets.get('Experimental Validation', '')
        return str(value) if value else ''

    @property
    def potential_impact(self) -> str:
        """Get the potential impact from facets"""
        value = self.facets.get('Potential Impact', '')
        return str(value) if value else ''


class IdeaGenerator(BaseAgent):
    """
    Enhanced idea generator that uses knowledge graphs and graph-of-thought reasoning
    to generate novel research ideas following the methodology from the paper.
    Now includes literature-informed planning and idea generation.
    """
    
    def __init__(self, knowledge_graph: KGBuilder, config: Dict = None, logger=None, llm_config: Dict = None):
        super().__init__("IdeaGenerator")
        self.knowledge_graph = knowledge_graph
        self.config = config or {}
        self.logger = logger
        self.llm = LLMInterface(config=llm_config, logger=logger)
        
        # Initialize reasoning modules
        self.planning_module = PlanningModule(knowledge_graph, self.llm, config=self.config)
        self.graph_of_thought = GraphOfThought(self.llm, config=self.config, knowledge_graph=knowledge_graph)
        
        # Generated ideas storage
        self.generated_ideas = []
        self.current_exploration_state = {}
        
        # Literature context storage
        self.literature_context = []
        self.literature_summary = ""

        # Use unified log structure from debug_logger
        # Idea logs will go to logs/idea/ directory
        self.idea_logs_dir = None
        if self.logger and hasattr(self.logger, 'idea_log_dir'):
            self.idea_logs_dir = self.logger.idea_log_dir
        else:
            # Fallback for backward compatibility
            self.idea_logs_dir = Path("logs/idea")
            self.idea_logs_dir.mkdir(parents=True, exist_ok=True)

    def _debug_log_idea(self, idea: 'ResearchIdea', source_method: str = "unknown"):
        """Log full idea details in debug mode"""
        if self.logger and hasattr(self.logger, 'debug_mode') and self.logger.debug_mode:
            debug_info = f"""
                ========== NEW RESEARCH IDEA GENERATED ==========
                Source Method: {source_method}
                Topic: {idea.topic}
                Source: {idea.source}
                Method: {idea.method}

                Problem Statement:
                {idea.problem_statement}

                Proposed Methodology:
                {idea.proposed_methodology}

                Experimental Validation:
                {idea.experimental_validation}

                Novelty Score: {getattr(idea, 'novelty_score', 'Not set')}
                =================================================""".strip()
            self.logger.log_debug(debug_info, "idea_generator")

        # Also do a brief info log even without debug mode
        elif self.logger:
            self.logger.log_info(f"Generated idea: '{idea.topic}' via {source_method} ({idea.method})", "idea_generator")

    def set_literature_context(self, papers: List):
        """Initialize literature context and precompute summary.

        Handles None/empty inputs gracefully and resets prior summary when needed.
        """
        papers = papers or []
        self.literature_context = papers
        # Initialize literature_summary using _summarize_literature only if papers exist
        self.literature_summary = self._summarize_literature(papers) if papers else ""
        if self.logger:
            self.logger.log_info(f"Set literature context with {len(papers)} papers", "idea_generator")

    def _summarize_literature(self, papers: List) -> str:
        """Create a comprehensive summary of literature findings"""
        if not papers:
            return ""
        
        summary_parts = []
        
        # Key findings and approaches
        summary_parts.append("RECENT LITERATURE FINDINGS:")
        for i, paper in enumerate(papers[:self.config.get('idea_generation', {}).get('literature_summary_papers', 5)], 1):
            authors_str = ", ".join(paper.authors[:self.config.get('idea_generation', {}).get('literature_summary_authors', 3)]) if paper.authors else "Unknown"
            summary_parts.append(f"{i}. {paper.title} ({paper.year}) by {authors_str}")
            if paper.abstract:
                summary_parts.append(f"   Key contribution: {paper.abstract[:self.config.get('idea_generation', {}).get('literature_abstract_length', 200)]}...")
            summary_parts.append(f"   Citations: {paper.citation_count}, Keywords: {', '.join(paper.keywords[:self.config.get('idea_generation', {}).get('literature_keywords', 5)])}")
        
        # Research gaps and opportunities
        summary_parts.append("\nIDENTIFIED RESEARCH PATTERNS:")
        common_keywords = self._extract_common_themes(papers)
        if common_keywords:
            summary_parts.append(f"Common research themes: {', '.join(common_keywords[:self.config.get('idea_generation', {}).get('common_themes', 10)])}")
        
        # Potential gaps
        summary_parts.append("\nPOTENTIAL RESEARCH GAPS:")
        summary_parts.append("Areas not fully addressed by existing literature require further investigation")
        
        return "\n".join(summary_parts)

    def _save_to_json_log(self, data: Dict, filename_prefix: str):
        """Save data to JSON file in logs/idea/ directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Unified naming: timestamp only (no prefix)
        filename = f"{timestamp}.json"
        filepath = self.idea_logs_dir / filename

        try:
            # Convert any non-serializable objects to dictionaries
            serializable_data = self._make_json_serializable(data)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            if self.logger:
                self.logger.log_info(f"Saved ideas to {filepath}", "idea_generator")
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Failed to save {filename_prefix} to JSON: {e}", "idea_generator")

    def _make_json_serializable(self, obj):
        """Recursively convert objects to JSON-serializable format"""
        if hasattr(obj, 'to_dict'):
            # Object has a to_dict method, use it
            return self._make_json_serializable(obj.to_dict())
        elif isinstance(obj, dict):
            # Recursively process dictionary values
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            # Recursively process list items
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            # Basic JSON-serializable types
            return obj
        else:
            # For other types, try to convert to string as fallback
            return str(obj)

    def _extract_common_themes(self, papers: List) -> List[str]:
        """Extract common themes and keywords from literature"""
        all_keywords = []
        for paper in papers:
            if paper.keywords:
                all_keywords.extend([k.lower() for k in paper.keywords])
        
        # Count frequency and return most common
        keyword_counts = Counter(all_keywords)
        return [keyword for keyword, count in keyword_counts.most_common(self.config.get('idea_generation', {}).get('common_themes', 10))]

    @limit_async_func_call(config_path='idea_generation.async_func_max_size')
    async def generate_ideas(self, seed_topic: str, num_ideas: int = None,
                            exploration_depth: int = None, overgenerate_factor: int = None) -> List[ResearchIdea]:
        """
        Main method to generate research ideas following the paper's methodology:
        1. Build knowledge graph from seed topic
        2. Use faceted decomposition via planning module  
        3. Employ graph-of-thought reasoning for creative exploration
        4. Generate multiple idea variants with duplicate removal
        
        Args:
            overgenerate_factor: Generate this many times more ideas than requested
                                (e.g., factor=10 means generate 10x ideas for later filtering)
        """
        # Use config defaults if not provided
        idea_cfg = self.config.get('idea_generation', {})
        num_ideas = num_ideas or idea_cfg.get('num_ideas', 3)
        exploration_depth = exploration_depth or idea_cfg.get('exploration_depth', 2)
        overgenerate_factor = overgenerate_factor or idea_cfg.get('overgeneration_factor', 1)

        try:
            # Add global timeout to entire generation process
            return await asyncio.wait_for(
                self._generate_ideas_internal(seed_topic, num_ideas, exploration_depth, overgenerate_factor),
                timeout=self.config.get('llm', {}).get('timeouts', {}).get('idea_generation', 1800)
            )
        except asyncio.TimeoutError:
            print(f"Error: Idea generation timed out after {self.config.get('llm', {}).get('timeouts', {}).get('idea_generation', 1800)//60} minutes")
            # Return whatever we have generated so far
            if hasattr(self, 'generated_ideas') and self.generated_ideas:
                return self.generated_ideas[:num_ideas]
            return []
    
    async def _generate_ideas_internal(self, seed_topic: str, num_ideas: int = None, 
                                     exploration_depth: int = None, overgenerate_factor: int = None) -> List[ResearchIdea]:
        """Internal method for idea generation with literature-informed planning"""
        
        # Use config defaults if not provided
        num_ideas = num_ideas or self.config.get('idea_generation', {}).get('num_ideas', 3)
        exploration_depth = exploration_depth or self.config.get('idea_generation', {}).get('exploration_depth', 2)
        overgenerate_factor = overgenerate_factor or self.config.get('idea_generation', {}).get('overgeneration_factor', 10)

        # GoT will use config from graph_of_thought section (no need to pass parameters)

        # Calculate actual number of ideas to generate
        target_generation_count = num_ideas * overgenerate_factor

        # Ensure literature_summary is initialized if we have literature_context
        if self.literature_context and not self.literature_summary:
            self.literature_summary = self._summarize_literature(self.literature_context)
        
        # Step 1: Build/expand knowledge graph with literature context
        print(f"Building literature-informed knowledge graph for topic: {seed_topic}")
        await self.knowledge_graph.build_from_seed_topic(seed_topic, literature_papers=self.literature_context)
        
        # Step 2: Literature-informed planning and faceted decomposition
        print("Performing literature-informed faceted decomposition...")
        plan = await self.planning_module.plan_research_steps(
            seed_topic,
            literature_papers=self.literature_context,
            literature_summary=self.literature_summary
        )

        # Step 3: Generate multiple ideas with different strategies
        print(f"Generating {target_generation_count} research ideas ({overgenerate_factor}x overgeneration)...")
        generated_ideas = []

        # Strategy 1: Generate detailed planning idea first
        planning_research_idea = ResearchIdea(
            topic=f"{seed_topic} (Initial Planning)",
            facets=plan['facets'],
            source="idea_generator",
            method="initial_planning"
        )
        # Debug log the generated idea
        self._debug_log_idea(planning_research_idea, "initial_planning")
        # Add literature context to the planning idea
        planning_research_idea.literature_context = {
            'referenced_papers': self.literature_context[:self.config.get('idea_generation', {}).get('top_papers_context', 3)],  # Top papers
            'literature_summary': self.literature_summary,
            'identified_gaps': plan.get('literature_gaps', [])
        }
        # Prepare graph-of-thought path sampling from KG
        try:
            got_reasoning_state = await self.graph_of_thought.build_graph_of_thoughts(
                seed_topic=seed_topic,
                facets=plan['facets'],
                knowledge_graph=self.knowledge_graph,
            )
            self.current_exploration_state = got_reasoning_state
            print(f"GoT sampled {len(got_reasoning_state.get('paths', []))} reasoning paths")
        except Exception as e:
            print(f"Warning: Graph-of-thought path sampling failed: {e}")
            if self.logger:
                self.logger.log_warning(f"Graph-of-thought path sampling failed: {e}", "idea_generator")
            self.current_exploration_state = {
                "paths": [],
                "facet_nodes": [],
                "cross_connections": [],
                "graph_summary": {},
            }

        # Strategy 2, 3 & 4: Generate variants using three strategies (base, GoT, pollination)
        # Total generation = 1.2x overgeneration_factor (configurable)
        total_multiplier = self.config.get('idea_generation', {}).get('total_generation_multiplier', 1.2)
        base_ratio = self.config.get('idea_generation', {}).get('base_variant_ratio', 0.4)
        got_ratio = self.config.get('idea_generation', {}).get('got_variant_ratio', 0.4)
        pollination_ratio = self.config.get('idea_generation', {}).get('pollination_ratio', 0.4)

        # Calculate actual counts based on overgeneration_factor
        overgen_base = overgenerate_factor * num_ideas
        total_to_generate = int(overgen_base * total_multiplier)
        base_variant_count = int(overgen_base * base_ratio)
        got_variant_count = int(overgen_base * got_ratio)
        pollination_count = int(overgen_base * pollination_ratio)

        # Adjust to ensure we hit the total
        total_allocated = base_variant_count + got_variant_count + pollination_count
        if total_allocated < total_to_generate:
            # Add remainder to base variants
            base_variant_count += (total_to_generate - total_allocated)

        print(f"Generating {total_to_generate} total ideas: {base_variant_count} base variants, {got_variant_count} GoT variants, {pollination_count} pollination variants")

        if base_variant_count > 0 or got_variant_count > 0:

            # Create tasks for concurrent variant generation
            variant_tasks = []
            # Start numbering from 1 for variants after the initial planning idea
            for i in range(base_variant_count):
                task = self.generate_variant_idea_async(seed_topic, planning_research_idea, i+1)
                variant_tasks.append(task)

            for i in range(got_variant_count):
                task = self.generate_got_variant_async(
                    seed_topic,
                    plan['facets'],
                    base_variant_count + i + 1,
                    reasoning_state=self.current_exploration_state,
                )
                variant_tasks.append(task)
            
            # Execute variant generation concurrently with timeout
            if variant_tasks:
                try:
                    variant_results = await asyncio.wait_for(
                        asyncio.gather(*variant_tasks, return_exceptions=True),
                        timeout=self.config.get('llm', {}).get('timeouts', {}).get('variant_generation', 900)
                    )
                    for variant_idea in variant_results:
                        if isinstance(variant_idea, ResearchIdea) and not self.is_duplicate_idea(variant_idea, generated_ideas):
                            generated_ideas.append(variant_idea)
                        elif isinstance(variant_idea, Exception):
                            print(f"Warning: Variant generation task failed: {variant_idea}")
                            if self.logger:
                                self.logger.log_warning(f"Variant generation failed: {variant_idea}", "idea_generator")
                except asyncio.TimeoutError:
                    print(f"Warning: Variant generation timed out after {self.config.get('llm', {}).get('timeouts', {}).get('variant_generation', 900)//60} minutes")

        # Strategy 4: Cross-pollination as a primary strategy (not fallback)
        if pollination_count > 0 and len(generated_ideas) >= self.config.get('idea_generation', {}).get('min_ideas_for_cross_poll', 2):
            print(f"Generating {pollination_count} cross-pollination variants")
            try:
                # Generate pollination_count ideas through cross-pollination
                pollination_tasks = []
                # Use existing ideas as seed for cross-pollination
                seed_ideas = generated_ideas[:min(self.config.get('idea_generation', {}).get('max_ideas_for_cross_poll', 5), len(generated_ideas))]

                # Generate multiple batches if needed
                for i in range(pollination_count):
                    cross_pollinated = await self.generate_cross_pollinated_ideas_async(seed_ideas)
                    for cp_idea in cross_pollinated:
                        if not self.is_duplicate_idea(cp_idea, generated_ideas):
                            generated_ideas.append(cp_idea)
                            break  # Only take one per iteration
                    if len(generated_ideas) >= total_to_generate:
                        break

            except Exception as e:
                print(f"Warning: Cross-pollination generation failed: {e}")
                if self.logger:
                    self.logger.log_warning(f"Cross-pollination generation failed: {e}", "idea_generator")

        # Final check and summary
        print(f"Generated {len(generated_ideas)} total ideas (attempted {total_to_generate}, will trim to {target_generation_count})")

        # Trim to target count (10x overgeneration) if we generated more
        if len(generated_ideas) > target_generation_count:
            print(f"Trimming from {len(generated_ideas)} to {target_generation_count} ideas for selection pipeline")
            generated_ideas = generated_ideas[:target_generation_count]

        # Check if self-critique is enabled
        enable_self_critique = self.config.get('idea_generation', {}).get('enable_self_critique', True)

        # Keep a copy of originals for logging and refinement tracking
        originals = [idea.facets.copy() for idea in generated_ideas]

        if enable_self_critique:
            # Self-critique all ideas (async) and then apply revisions via LLM
            print("Running self-critique on generated ideas and applying revisions...")

            # 1) Critique
            critique_tasks = [self.self_critique_idea(idea) for idea in generated_ideas]
            critiques = await asyncio.gather(*critique_tasks, return_exceptions=True)

            default_overall = self.config.get('idea_generation', {}).get('default_overall_score', self.config.get('idea_generation', {}).get('default_facet_score', 3))

            # Normalize critiques and attach feedback
            normalized_critiques: List[Dict] = []
            for idea, crit in zip(generated_ideas, critiques):
                if isinstance(crit, Exception) or not isinstance(crit, dict):
                    # Fallback minimal critique
                    crit = {
                        "overall_score": default_overall,
                        "strengths": ["Coherent"],
                        "weaknesses": ["Needs more specificity"],
                        "suggestions": ["Clarify evaluation metrics"]
                    }
                idea.add_feedback({"type": "self_critique", **crit})
                normalized_critiques.append(crit)

            # 2) Apply critique (revise with LLM)
            apply_tasks = [self.apply_critique_to_idea_async(idea, crit) for idea, crit in zip(generated_ideas, normalized_critiques)]
            revised_facets_list = await asyncio.gather(*apply_tasks, return_exceptions=True)

            # Update ideas with revisions
            for idx, (idea, revised_facets) in enumerate(zip(generated_ideas, revised_facets_list)):
                if isinstance(revised_facets, Exception) or not isinstance(revised_facets, dict) or not any(revised_facets.values()):
                    # Fallback: keep original facets if revision failed
                    revised_facets = idea.facets.copy()

                # Track refinement history and set revised facets
                idea.refine(revised_facets, reasoning="Applied self-critique suggestions via LLM refinement")
        else:
            print("Self-critique disabled, skipping round 2 refinement...")
            normalized_critiques = [{
                "overall_score": self.config.get('idea_generation', {}).get('default_overall_score', 4.0),
                "strengths": ["Initial plan"],
                "weaknesses": [],
                "suggestions": []
            } for _ in generated_ideas]

        post_self_critique_facets = [idea.facets.copy() for idea in generated_ideas]

        # Check if elaboration is enabled
        enable_elaboration = self.config.get('idea_generation', {}).get('enable_elaboration', True)

        # Experimental validation elaboration (round 3)
        elaboration_cfg = self.config.get('idea_generation', {}).get('methodology_elaboration', {})
        if enable_elaboration and elaboration_cfg.get('enabled', True) and generated_ideas:
            print("Elaborating experimental validation facets for each idea...")
            await self.elaborate_methodology_facets_async(generated_ideas, elaboration_cfg)
        elif not enable_elaboration:
            print("Elaboration disabled, skipping round 3 refinement...")

        post_elaboration_facets = [idea.facets.copy() for idea in generated_ideas]

        # Build logging sections based on enabled features
        logger_sections = ["initial"]  # Always include initial
        if enable_self_critique:
            logger_sections.append("self_critique")
        if enable_elaboration:
            logger_sections.append("elaboration")

        combined_records = []

        for idx, (idea, original_facets, self_facets, final_facets, critique) in enumerate(zip(
            generated_ideas,
            originals,
            post_self_critique_facets,
            post_elaboration_facets,
            normalized_critiques,
        )):

            refinement_rounds = []

            # Round 1: Always include initial
            refinement_rounds.append({
                "round": 1,
                "label": "initial",
                "reason": "Initial plan output",
                "facets": original_facets,
            })

            # Round 2: Only if self-critique is enabled
            if enable_self_critique:
                refinement_rounds.append({
                    "round": 2,
                    "label": "self_critique",
                    "reason": "Applied critique refinement",
                    "facets": self_facets,
                    "critique": critique,
                })

            # Round 3: Only if elaboration is enabled
            validation_context = {}
            if hasattr(idea, 'reasoning_context'):
                validation_context = idea.reasoning_context.get('validation_elaboration', {})

            if enable_elaboration and (validation_context or final_facets != self_facets):
                refinement_rounds.append({
                    "round": 3,
                    "label": "elaboration",
                    "reason": "Experimental validation elaboration (critique integration)",
                    "facets": final_facets,
                })

            combined_records.append({
                "topic": idea.topic,
                "source": idea.source,
                "method": idea.method,
                "refinement_rounds": refinement_rounds,
            })

        # Set final ideas (revised)
        self.generated_ideas = generated_ideas

        print(f"✅ LITERATURE-INFORMED IDEA GENERATION COMPLETE: Generated {len(self.generated_ideas)} revised ideas (target: {target_generation_count})")

        # Save all generated ideas to JSON file with original+revised versions
        ideas_data = {
            "timestamp": datetime.now().isoformat(),
            "seed_topic": seed_topic,
            "target_generation_count": target_generation_count,
            "actual_generation_count": len(self.generated_ideas),
            "overgenerate_factor": overgenerate_factor,
            "ideas": combined_records,
            "literature_context_count": len(self.literature_context)
        }
        self._save_to_json_log(ideas_data, "ideas")

        # Return revised ideas
        return self.generated_ideas

    @limit_async_func_call(config_path='idea_generation.async_func_max_size')
    @retry_with_timeout(max_retries=5, timeout=300, delay=2)
    async def self_critique_idea(self, idea: ResearchIdea) -> Dict:
        """Perform self-critique on a generated idea (async)."""
        user_prompt = SELF_CRITIQUE_USER_PROMPT.format(idea=idea)
        max_tokens = self.config.get('llm', {}).get('max_tokens', 16384)
        response = await self.llm.generate_with_system_prompt(
            SELF_CRITIQUE_SYSTEM_PROMPT,
            user_prompt,
            max_tokens=max_tokens,
            caller="idea_generator_self_critique",
            task_type="idea_generation"
        )
        critique = self.parse_critique_response(response)
        return critique

    def parse_critique_response(self, response: str) -> Dict:
        """Parse critique response into structured data using JSON-first approach"""
        default_facet = self.config.get('idea_generation', {}).get('default_facet_score', 3)
        default_overall = self.config.get('idea_generation', {}).get('default_overall_score', default_facet)
        critique = {
            "overall_score": default_overall,
            "novelty_score": default_facet,
            "feasibility_score": default_facet,
            "clarity_score": default_facet,
            "impact_score": default_facet,
            "needs_refinement": True,
            "strengths": ["Well-structured research approach"],
            "weaknesses": ["Could benefit from more specific metrics"],
            "suggestions": ["Consider additional validation methods"]
        }

        # Try JSON parsing first
        json_result = safe_json_parse(response)

        if json_result and isinstance(json_result, dict):
            # Handle JSON structure - parse individual scores but NOT overall_score
            # (overall_score will be calculated from individual scores in _finalize_critique_scores)
            # Scores may appear in various shapes
            for k in ["novelty_score", "feasibility_score", "clarity_score", "impact_score"]:
                if k in json_result:
                    try:
                        critique[k] = float(json_result[k])
                    except Exception:
                        pass
            if "criteria_scores" in json_result and isinstance(json_result["criteria_scores"], dict):
                cs = json_result["criteria_scores"]
                for k in ["novelty", "feasibility", "clarity", "impact"]:
                    v = cs.get(k)
                    if isinstance(v, (int, float)):
                        critique[f"{k}_score"] = float(v)
                    elif isinstance(v, str):
                        try:
                            critique[f"{k}_score"] = float(v)
                        except Exception:
                            pass
            # Needs refinement flag
            if "needs_refinement" in json_result:
                val = json_result.get("needs_refinement")
                if isinstance(val, str):
                    critique["needs_refinement"] = val.strip().lower() in {"yes", "true", "y"}
                elif isinstance(val, bool):
                    critique["needs_refinement"] = val
            critique["strengths"] = json_result.get("strengths", critique["strengths"])
            critique["weaknesses"] = json_result.get("weaknesses", critique["weaknesses"])
            critique["suggestions"] = json_result.get("suggestions", critique["suggestions"])
            return self._finalize_critique_scores(critique, default_overall)

        # Fallback to text parsing
        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            # Skip generic 'score' lines - we only want specific criteria scores
            # (overall_score will be calculated from criteria scores in _finalize_critique_scores)
            # Per-criteria scores
            for key in ["novelty", "feasibility", "clarity", "impact"]:
                if key in line.lower():
                    try:
                        m = re.search(r'(\d+(?:\.\d+)?)', line)
                        if m:
                            critique[f"{key}_score"] = float(m.group(1))
                    except Exception:
                        pass
            if 'needs_refinement' in line.lower():
                critique['needs_refinement'] = 'yes' in line.lower() or 'true' in line.lower()
            elif 'strength' in line.lower():
                current_section = "strengths"
            elif 'weakness' in line.lower():
                current_section = "weaknesses"
            elif 'suggestion' in line.lower():
                current_section = "suggestions"
            elif line.startswith('-') or line.startswith('•'):
                if current_section and current_section in critique:
                    critique[current_section].append(line[1:].strip())

        # If overall score missing or default, derive from available facet scores
        return self._finalize_critique_scores(critique, default_overall)

    def _finalize_critique_scores(self, critique: Dict, default_overall: float) -> Dict:
        """Ensure numeric scores are floats and derive overall score if missing."""
        if critique is None:
            return {}

        score_fields = ["overall_score", "novelty_score", "feasibility_score", "clarity_score", "impact_score"]
        for key in score_fields:
            value = critique.get(key)
            if isinstance(value, str):
                try:
                    critique[key] = float(value)
                except Exception:
                    continue

        weights = self._get_critique_weights()
        facet_keys = ["novelty_score", "feasibility_score", "clarity_score", "impact_score"]
        facet_values = []
        for key in facet_keys:
            value = critique.get(key)
            if isinstance(value, (int, float)):
                facet_values.append(float(value))

        weighted_total = 0.0
        total_weight = 0.0
        for weight_key, weight in weights.items():
            score_key = f"{weight_key}_score"
            score = critique.get(score_key)
            if isinstance(score, str):
                try:
                    score = float(score)
                except Exception:
                    score = None
            if isinstance(score, (int, float)):
                weighted_total += float(score) * weight
                total_weight += weight

        if total_weight > 0:
            critique['overall_score'] = round(weighted_total / total_weight, 2)
        else:
            overall = critique.get('overall_score')
            if isinstance(overall, (int, float)):
                critique['overall_score'] = float(overall)
            elif facet_values:
                critique['overall_score'] = round(sum(facet_values) / len(facet_values), 2)
            else:
                critique['overall_score'] = float(default_overall)

        return critique

    def _get_critique_weights(self) -> Dict[str, float]:
        """Fetch critique weighting configuration, defaulting to uniform weights."""
        weights_cfg = self.config.get('idea_generation', {}).get('critique_weights')
        if not weights_cfg:
            return {
                'novelty': 1.0,
                'feasibility': 1.0,
                'clarity': 1.0,
                'impact': 1.0,
            }

        weights: Dict[str, float] = {}
        for key, value in weights_cfg.items():
            try:
                weights[key.lower()] = float(value)
            except Exception:
                continue

        # Ensure at least one weight exists
        if not weights:
            weights = {
                'novelty': 1.0,
                'feasibility': 1.0,
                'clarity': 1.0,
                'impact': 1.0,
            }
        return weights

    @limit_async_func_call(config_path='idea_generation.async_func_max_size')
    @retry_with_timeout(max_retries=5, timeout=300, delay=2)
    async def apply_critique_to_idea_async(self, idea: ResearchIdea, critique: Dict) -> Dict[str, str]:
        """Apply critique by performing an LLM-guided refinement to improve idea facets (async)."""
        # Prepare inputs
        suggestions_text = "\n".join(critique.get('suggestions', [])) if critique.get('suggestions') else ""
        user_prompt = REFINEMENT_USER_PROMPT.format(
            idea=str(idea),
            suggestions=suggestions_text,
            novelty_score=critique.get('novelty_score', self.config.get('idea_generation', {}).get('default_facet_score', 3)),
            feasibility_score=critique.get('feasibility_score', self.config.get('idea_generation', {}).get('default_facet_score', 3)),
            clarity_score=critique.get('clarity_score', self.config.get('idea_generation', {}).get('default_facet_score', 3)),
            impact_score=critique.get('impact_score', self.config.get('idea_generation', {}).get('default_facet_score', 3)),
        ) + "\n\n" + create_json_prompt_suffix()

        response = await self.llm.generate_with_system_prompt(
            REFINEMENT_SYSTEM_PROMPT,
            user_prompt,
            max_tokens=None,
            caller="idea_generator_refinement",
            task_type="idea_generation"
        )

        # Parse refined facets with robust fallback
        refined_facets = parse_json_response(
            response,
            ["topic", "Problem Statement", "Proposed Methodology", "Experimental Validation"],
            fallback_type="facets"
        )

        return refined_facets

    async def elaborate_methodology_facets_async(self, ideas: List[ResearchIdea], config: Dict):
        """Elaborate methodology and validation facets for a list of ideas."""
        if not ideas:
            return

        validation_components = max(2, int(config.get('validation_components', 4)))
        batch_size = max(1, int(config.get('async_batch_size', 3)))

        for i in range(0, len(ideas), batch_size):
            batch = ideas[i:i + batch_size]
            tasks = [
                self.elaborate_experimental_validation_async(
                    idea,
                    validation_components=validation_components,
                    suggestions=self._collect_self_critique_suggestions(idea),
                )
                for idea in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for idea, result in zip(batch, results):
                if isinstance(result, dict) and result.get('Experimental Validation'):
                    updated_facets = idea.facets.copy()
                    updated_facets['Experimental Validation'] = self._normalize_validation_output(result['Experimental Validation'])

                    idea.refine(updated_facets, reasoning="Experimental validation elaboration")
                    idea.reasoning_context.setdefault('validation_elaboration', result)
                elif isinstance(result, Exception) and self.logger:
                    self.logger.log_warning(f"Validation elaboration failed: {result}", "idea_generator")

    @retry_with_timeout(max_retries=4, timeout=360, delay=2)
    async def elaborate_experimental_validation_async(
        self,
        idea: ResearchIdea,
        validation_components: int = 4,
        suggestions: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Expand the experimental validation facet with implementation-ready details."""
        suggestions = suggestions or []
        suggestions_text = "\n".join(f"- {s}" for s in suggestions if s)
        if not suggestions_text:
            suggestions_text = "(no additional critique suggestions supplied)"

        user_prompt = VALIDATION_ELABORATION_USER_PROMPT.format(
            topic=idea.topic,
            problem=idea.facets.get('Problem Statement', ''),
            methodology=idea.facets.get('Proposed Methodology', ''),
            validation=idea.facets.get('Experimental Validation', ''),
            validation_components=validation_components,
            critique_suggestions=suggestions_text,
            json_format_suffix=create_json_prompt_suffix()
        )

        response = await self.llm.generate_with_system_prompt(
            VALIDATION_ELABORATION_SYSTEM_PROMPT,
            user_prompt,
            max_tokens=self.config.get('llm', {}).get('max_tokens', 16384),
            caller="idea_generator_validation_elaboration",
            task_type="idea_generation"
        )

        elaborated_facets = parse_json_response(
            response,
            ["Experimental Validation"],
            fallback_type="facets"
        )

        return elaborated_facets

    def _collect_self_critique_suggestions(self, idea: ResearchIdea) -> List[str]:
        """Retrieve the latest self-critique suggestions for an idea."""
        if not hasattr(idea, 'review_feedback'):
            return []

        for feedback in reversed(idea.review_feedback):
            if isinstance(feedback, dict) and feedback.get('type') == 'self_critique':
                suggestions = feedback.get('suggestions') or []
                return [s for s in suggestions if isinstance(s, str)]

        return []

    def _normalize_validation_output(self, validation) -> str:
        """Ensure experimental validation facet is a well-formed paragraph string."""
        if isinstance(validation, str):
            stripped = validation.strip()
            if stripped.startswith(("[", "{")):
                try:
                    parsed = ast.literal_eval(stripped)
                except (ValueError, SyntaxError):
                    return stripped
                else:
                    return self._normalize_validation_output(parsed)
            return stripped

        if isinstance(validation, list):
            cleaned_items = []
            for item in validation:
                if isinstance(item, str):
                    cleaned_items.append(item.strip())
                elif isinstance(item, dict):
                    parts = []
                    for key, value in item.items():
                        parts.append(f"{key}: {value}")
                    cleaned_items.append("; ".join(parts))
                else:
                    cleaned_items.append(str(item))
            return ' '.join(cleaned_items)

        if isinstance(validation, dict):
            return "; ".join(f"{k}: {v}" for k, v in validation.items())

        return str(validation)

    def get_generated_ideas(self) -> List[ResearchIdea]:
        """Get the list of generated ideas"""
        return self.generated_ideas

    def parse_variant_response(self, response: str, seed_topic: str, variant_num: int) -> ResearchIdea:
        """Parse variant response into ResearchIdea"""
        variant_facets = parse_json_response(response, ["topic", "Problem Statement", "Proposed Methodology", "Experimental Validation"], fallback_type="facets")

        # Use extracted topic if available, otherwise fallback to generated format
        topic = variant_facets.get("topic", f"{seed_topic} - Variant {variant_num}")

        variant_idea = ResearchIdea(
            topic=topic,
            facets=variant_facets,
            source="idea_generator",
            method="got_variant"
        )
        # Debug log the generated idea
        self._debug_log_idea(variant_idea, "parse_variant_response")
        return variant_idea

    def parse_got_response(self, response: str, seed_topic: str, direction: str, variant_num: int) -> ResearchIdea:
        """Parse GoT response into ResearchIdea"""
        got_facets = parse_json_response(response, ["topic", "Problem Statement", "Proposed Methodology", "Experimental Validation"], fallback_type="facets")

        # Use extracted topic if available, otherwise fallback to generated format
        topic = got_facets.get("topic", f"{seed_topic} - GoT {direction.title()} {variant_num}")

        got_idea = ResearchIdea(
            topic=topic,
            facets=got_facets,
            source="idea_generator",
            method="got_reasoning"
        )
        # Debug log the generated idea
        self._debug_log_idea(got_idea, "parse_got_response")
        return got_idea

    def is_duplicate_idea(self, new_idea: ResearchIdea, existing_ideas: List[ResearchIdea]) -> bool:
        """Check if a new idea is a duplicate of existing ideas"""
        new_topic = new_idea.topic.lower().strip()
        new_problem = new_idea.facets.get('Problem Statement', '').lower().strip()
        
        for existing in existing_ideas:
            existing_topic = existing.topic.lower().strip()
            existing_problem = existing.facets.get('Problem Statement', '').lower().strip()
            
            # Check topic similarity
            if new_topic == existing_topic:
                return True
                
            # Check problem statement similarity
            if new_problem and existing_problem:
                if new_problem == existing_problem:
                    return True
        
        return False

    # Async variant methods  
    @retry_with_timeout(max_retries=5, timeout=300, delay=2)
    async def generate_variant_idea_async(self, seed_topic: str, base_idea: 'ResearchIdea', variant_num: int) -> ResearchIdea:
        """Asynchronously generate a variant of a research idea"""
        try:
            # Define different approaches for variants
            approaches = ["methodological innovation", "interdisciplinary application", "scalability enhancement", "novel validation"]
            approach = approaches[variant_num % len(approaches)]

            base_facets = base_idea.facets if base_idea else {}
            base_facets_context = self._format_facets_context(base_facets)
            base_description_parts = []
            topic_label = base_idea.topic if base_idea and base_idea.topic else seed_topic
            base_description_parts.append(f"Topic: {topic_label}")
            if base_facets_context:
                base_description_parts.append(base_facets_context)
            base_idea_description = "\n".join(base_description_parts)

            user_prompt = VARIANT_USER_PROMPT.format(
                seed_topic=seed_topic,
                base_idea=base_idea_description,
                approach=approach,
                json_format_suffix=create_json_prompt_suffix()
            )
            
            response = await self.llm.generate_with_system_prompt(
                VARIANT_SYSTEM_PROMPT.format(approach=approach),
                user_prompt, 
                max_tokens=None,  # Use configured variant_generation context window
                caller="idea_generator_variant"
            )
            
            return self.parse_variant_response(response, seed_topic, variant_num)
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Failed to generate variant {variant_num}: {e}", "idea_generator")
            # Return a fallback idea
            fallback_idea = ResearchIdea(
                topic=f"{seed_topic} - Variant {variant_num}",
                facets=base_facets.copy() if isinstance(base_facets, dict) else {},
                source="idea_generator",
                method="variant_fallback"
            )
            # Debug log the fallback idea
            self._debug_log_idea(fallback_idea, "async_variant_fallback")
            return fallback_idea

    @retry_with_timeout(max_retries=5, timeout=300, delay=2)
    async def generate_got_variant_async(
        self,
        seed_topic: str,
        base_facets: Dict[str, str],
        variant_num: int,
        reasoning_state: Optional[Dict[str, Any]] = None,
    ) -> ResearchIdea:
        """Generate a GoT-guided variant that leverages KG-grounded exploration."""

        exploration_directions = [
            "theoretical foundations",
            "practical applications",
            "computational efficiency",
            "data requirements",
            "evaluation methods",
            "broader impact",
        ]

        direction = exploration_directions[variant_num % len(exploration_directions)]
        system_prompt = GOT_VARIANT_SYSTEM_PROMPT.format(direction=direction)

        state = reasoning_state or self.current_exploration_state or {}
        path_context, path_metadata = self._render_reasoning_path_context(state, variant_num)
        cross_context = self._format_cross_connections(state)
        facet_context = self._format_facets_context(base_facets)
        graph_summary_context = self._format_graph_summary(state)

        context_sections = []
        if facet_context:
            context_sections.append(f"Core plan facets:\n{facet_context}")
        if path_context:
            context_sections.append(f"Graph-of-thought path insights:\n{path_context}")
        if cross_context:
            context_sections.append(f"Knowledge-graph bridges:\n{cross_context}")
        if graph_summary_context:
            context_sections.append(f"Idea graph summary:\n{graph_summary_context}")

        context_block = "\n\n".join(context_sections)

        user_prompt_parts = [
            f"Base Research Area: {seed_topic}",
            f"Exploration Direction: {direction}",
        ]
        if context_block:
            user_prompt_parts.append(f"Grounding Context:\n{context_block}")
        user_prompt_parts.append(
            "Generate a research idea that specifically focuses on "
            f"{direction} while being distinct from general approaches."
        )
        user_prompt_parts.append(create_json_prompt_suffix())

        user_prompt = "\n\n".join(user_prompt_parts)

        response = await self.llm.generate_with_system_prompt(
            system_prompt,
            user_prompt,
            max_tokens=self.config.get('llm', {}).get('max_tokens', 16384),
            caller="idea_generator_got_variant",
        )

        got_idea = self.parse_got_response(response, seed_topic, direction, variant_num)
        got_idea.reasoning_context = {
            "direction": direction,
            "path": path_metadata,
            "cross_connections": state.get('cross_connections', []),
            "graph_summary": state.get('graph_summary', {}),
        }
        return got_idea

    def _truncate_text(self, text: str, limit: int = 280) -> str:
        text = (text or "").strip()
        if len(text) <= limit:
            return text
        truncated = text[:limit].rsplit(' ', 1)[0]
        return f"{truncated}..."

    def _format_facets_context(self, facets: Dict[str, str]) -> str:
        if not facets:
            return ""

        ordered_keys = ["Problem Statement", "Proposed Methodology", "Experimental Validation", "Potential Impact"]
        parts = []
        for key in ordered_keys:
            value = facets.get(key)
            if not value:
                continue
            parts.append(f"{key}: {self._truncate_text(str(value), 220)}")
        return "\n".join(parts)

    def _render_reasoning_path_context(
        self,
        reasoning_state: Dict[str, Any],
        variant_num: int,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Render a GoT reasoning path as context for idea generation.

        New format: paths contain nodes + edges information
        """
        paths: List[Dict] = reasoning_state.get('paths', [])
        if not paths:
            return "", {}

        # Cycle through paths
        path_index = (variant_num - 1) % len(paths)
        selected_path = paths[path_index]

        # Extract path components
        path_nodes = selected_path.get('nodes', [])
        path_edges = selected_path.get('edges', [])
        path_score = selected_path.get('score', 0.0)
        path_string = selected_path.get('path_string', '')

        # Format as readable context
        context_lines = []
        context_lines.append(f"=== Reasoning Path (Quality Score: {path_score:.2f}) ===")
        context_lines.append("")

        # Access thought graph to get node details
        graph = self.graph_of_thought.thought_graph

        # Format nodes with details
        for i, node_id in enumerate(path_nodes):
            if node_id not in graph:
                continue

            thought = graph.nodes[node_id]['thought']
            thought_type = thought.thought_type
            content = self._truncate_text(thought.content, 200)

            context_lines.append(f"  [{i+1}] {thought_type.upper()}: {content}")

            # Add relationship to next node
            if i < len(path_edges):
                edge = path_edges[i]
                relation = edge.get('relation', 'connects')
                edge_type = edge.get('edge_type', '')
                context_lines.append(f"      └─ ({edge_type}) --{relation}--> ")

        context_lines.append("")
        context_lines.append(f"Path Summary: {path_string[:300]}...")

        # Build metadata
        path_metadata = {
            "path_index": path_index,
            "num_nodes": len(path_nodes),
            "num_edges": len(path_edges),
            "score": path_score,
            "nodes": path_nodes,
            "edges": path_edges,
        }

        return "\n".join(context_lines), path_metadata

    def _format_cross_connections(self, reasoning_state: Dict[str, Any]) -> str:
        connections = reasoning_state.get('cross_connections') or []
        limit = 5  # Show top 5 cross-cluster connections
        formatted = []
        for entity1, relationship, entity2 in connections[:limit]:
            formatted.append(f"{entity1} --{relationship}--> {entity2}")
        return "\n".join(formatted)

    def _format_graph_summary(self, reasoning_state: Dict[str, Any]) -> str:
        """Format GoT graph summary for context"""
        summary = reasoning_state.get('graph_summary') or {}
        if not summary:
            return ""

        parts = []

        # Core stats
        num_thoughts = summary.get('num_thoughts', 0)
        num_edges = summary.get('num_edges', 0)
        num_paths = summary.get('num_paths', 0)

        if num_thoughts:
            parts.append(f"Thoughts: {num_thoughts}")
        if num_edges:
            parts.append(f"Connections: {num_edges}")
        if num_paths:
            parts.append(f"Paths: {num_paths}")

        # Thought types
        thought_types = summary.get('thought_types') or {}
        if thought_types:
            type_str = ', '.join(f"{k}={v}" for k, v in list(thought_types.items())[:3])
            parts.append(f"Types: {type_str}")

        # Path quality
        avg_path_score = summary.get('avg_path_score', 0.0)
        if avg_path_score > 0:
            parts.append(f"Avg Path Quality: {avg_path_score:.2f}")

        return " | ".join(parts) if parts else ""
    
    @limit_async_func_call(config_path='idea_generation.async_func_max_size')
    async def generate_cross_pollinated_ideas_async(self, base_ideas: List[ResearchIdea]) -> List[ResearchIdea]:
        """Async version of generate_cross_pollinated_ideas"""
        cross_pollinated = []
        
        if len(base_ideas) < self.config.get('idea_generation', {}).get('min_ideas_for_cross_poll', 2):
            return cross_pollinated
        
        # Get cross-cluster connections from knowledge graph for inspiration
        cross_connections = self.knowledge_graph.get_cross_cluster_connections()
        
        # Create tasks for cross-pollination pairs
        tasks = []
        for i, idea1 in enumerate(base_ideas):
            for idea2 in base_ideas[i+1:]:
                task = self._cross_pollinate_pair_async(idea1, idea2, cross_connections[:self.config.get('idea_generation', {}).get('cross_connections', 2)])
                tasks.append(task)
        
        # Execute cross-pollination concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, ResearchIdea):
                cross_pollinated.append(result)
                if len(cross_pollinated) >= self.config.get('idea_generation', {}).get('hybrid_ideas_limit', 3):  # Limit hybrid ideas
                    break
                    
        return cross_pollinated[:self.config.get('idea_generation', {}).get('hybrid_ideas_final_limit', 1)]  # Limit hybrid ideas
    
    async def _cross_pollinate_pair_async(self, idea1: ResearchIdea, idea2: ResearchIdea, cross_connections: List) -> Optional[ResearchIdea]:
        """Cross-pollinate a pair of ideas asynchronously"""
        user_prompt = CROSS_POLLINATION_USER_PROMPT.format(
            idea1=idea1,
            idea2=idea2,
            cross_connections=cross_connections,
            json_format_suffix=create_json_prompt_suffix()
        )
        
        response = await self.llm.generate_with_system_prompt(CROSS_POLLINATION_SYSTEM_PROMPT, user_prompt, max_tokens=None, caller="idea_generator")
        hybrid_facets = parse_json_response(response, ["topic", "Problem Statement", "Proposed Methodology", "Experimental Validation"], fallback_type="facets")

        if any(hybrid_facets.values()):  # Only add if we got meaningful facets
            # Use extracted topic if available, otherwise fallback to generated format
            topic = hybrid_facets.get("topic", f"Hybrid: {idea1.topic} + {idea2.topic}")

            hybrid_idea = ResearchIdea(
                topic=topic,
                facets=hybrid_facets,
                source="idea_generator",
                method="cross_pollination"
            )
            # Debug log the generated idea
            self._debug_log_idea(hybrid_idea, "async_cross_pollination")
            return hybrid_idea
        return None
    
    @limit_async_func_call(config_path='idea_generation.expansion_async_max_size')
    async def expand_idea_exploration_async(self, idea: ResearchIdea, expansion_directions: List[str] = None) -> List[ResearchIdea]:
        """Async version of expand_idea_exploration"""
        if expansion_directions is None:
            expansion_directions = ["methodology", "application_domain", "evaluation_approach"]
        
        # Create tasks for concurrent expansion
        expansion_tasks = []
        for direction in expansion_directions:
            task = self._expand_idea_direction_async(idea, direction)
            expansion_tasks.append(task)
        
        # Execute expansions concurrently with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*expansion_tasks, return_exceptions=True),
                timeout=self.config.get('llm', {}).get('timeouts', {}).get('cross_pollination', 600)
            )
        except asyncio.TimeoutError:
            print(f"Warning: Idea expansion timed out after {self.config.get('llm', {}).get('timeouts', {}).get('cross_pollination', 600)//60} minutes")
            return []
        
        expanded_ideas = []
        for result in results:
            if isinstance(result, ResearchIdea):
                expanded_ideas.append(result)
            elif isinstance(result, Exception):
                print(f"Warning: Expansion task failed: {result}")
                
        return expanded_ideas
    
    async def _expand_idea_direction_async(self, idea: ResearchIdea, direction: str) -> Optional[ResearchIdea]:
        """Expand an idea in a specific direction asynchronously"""
        system_prompt = EXPANSION_SYSTEM_PROMPT.format(direction=direction)
        user_prompt = EXPANSION_USER_PROMPT.format(
            idea=idea,
            direction=direction,
            json_format_suffix=create_json_prompt_suffix()
        )
        
        response = await self.llm.generate_with_system_prompt(system_prompt, user_prompt, max_tokens=None, caller="idea_generator")
        variant_facets = parse_json_response(response, ["topic", "Problem Statement", "Proposed Methodology", "Experimental Validation"], fallback_type="facets")

        if any(variant_facets.values()):
            # Use extracted topic if available, otherwise fallback to generated format
            topic = variant_facets.get("topic", f"{idea.topic} - {direction} variant")

            variant_idea = ResearchIdea(
                topic=topic,
                facets=variant_facets,
                source="idea_generator",
                method="async_expansion_variant"
            )
            # Debug log the generated idea
            self._debug_log_idea(variant_idea, "async_expansion_variant")
            return variant_idea
        return None
