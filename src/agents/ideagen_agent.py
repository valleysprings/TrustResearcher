"""
IdeaGen Agent

Orchestrates research idea generation using specialized tools.
Clean orchestrator that coordinates planning, variant generation, critique, and refinement.
"""

import asyncio
import math
from typing import List, Dict, Optional
from ..knowledge_graph.kg_ops import KGOps
from .idea_gen.planning_module import PlanningModule
from .idea_gen.research_idea import ResearchIdea
from .reviewer_agent import IdeaCritique
from .idea_gen.idea_refinement import IdeaRefinement
from .idea_gen.idea_generator import IdeaGenerator
from ..utils.llm_interface import LLMInterface
from ..utils.token_cost_tracker import get_current_token_count
from ..skills.ideagen.planning import (
    STRATEGIC_DIRECTIONS_BASE_SYSTEM_PROMPT, STRATEGIC_DIRECTIONS_BASE_USER_PROMPT,
    STRATEGIC_DIRECTIONS_GOT_SYSTEM_PROMPT, STRATEGIC_DIRECTIONS_GOT_USER_PROMPT,
    STRATEGIC_DIRECTIONS_CROSS_SYSTEM_PROMPT, STRATEGIC_DIRECTIONS_CROSS_USER_PROMPT,
)


class IdeaGenAgent:
    """Orchestrator for research idea generation using specialized tools."""

    def __init__(self, kg: KGOps, idea_generation_config: Dict = None, planning_config: Dict = None, logger=None, llm_config: Dict = None):
        self.kg = kg
        self.idea_generation_config = idea_generation_config or {}
        self.planning_config = planning_config or {}
        self.logger = logger
        self.llm = LLMInterface(config=llm_config, logger=logger)

        # Initialize reasoning modules
        self.planning_module = PlanningModule(kg, self.llm, planning_config=self.planning_config)

        # Initialize specialized tools
        self.critique_tool = IdeaCritique(self.llm, config=self.idea_generation_config.get('reviewer', {}), logger=logger)
        self.refinement_tool = IdeaRefinement(self.llm, idea_generation_config=self.idea_generation_config, logger=logger)
        self.idea_generator = IdeaGenerator(self.llm, kg, self.planning_module, idea_generation_config=self.idea_generation_config, logger=logger)

        # State
        self.generated_ideas = []
        self.current_exploration_state = {}
        self.plan = {}
        self.literature_context = []
        self.literature_summary = ""

    async def generate_ideas(self, seed_topic: str, num_ideas: int = 5, literature_context: List = None) -> List[ResearchIdea]:
        """
        Main entry point for idea generation.

        Args:
            seed_topic: Research topic to generate ideas for
            num_ideas: Number of ideas to generate
            literature_context: Optional literature context from semantic scholar

        Returns:
            List of generated research ideas
        """
        self.literature_context = literature_context or []
        self.generated_ideas = []

        print(f"\n{'='*80}")
        print(f"Generating {num_ideas} research ideas for: {seed_topic}")
        print(f"{'='*80}\n")

        # Creating research plan
        print("Creating research plan...")
        self.plan = await self.planning_module.create_plan(seed_topic, self.literature_context)

        # Build Knowledge Graph
        print("\nBuilding knowledge graph...")
        await self.kg.build(seed_topic, self.literature_context)

        # Sample paths from KG
        print("\nSampling paths from knowledge graph...")
        idea_gen_config = self.idea_generation_config
        overgeneration_factor = idea_gen_config['overgeneration_factor']
        got_variant_ratio = idea_gen_config['got_variant_ratio']
        got_count = math.ceil(num_ideas * overgeneration_factor * got_variant_ratio)
        self.kg.sample_paths(got_count=got_count)

        # Generate Ideas
        print("\nGenerating idea variants...")
        await self._generate_ideas_internal(seed_topic, num_ideas)

        # Critique and Refinement
        if self.idea_generation_config['enable_self_critique']:
            print("\nCritiquing and refining ideas...")
            await self._critique_and_refine()

        # Validation Elaboration
        if self.idea_generation_config['enable_elaboration']:
            print("\nElaborating validation plans...")
            await self._elaborate_validation()

        print(f"\n{'='*80}")
        print(f"Generated {len(self.generated_ideas)} ideas")
        print(f"{'='*80}\n")

        # Save all generated ideas to logs/idea/ directory via logger
        if self.logger:
            overgenerate_factor = self.idea_generation_config['overgeneration_factor']
            self.logger.save_ideas(self.generated_ideas, seed_topic, num_ideas, overgenerate_factor)

        return self.generated_ideas

    async def _generate_ideas_internal(self, seed_topic: str, num_ideas: int):
        """Generate idea variants using base and GoT generators."""
        # Calculate variant distribution based on overgeneration
        idea_gen_config = self.idea_generation_config

        # Dynamic calculation: num_ideas * overgeneration_factor * ratio
        overgen = idea_gen_config['overgeneration_factor']
        base_count = max(1, math.ceil(num_ideas * overgen * idea_gen_config['base_variant_ratio']))
        got_count = max(1, math.ceil(num_ideas * overgen * idea_gen_config['got_variant_ratio']))
        cross_count = max(1, math.ceil(num_ideas * overgen * idea_gen_config['pollination_ratio']))

        total_target = base_count + got_count + cross_count
        print(f"  Target: {total_target} ideas ({base_count} base + {got_count} GoT + {cross_count} cross-poll)")

        # Extract global grounding from plan (shared across all ideas)
        global_grounding = self.plan['global_grounding']

        # Generate base variants
        if base_count > 0:
            print(f"  Generating {base_count} base variants...")
            tokens_before = get_current_token_count()
            base_tasks = [
                self.idea_generator.generate_base_variant(seed_topic, i, global_grounding)
                for i in range(base_count)
            ]
            base_results = await asyncio.gather(*base_tasks, return_exceptions=True)
            base_success = 0
            for result in base_results:
                if isinstance(result, ResearchIdea):
                    self.generated_ideas.append(result)
                    base_success += 1
            tokens_used = get_current_token_count() - tokens_before
            print(f"    Base: {base_success}/{base_count} succeeded | Tokens: {tokens_used:,}")

        # Generate GoT variants (with softmax path sampling)
        if got_count > 0:
            print(f"  Generating {got_count} GoT variants...")
            tokens_before = get_current_token_count()
            # Softmax sample paths for GoT variants (returns List[List[Dict]])
            sampled_paths = self.kg.sample_paths_for_got(got_count)
            if sampled_paths:
                got_tasks = [
                    self.idea_generator.generate_got_variant(
                        seed_topic, i,
                        selected_paths=sampled_paths[i],
                        global_grounding=global_grounding
                    )
                    for i in range(min(got_count, len(sampled_paths)))
                ]
                got_results = await asyncio.gather(*got_tasks, return_exceptions=True)
                got_success = 0
                for result in got_results:
                    if isinstance(result, ResearchIdea):
                        self.generated_ideas.append(result)
                        got_success += 1
                tokens_used = get_current_token_count() - tokens_before
                print(f"    GoT: {got_success}/{got_count} succeeded | Tokens: {tokens_used:,}")
            else:
                print(f"    GoT: 0/{got_count} (no paths sampled from KG)")

        # Critique base+GoT ideas before cross-pollination (so they have scores)
        if self.idea_generation_config['enable_self_critique'] and len(self.generated_ideas) >= 2:
            print(f"\n  Critiquing base+GoT ideas before cross-pollination...")
            await self._critique_ideas_batch(self.generated_ideas, "base+GoT ideas")

        # Generate cross-pollination variants
        if cross_count > 0 and len(self.generated_ideas) >= 2:
            print(f"  Generating {cross_count} cross-pollination variants...")
            tokens_before = get_current_token_count()
            cross_results = await self.idea_generator.generate_cross_pollinated_ideas(
                seed_topic,
                self.generated_ideas,
                global_grounding
            )
            added = cross_results[:cross_count]
            self.generated_ideas.extend(added)
            tokens_used = get_current_token_count() - tokens_before
            print(f"    Cross-poll: {len(added)}/{cross_count} succeeded | Tokens: {tokens_used:,}")

            # Critique cross-pollinated ideas
            if self.idea_generation_config['enable_self_critique'] and added:
                print(f"\n  Critiquing cross-pollinated ideas...")
                await self._critique_ideas_batch(added, "cross-poll ideas")
        elif cross_count > 0:
            print(f"    Cross-poll: skipped (need >= 2 base ideas)")

    async def _critique_ideas_batch(self, ideas: List[ResearchIdea], batch_label: str = "ideas"):
        """Critique and refine a specific batch of ideas.

        Args:
            ideas: List of ideas to critique
            batch_label: Label for logging (e.g., "base+GoT", "cross-poll")
        """
        if not ideas:
            return

        total = len(ideas)
        print(f"  Critiquing {total} {batch_label}...")

        # Critique ideas
        tokens_before = get_current_token_count()
        critique_tasks = [self.critique_tool.critique_idea(idea) for idea in ideas]
        critiques = await asyncio.gather(*critique_tasks, return_exceptions=True)
        critique_tokens = get_current_token_count() - tokens_before
        print(f"    Critiques: {sum(1 for c in critiques if isinstance(c, dict))}/{total} succeeded | Tokens: {critique_tokens:,}")

        # Apply refinements
        refinement_tasks = []
        for idea, critique in zip(ideas, critiques):
            if isinstance(critique, dict):
                idea.add_feedback({"type": "self_critique", **critique})
                refinement_tasks.append(self.refinement_tool.refine_idea(idea, critique))

        if refinement_tasks:
            print(f"    Refining {len(refinement_tasks)} {batch_label}...")
            tokens_before = get_current_token_count()
            refined_facets_list = await asyncio.gather(*refinement_tasks, return_exceptions=True)
            refine_tokens = get_current_token_count() - tokens_before
            print(f"    Refinements: {sum(1 for r in refined_facets_list if isinstance(r, dict))}/{len(refinement_tasks)} succeeded | Tokens: {refine_tokens:,}")

            # Update ideas with refined facets
            for idea, refined_facets in zip(ideas, refined_facets_list):
                if isinstance(refined_facets, dict) and any(refined_facets.values()):
                    idea.refine(refined_facets, reasoning="Applied self-critique suggestions")

    async def _critique_and_refine(self):
        """Critique and refine all generated ideas (safety net for any uncritiqued ideas)."""
        if not self.generated_ideas:
            return

        # Filter out ideas that already have critique scores
        uncritiqued_ideas = []
        for idea in self.generated_ideas:
            has_critique = any(
                isinstance(fb, dict) and fb.get('type') == 'self_critique'
                for fb in idea.review_feedback
            )
            if not has_critique:
                uncritiqued_ideas.append(idea)

        if uncritiqued_ideas:
            print(f"\n  Critiquing {len(uncritiqued_ideas)} remaining uncritiqued ideas...")
            await self._critique_ideas_batch(uncritiqued_ideas, "remaining ideas")

    async def _elaborate_validation(self):
        """Elaborate validation facets for all ideas."""
        if not self.generated_ideas:
            return

        elaboration_config = self.idea_generation_config['methodology_elaboration']
        await self.refinement_tool.elaborate_ideas(self.generated_ideas, elaboration_config)

    async def _generate_strategic_directions_for_variant(self, variant_type: str, context_vars: Dict) -> List[str]:
        """Generate strategic directions for variant generation."""
        if variant_type == 'base':
            system_prompt = STRATEGIC_DIRECTIONS_BASE_SYSTEM_PROMPT
            user_prompt = STRATEGIC_DIRECTIONS_BASE_USER_PROMPT.format(**context_vars)
        elif variant_type == 'got':
            system_prompt = STRATEGIC_DIRECTIONS_GOT_SYSTEM_PROMPT
            user_prompt = STRATEGIC_DIRECTIONS_GOT_USER_PROMPT.format(**context_vars)
        elif variant_type == 'cross':
            system_prompt = STRATEGIC_DIRECTIONS_CROSS_SYSTEM_PROMPT
            user_prompt = STRATEGIC_DIRECTIONS_CROSS_USER_PROMPT.format(**context_vars)
        else:
            return []

        response = await self.llm.generate_with_system_prompt(
            system_prompt, user_prompt,
            max_tokens=self.idea_generation_config['strategic_directions_max_tokens'],
            caller=f"strategic_directions_{variant_type}"
        )

        # Parse strategic directions from response
        directions = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('•'):
                directions.append(line[1:].strip())

        return directions
