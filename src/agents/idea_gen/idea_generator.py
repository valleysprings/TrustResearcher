"""
Idea Generator

Central idea generation with three strategies:
- Base variants: cluster-based context + sampled direction
- GoT variants: reasoning path context + sampled direction
- Cross-pollination: hybrid ideas from existing ideas + sampled direction

Each idea uses: global_grounding + strategy_context + strategy_gaps + sampled_direction
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any
from .research_idea import ResearchIdea
from ...utils.async_utils import limit_async_func_call, retry_with_timeout
from ...utils.text_utils import safe_json_parse
from ...skills.ideagen.ideagen import (
    BASE_IDEA_GENERATION_SYSTEM_PROMPT,
    BASE_IDEA_GENERATION_USER_PROMPT,
    GOT_IDEA_GENERATION_SYSTEM_PROMPT,
    GOT_IDEA_GENERATION_USER_PROMPT,
    CROSS_IDEA_GENERATION_SYSTEM_PROMPT,
    CROSS_IDEA_GENERATION_USER_PROMPT,
)


class IdeaGenerator:
    """Central idea generator with multiple strategies."""

    def __init__(self, llm_interface, kg, planning_module, idea_generation_config: Dict = None, logger=None):
        self.llm = llm_interface
        self.kg = kg
        self.planning_module = planning_module
        self.idea_generation_config = idea_generation_config or {}
        self.idea_gen_config = self.idea_generation_config
        self.logger = logger

    # ==================== Base Variant ====================

    @retry_with_timeout()
    async def generate_base_variant(
        self, seed_topic: str, variant_num: int, global_grounding: Dict
    ) -> ResearchIdea:
        """Generate a base variant idea.

        Args:
            seed_topic: Research topic
            variant_num: Variant index (used to sample direction and community)
            global_grounding: Shared global grounding from plan
        """
        # Get community context from KG
        cluster_context = self.kg.get_community_context(variant_num)
        if not cluster_context:
            print(f"    Warning: Empty cluster_context for variant {variant_num} (KG has {len(self.kg.community_info)} communities)")
            cluster_context = f"No specific cluster context available. Generate ideas based on the seed topic: {seed_topic}"

        # Generate directions specific to this cluster
        directions_result = await self.planning_module.generate_base_directions(
            seed_topic, global_grounding, cluster_context
        )
        directions = directions_result['directions']
        cluster_gaps = directions_result['gaps']

        # Sample one direction for this idea
        sampled_direction = directions[variant_num % len(directions)] if directions else ""

        # Format grounding and gaps
        grounding_str = self._format_grounding(global_grounding)
        gaps_str = "\n".join(f"- {gap}" for gap in cluster_gaps) if cluster_gaps else ""

        context_vars = {
            'seed_topic': seed_topic,
            'global_grounding': grounding_str,
            'cluster_context': cluster_context,
            'cluster_gaps': gaps_str,
            'sampled_direction': sampled_direction,
        }

        response = await self.llm.generate_with_system_prompt(
            BASE_IDEA_GENERATION_SYSTEM_PROMPT,
            BASE_IDEA_GENERATION_USER_PROMPT.format(**context_vars),
            temperature=0.8,
            caller="idea_generator_base"
        )

        idea_data = safe_json_parse(response)
        if not idea_data or not isinstance(idea_data, dict):
            raise ValueError("Failed to parse base variant response")

        return ResearchIdea(
            topic=idea_data['topic'],
            facets={
                'Problem Statement': idea_data['problem_statement'],
                'Proposed Methodology': idea_data['proposed_methodology'],
                'Experimental Validation': idea_data['experimental_validation'],
            },
            source="idea_generator",
            method="base_variant"
        )

    # ==================== GoT Variant ====================

    @retry_with_timeout()
    async def generate_got_variant(
        self, seed_topic: str, variant_num: int,
        selected_paths: List[Dict], global_grounding: Dict
    ) -> ResearchIdea:
        """Generate a GoT-guided variant idea.

        Args:
            seed_topic: Research topic
            variant_num: Variant index
            selected_paths: List of sampled paths for this idea (10 paths)
            global_grounding: Shared global grounding from plan
        """
        # Build path context from selected paths
        path_context = self._format_paths_context(selected_paths)

        # Generate directions specific to this path context
        directions_result = await self.planning_module.generate_got_directions(
            seed_topic, global_grounding, path_context
        )
        directions = directions_result['directions']
        path_gaps = directions_result['gaps']

        # Sample one direction for this idea
        sampled_direction = directions[variant_num % len(directions)] if directions else ""

        # Format grounding and gaps
        grounding_str = self._format_grounding(global_grounding)
        gaps_str = "\n".join(f"- {gap}" for gap in path_gaps) if path_gaps else ""

        context_vars = {
            'seed_topic': seed_topic,
            'global_grounding': grounding_str,
            'path_context': path_context,
            'path_gaps': gaps_str,
            'sampled_direction': sampled_direction,
        }

        response = await self.llm.generate_with_system_prompt(
            GOT_IDEA_GENERATION_SYSTEM_PROMPT,
            GOT_IDEA_GENERATION_USER_PROMPT.format(**context_vars),
            temperature=0.8,
            caller="idea_generator_got"
        )

        idea_data = safe_json_parse(response)
        if not idea_data or not isinstance(idea_data, dict):
            raise ValueError("Failed to parse GoT variant response")

        idea = ResearchIdea(
            topic=idea_data['topic'],
            facets={
                'Problem Statement': idea_data['problem_statement'],
                'Proposed Methodology': idea_data['proposed_methodology'],
                'Experimental Validation': idea_data['experimental_validation'],
            },
            source="idea_generator",
            method="got_variant"
        )
        idea.reasoning_context = {"variant_num": variant_num, "num_paths": len(selected_paths)}
        return idea

    # ==================== Cross-Pollination ====================

    @limit_async_func_call(config_path='idea_generation.async_func_max_size')
    async def generate_cross_pollinated_ideas(
        self, seed_topic: str, base_ideas: List[ResearchIdea], global_grounding: Dict
    ) -> List[ResearchIdea]:
        """Generate hybrid ideas by cross-pollinating high-scoring ideas.

        Args:
            seed_topic: Research topic
            base_ideas: Ideas to cross-pollinate (should have critique scores)
            global_grounding: Shared global grounding from plan
        """
        if len(base_ideas) < self.idea_gen_config['min_ideas_for_cross_poll']:
            return []

        # Calculate target number of pairs to generate (with small buffer for failures)
        target_pairs = self.idea_gen_config['hybrid_ideas_final_limit'] * 2  # 2x buffer

        # Select high-scoring idea pairs using softmax sampling
        selected_pairs = self._select_idea_pairs_by_score(base_ideas, target_pairs)

        if not selected_pairs:
            if self.logger:
                self.logger.log_warning("No valid idea pairs selected for cross-pollination", "idea_generator")
            return []

        # Generate cross-pollinated ideas from selected pairs
        tasks = []
        for pair_idx, (idea1, idea2) in enumerate(selected_pairs):
            tasks.append(self._cross_pollinate_pair(
                seed_topic, idea1, idea2, pair_idx, global_grounding
            ))

        if self.logger:
            self.logger.log_info(
                f"Cross-pollination: creating {len(tasks)} pair tasks from {len(base_ideas)} ideas using score-based selection",
                "idea_generator"
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        cross_pollinated = [r for r in results if isinstance(r, ResearchIdea)]
        return cross_pollinated[:self.idea_gen_config['hybrid_ideas_final_limit']]

    async def _cross_pollinate_pair(
        self, seed_topic: str, idea1: ResearchIdea, idea2: ResearchIdea,
        pair_idx: int, global_grounding: Dict
    ) -> Optional[ResearchIdea]:
        """Cross-pollinate a pair of ideas."""
        # Build bridge context
        bridge_context = self._build_bridge_context(idea1, idea2)
        sampled_ideas = self._format_ideas_summary([idea1, idea2])

        # Generate directions specific to this bridge
        directions_result = await self.planning_module.generate_cross_directions(
            seed_topic, global_grounding, bridge_context, sampled_ideas
        )
        directions = directions_result['directions']
        bridge_gaps = directions_result['gaps']

        # Sample one direction for this pair
        sampled_direction = directions[pair_idx % len(directions)] if directions else ""

        # Format grounding and gaps
        grounding_str = self._format_grounding(global_grounding)
        gaps_str = "\n".join(f"- {gap}" for gap in bridge_gaps) if bridge_gaps else ""

        context_vars = {
            'seed_topic': seed_topic,
            'global_grounding': grounding_str,
            'bridge_context': bridge_context,
            'sampled_ideas': sampled_ideas,
            'bridge_gaps': gaps_str,
            'sampled_direction': sampled_direction,
        }

        try:
            response = await self.llm.generate_with_system_prompt(
                CROSS_IDEA_GENERATION_SYSTEM_PROMPT,
                CROSS_IDEA_GENERATION_USER_PROMPT.format(**context_vars),
                temperature=0.8,
                caller="idea_generator_cross"
            )

            idea_data = safe_json_parse(response)
            if not idea_data or not isinstance(idea_data, dict):
                return None

            return ResearchIdea(
                topic=idea_data['topic'],
                facets={
                    'Problem Statement': idea_data['problem_statement'],
                    'Proposed Methodology': idea_data['proposed_methodology'],
                    'Experimental Validation': idea_data['experimental_validation'],
                },
                source="idea_generator",
                method="cross_pollination"
            )
        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"Cross-pollination failed: {e}", "idea_generator")
            return None

    # ==================== Helper Methods ====================

    def _get_idea_score(self, idea: ResearchIdea) -> Optional[float]:
        """Extract overall_score from idea's critique feedback.

        Returns:
            Score if available, None if no valid critique found
        """
        for feedback in idea.review_feedback:
            if isinstance(feedback, dict) and feedback.get('type') == 'self_critique':
                score = feedback.get('overall_score')
                if score is not None:
                    return float(score)
        return None

    def _select_idea_pairs_by_score(
        self, ideas: List[ResearchIdea], num_pairs: int
    ) -> List[tuple]:
        """Select idea pairs using softmax sampling based on critique scores.

        Args:
            ideas: List of ideas with critique scores
            num_pairs: Number of pairs to select

        Returns:
            List of (idea1, idea2) tuples
        """
        # Filter ideas that have valid critique scores
        scored_ideas = []
        for idea in ideas:
            score = self._get_idea_score(idea)
            if score is not None:
                scored_ideas.append((idea, score))

        if len(scored_ideas) < 2:
            if self.logger:
                self.logger.log_warning(
                    f"Not enough ideas with critique scores for cross-pollination: {len(scored_ideas)}/2 required",
                    "idea_generator"
                )
            return []

        # Extract ideas and scores
        valid_ideas = [item[0] for item in scored_ideas]
        scores = np.array([item[1] for item in scored_ideas])

        # Apply softmax to get probabilities
        exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
        probabilities = exp_scores / exp_scores.sum()

        selected_pairs = []
        used_indices = set()

        for _ in range(num_pairs):
            # Get available indices (not yet used)
            available_indices = [i for i in range(len(valid_ideas)) if i not in used_indices]

            if len(available_indices) < 2:
                break  # Not enough ideas left

            # Get probabilities for available ideas
            available_probs = probabilities[available_indices]
            available_probs = available_probs / available_probs.sum()  # Renormalize

            # Sample two ideas without replacement
            selected_idx = np.random.choice(
                available_indices,
                size=min(2, len(available_indices)),
                replace=False,
                p=available_probs
            )

            if len(selected_idx) == 2:
                idea1, idea2 = valid_ideas[selected_idx[0]], valid_ideas[selected_idx[1]]
                selected_pairs.append((idea1, idea2))
                used_indices.update(selected_idx)

        return selected_pairs

    def _format_grounding(self, grounding: Dict) -> str:
        """Format global grounding dict as string for prompts."""
        parts = []
        if grounding.get("field_overview"):
            parts.append(f"FIELD OVERVIEW:\n{grounding['field_overview']}")
        if grounding.get("key_findings"):
            parts.append(f"KEY FINDINGS:\n{grounding['key_findings']}")
        if grounding.get("kg_insights"):
            parts.append(f"KG INSIGHTS:\n{grounding['kg_insights']}")
        if grounding.get("landscape_gaps"):
            parts.append(f"LANDSCAPE GAPS:\n{grounding['landscape_gaps']}")
        return "\n\n".join(parts) if parts else "No global grounding available."

    def _build_bridge_context(self, idea1: ResearchIdea, idea2: ResearchIdea) -> str:
        """Build context from two ideas for cross-pollination."""
        parts = [
            f"Idea 1: {idea1.topic}",
            f"  Problem: {idea1.facets.get('Problem Statement', '')[:200]}",
            f"  Method: {idea1.facets.get('Proposed Methodology', '')[:200]}",
            "",
            f"Idea 2: {idea2.topic}",
            f"  Problem: {idea2.facets.get('Problem Statement', '')[:200]}",
            f"  Method: {idea2.facets.get('Proposed Methodology', '')[:200]}",
        ]
        if self.kg:
            try:
                cross_conn = self.kg.get_cross_cluster_connections()
                if cross_conn:
                    parts.append("")
                    parts.append(f"Cross-domain bridges: {cross_conn[:5]}")
            except:
                pass
        return "\n".join(parts)

    def _format_paths_context(self, paths: List[Dict]) -> str:
        """Format multiple paths into context string for GoT variant."""
        if not paths:
            return "No reasoning paths available."

        parts = [f"REASONING PATHS ({len(paths)} sampled):"]
        for i, path in enumerate(paths, 1):
            path_str = path.get("path_string", "")
            score = path.get("score", 0.0)
            parts.append(f"\nPath {i} (score={score:.2f}):")
            parts.append(f"  {path_str}")

        return "\n".join(parts)

    def _format_ideas_summary(self, ideas: List[ResearchIdea]) -> str:
        """Format ideas for prompt context."""
        summaries = []
        for i, idea in enumerate(ideas, 1):
            summaries.append(f"{i}. {idea.topic}")
            if idea.facets.get('Problem Statement'):
                summaries.append(f"   Problem: {idea.facets['Problem Statement'][:150]}")
        return "\n".join(summaries)
