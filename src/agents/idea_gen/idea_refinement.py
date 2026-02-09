"""
Idea Refinement

Refines and elaborates research ideas:
- refine_idea: Apply critique feedback to improve idea facets
- elaborate_ideas: Expand experimental validation with implementation details
"""

import ast
import asyncio
from typing import Dict, List, Optional
from ...utils.async_utils import limit_async_func_call, retry_with_timeout
from ...utils.text_utils import parse_json_response
from ...skills.ideagen.ideagen import (
    REFINEMENT_SYSTEM_PROMPT,
    REFINEMENT_USER_PROMPT,
    VALIDATION_ELABORATION_SYSTEM_PROMPT,
    VALIDATION_ELABORATION_USER_PROMPT,
)


class IdeaRefinement:
    """Tool for refining and elaborating research ideas."""

    def __init__(self, llm_interface, idea_generation_config: Dict = None, logger=None):
        self.llm = llm_interface
        self.idea_generation_config = idea_generation_config or {}
        self.idea_gen_config = self.idea_generation_config
        self.logger = logger

    @limit_async_func_call(config_path='idea_generation.async_func_max_size')
    @retry_with_timeout()
    async def refine_idea(self, idea, critique: Dict) -> Dict[str, str]:
        """Apply critique feedback to improve idea facets."""
        default_score = self.idea_gen_config['default_facet_score']
        suggestions_text = "\n".join(critique.get('suggestions', [])) if critique.get('suggestions') else ""

        user_prompt = REFINEMENT_USER_PROMPT.format(
            idea=str(idea),
            suggestions=suggestions_text,
            novelty_score=critique.get('novelty_score', default_score),
            feasibility_score=critique.get('feasibility_score', default_score),
            clarity_score=critique.get('clarity_score', default_score),
            impact_score=critique.get('impact_score', default_score),
        )

        response = await self.llm.generate_with_system_prompt(
            REFINEMENT_SYSTEM_PROMPT,
            user_prompt,
            max_tokens=None,
            caller="idea_refinement",
            task_type="idea_generation"
        )

        return parse_json_response(
            response,
            ["topic", "Problem Statement", "Proposed Methodology", "Experimental Validation"],
            fallback_type="facets"
        )

    async def elaborate_ideas(self, ideas: List, elaboration_config: Dict):
        """Elaborate validation facets for a list of ideas."""
        if not ideas:
            return

        validation_components = max(2, int(elaboration_config.get('validation_components', 4)))
        batch_size = max(1, int(elaboration_config.get('async_batch_size', 3)))

        for i in range(0, len(ideas), batch_size):
            batch = ideas[i:i + batch_size]
            tasks = [
                self._elaborate_validation(idea, validation_components)
                for idea in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for idea, result in zip(batch, results):
                if isinstance(result, dict) and result.get('Experimental Validation'):
                    updated_facets = idea.facets.copy()
                    updated_facets['Experimental Validation'] = self._normalize_output(result['Experimental Validation'])
                    idea.refine(updated_facets, reasoning="Experimental validation elaboration")
                    idea.reasoning_context.setdefault('validation_elaboration', result)
                elif isinstance(result, Exception) and self.logger:
                    self.logger.log_warning(f"Validation elaboration failed: {result}", "idea_refinement")

    @retry_with_timeout()
    async def _elaborate_validation(self, idea, validation_components: int = 4) -> Dict[str, str]:
        """Expand experimental validation facet with implementation details."""
        suggestions = self._collect_suggestions(idea)
        suggestions_text = "\n".join(f"- {s}" for s in suggestions if s)
        if not suggestions_text:
            suggestions_text = "(no additional critique suggestions supplied)"

        user_prompt = VALIDATION_ELABORATION_USER_PROMPT.format(
            topic=idea.topic,
            problem=idea.facets.get('Problem Statement', ''),
            methodology=idea.facets.get('Proposed Methodology', ''),
            validation=idea.facets.get('Experimental Validation', ''),
            validation_components=validation_components,
            critique_suggestions=suggestions_text
        )

        response = await self.llm.generate_with_system_prompt(
            VALIDATION_ELABORATION_SYSTEM_PROMPT,
            user_prompt,
            caller="idea_refinement",
            task_type="idea_generation"
        )

        return parse_json_response(response, ["Experimental Validation"], fallback_type="facets")

    def _collect_suggestions(self, idea) -> List[str]:
        """Retrieve self-critique suggestions for an idea."""
        if not hasattr(idea, 'review_feedback'):
            return []
        for feedback in reversed(idea.review_feedback):
            if isinstance(feedback, dict) and feedback.get('type') == 'self_critique':
                suggestions = feedback.get('suggestions') or []
                return [s for s in suggestions if isinstance(s, str)]
        return []

    def _normalize_output(self, validation) -> str:
        """Normalize validation output to well-formed string."""
        if isinstance(validation, str):
            stripped = validation.strip()
            if stripped.startswith(("[", "{")):
                try:
                    parsed = ast.literal_eval(stripped)
                    return self._normalize_output(parsed)
                except (ValueError, SyntaxError):
                    return stripped
            return stripped

        if isinstance(validation, list):
            cleaned = []
            for item in validation:
                if isinstance(item, str):
                    cleaned.append(item.strip())
                elif isinstance(item, dict):
                    cleaned.append("; ".join(f"{k}: {v}" for k, v in item.items()))
                else:
                    cleaned.append(str(item))
            return ' '.join(cleaned)

        if isinstance(validation, dict):
            return "; ".join(f"{k}: {v}" for k, v in validation.items())

        return str(validation)
