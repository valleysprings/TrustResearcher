"""
Reviewer Agent

Two-stage comprehensive peer review:
- Stage 1: Quick preliminary critique (IdeaCritique)
- Stage 2: Systematic per-criterion detailed evaluation

Replaces and consolidates: reviewer_agent.py, novelty_agent.py, aggregator.py
"""

import re
from typing import Dict, List, Optional
from .idea_gen.research_idea import ResearchIdea
from ..utils.llm_interface import LLMInterface
from ..utils.async_utils import limit_async_func_call, retry_with_timeout
from ..utils.text_utils import safe_json_parse
from ..skills.reviewer import (
    STAGE1_CRITIQUE_SYSTEM_PROMPT, STAGE1_CRITIQUE_USER_PROMPT
)
from ..skills.reviewer import (
    STAGE2_NOVELTY_SYSTEM_PROMPT, STAGE2_NOVELTY_USER_PROMPT,
    STAGE2_FEASIBILITY_SYSTEM_PROMPT, STAGE2_FEASIBILITY_USER_PROMPT,
    STAGE2_CLARITY_SYSTEM_PROMPT, STAGE2_CLARITY_USER_PROMPT,
    STAGE2_IMPACT_SYSTEM_PROMPT, STAGE2_IMPACT_USER_PROMPT
)


class IdeaCritique:
    """Stage 1: Quick preliminary critique on research ideas."""

    def __init__(self, llm_interface, config: Dict = None, logger=None):
        self.llm = llm_interface
        self.reviewer_config = config or {}
        self.logger = logger

    @limit_async_func_call(config_path='idea_generation.async_func_max_size')
    @retry_with_timeout()
    async def critique_idea(self, idea) -> Dict:
        """Perform preliminary critique on a generated idea."""
        user_prompt = STAGE1_CRITIQUE_USER_PROMPT.format(idea=idea)
        response = await self.llm.generate_with_system_prompt(
            STAGE1_CRITIQUE_SYSTEM_PROMPT,
            user_prompt,
            caller="idea_critique",
            task_type="idea_generation"
        )
        return self._parse_critique_response(response)

    def _parse_critique_response(self, response: str) -> Dict:
        """Parse critique response into structured data."""
        stage1_config = self.reviewer_config['stage1_critique']
        default_facet = stage1_config['default_facet_score']
        default_overall = stage1_config['default_overall_score']

        critique = {
            "overall_score": default_overall,
            "novelty_score": default_facet,
            "feasibility_score": default_facet,
            "clarity_score": default_facet,
            "impact_score": default_facet,
            "needs_refinement": True,
            "strengths": [],
            "weaknesses": [],
            "suggestions": []
        }

        json_result = safe_json_parse(response)
        if json_result and isinstance(json_result, dict):
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

            if "needs_refinement" in json_result:
                val = json_result.get("needs_refinement")
                if isinstance(val, str):
                    critique["needs_refinement"] = val.strip().lower() in {"yes", "true", "y"}
                elif isinstance(val, bool):
                    critique["needs_refinement"] = val

            critique["strengths"] = json_result.get("strengths", [])
            critique["weaknesses"] = json_result.get("weaknesses", [])
            critique["suggestions"] = json_result.get("suggestions", [])
            return self._finalize_scores(critique, default_overall)

        return self._parse_critique_text(response, critique, default_overall)

    def _parse_critique_text(self, response: str, critique: Dict, default_overall: float) -> Dict:
        """Fallback text parsing for critique response."""
        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            for key in ["novelty", "feasibility", "clarity", "impact"]:
                if key in line.lower():
                    m = re.search(r'(\d+(?:\.\d+)?)', line)
                    if m:
                        critique[f"{key}_score"] = float(m.group(1))

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

        return self._finalize_scores(critique, default_overall)

    def _finalize_scores(self, critique: Dict, default_overall: float) -> Dict:
        """Calculate overall score from individual scores."""
        facet_keys = ["novelty_score", "feasibility_score", "clarity_score", "impact_score"]
        facet_values = []

        for key in facet_keys:
            value = critique.get(key)
            if isinstance(value, (int, float)):
                facet_values.append(float(value))

        if facet_values:
            critique['overall_score'] = round(sum(facet_values) / len(facet_values), 2)
        else:
            critique['overall_score'] = float(default_overall)

        return critique


class ReviewerAgent:
    """Two-stage reviewer: preliminary critique + systematic per-criterion evaluation."""

    def __init__(self, reviewer_config: Dict = None, logger=None, llm_config: Dict = None):
        self.reviewer_config = reviewer_config or {}
        self.logger = logger
        self.llm = LLMInterface(config=llm_config, logger=logger)
        self.critique_tool = IdeaCritique(self.llm, config=self.reviewer_config, logger=logger)

        # Review criteria with weights (from config)
        stage2_config = self.reviewer_config['stage2_detailed']
        self.review_criteria = {}
        for criterion, criterion_config in stage2_config['criteria'].items():
            self.review_criteria[criterion] = {
                'weight': criterion_config['weight'],
                'enabled': criterion_config['enabled']
            }

        self.feedback_history = []

    async def review(self, idea: ResearchIdea, enable_stage1: bool = True, enable_stage2: bool = True) -> Dict:
        """
        Two-stage review: preliminary critique + detailed per-criterion evaluation.

        Args:
            idea: Research idea to review
            enable_stage1: Enable preliminary critique (fast)
            enable_stage2: Enable detailed per-criterion review (thorough)

        Returns:
            Comprehensive review with both stages
        """
        print(f"Reviewing: {idea.topic}")

        review_result = {
            'idea_topic': idea.topic,
            'stage1_preliminary': None,
            'stage2_detailed': None,
            'overall_assessment': {}
        }

        # Stage 1: Preliminary Critique (Fast)
        if enable_stage1:
            print("  Stage 1: Preliminary critique...")
            review_result['stage1_preliminary'] = await self.critique_tool.critique_idea(idea)

        # Stage 2: Systematic Per-Criterion Review (Detailed)
        if enable_stage2:
            print("  Stage 2: Detailed per-criterion review...")
            review_result['stage2_detailed'] = await self._systematic_review(
                idea,
                review_result['stage1_preliminary']
            )

        # Aggregate assessment
        review_result['overall_assessment'] = self._aggregate_assessment(
            review_result['stage1_preliminary'],
            review_result['stage2_detailed']
        )

        # Store and attach feedback - include both stages
        self.feedback_history.append(review_result)
        feedback_data = {
            'type': 'two_stage_review',
            'stage1_preliminary': review_result['stage1_preliminary'],
            'stage2_detailed': review_result['stage2_detailed'],
            **review_result['overall_assessment']
        }
        idea.add_feedback(feedback_data)

        return review_result

    async def _systematic_review(self, idea: ResearchIdea, stage1_result: Optional[Dict]) -> Dict:
        """Systematic per-criterion detailed evaluation."""
        detailed_review = {'criterion_reviews': {}, 'overall_detailed_score': None}

        # Evaluate each enabled criterion
        for criterion, config in self.review_criteria.items():
            if config.get('enabled', True):
                print(f"    Evaluating {criterion}...")
                detailed_review['criterion_reviews'][criterion] = await self._evaluate_criterion(
                    idea, criterion, stage1_result
                )

        # Calculate weighted overall score (only if all scores are present)
        total_weight = sum(c['weight'] for c in self.review_criteria.values() if c.get('enabled', True))

        # Collect valid scores
        valid_scores = []
        for c in detailed_review['criterion_reviews']:
            score = detailed_review['criterion_reviews'][c].get('score')
            if score is not None:
                valid_scores.append((score, self.review_criteria[c]['weight']))

        # Calculate weighted average only if we have all scores
        if len(valid_scores) == len(detailed_review['criterion_reviews']) and total_weight > 0:
            weighted_sum = sum(score * weight for score, weight in valid_scores)
            detailed_review['overall_detailed_score'] = round(weighted_sum / total_weight, 2)
        else:
            detailed_review['overall_detailed_score'] = None

        return detailed_review

    async def _evaluate_criterion(self, idea: ResearchIdea, criterion: str, stage1_result: Optional[Dict]) -> Dict:
        """Evaluate a single criterion in depth."""
        # Build context from Stage 1
        stage1_context = ""
        if stage1_result:
            stage1_score = stage1_result.get(f'{criterion}_score', 'N/A')
            stage1_context = f"\nPreliminary {criterion} score: {stage1_score}"

        # Route to specialized evaluation
        if criterion == 'novelty':
            return await self._evaluate_novelty(idea, stage1_context)
        elif criterion == 'feasibility':
            return await self._evaluate_feasibility(idea, stage1_context)
        elif criterion == 'clarity':
            return await self._evaluate_clarity(idea, stage1_context)
        elif criterion == 'impact':
            return await self._evaluate_impact(idea, stage1_context)
        else:
            return {'criterion': criterion, 'score': None, 'strengths': [], 'weaknesses': [], 'suggestions': []}

    async def _evaluate_novelty(self, idea: ResearchIdea, stage1_context: str) -> Dict:
        """Evaluate novelty with multi-dimensional assessment (integrates former NoveltyAgent)."""
        user_prompt = STAGE2_NOVELTY_USER_PROMPT.format(idea=idea, stage1_context=stage1_context)
        response = await self.llm.generate_with_system_prompt(
            STAGE2_NOVELTY_SYSTEM_PROMPT, user_prompt,
            caller="reviewer_novelty"
        )
        return self._parse_criterion_response(response, 'novelty')

    async def _evaluate_feasibility(self, idea: ResearchIdea, stage1_context: str) -> Dict:
        """Evaluate technical and practical feasibility."""
        user_prompt = STAGE2_FEASIBILITY_USER_PROMPT.format(idea=idea, stage1_context=stage1_context)
        response = await self.llm.generate_with_system_prompt(
            STAGE2_FEASIBILITY_SYSTEM_PROMPT, user_prompt,
            caller="reviewer_feasibility"
        )
        return self._parse_criterion_response(response, 'feasibility')

    async def _evaluate_clarity(self, idea: ResearchIdea, stage1_context: str) -> Dict:
        """Evaluate clarity of problem definition and methodology."""
        user_prompt = STAGE2_CLARITY_USER_PROMPT.format(idea=idea, stage1_context=stage1_context)
        response = await self.llm.generate_with_system_prompt(
            STAGE2_CLARITY_SYSTEM_PROMPT, user_prompt,
            caller="reviewer_clarity"
        )
        return self._parse_criterion_response(response, 'clarity')

    async def _evaluate_impact(self, idea: ResearchIdea, stage1_context: str) -> Dict:
        """Evaluate potential impact and significance."""
        user_prompt = STAGE2_IMPACT_USER_PROMPT.format(idea=idea, stage1_context=stage1_context)
        response = await self.llm.generate_with_system_prompt(
            STAGE2_IMPACT_SYSTEM_PROMPT,
            user_prompt,
            caller="reviewer_impact"
        )
        return self._parse_criterion_response(response, 'impact')

    def _map_criterion_fields(self, json_data: Dict, criterion: str) -> tuple:
        """Map criterion-specific JSON fields to strengths/weaknesses/suggestions."""
        strengths = []
        weaknesses = []
        suggestions = []

        if criterion == 'novelty':
            # Strengths: novel aspects, differentiation
            strengths.extend(json_data.get('novel_aspects', []))
            if json_data.get('differentiation'):
                strengths.append(f"Differentiation: {json_data['differentiation']}")
            if json_data.get('incremental_vs_breakthrough'):
                strengths.append(f"Innovation type: {json_data['incremental_vs_breakthrough']}")

            # Weaknesses: overlaps with existing work
            weaknesses.extend(json_data.get('existing_work_overlap', []))

            # Suggestions: recommendations
            suggestions.extend(json_data.get('recommendations', []))

        elif criterion == 'feasibility':
            # Strengths: technical strengths
            strengths.extend(json_data.get('technical_strengths', []))
            if json_data.get('timeline_assessment'):
                strengths.append(f"Timeline: {json_data['timeline_assessment']}")

            # Weaknesses: challenges, risks, resource requirements
            weaknesses.extend(json_data.get('technical_challenges', []))
            weaknesses.extend(json_data.get('risk_factors', []))
            if json_data.get('resource_requirements'):
                for req in json_data['resource_requirements']:
                    weaknesses.append(f"Resource need: {req}")

            # Suggestions: mitigation strategies
            suggestions.extend(json_data.get('mitigation_strategies', []))

        elif criterion == 'clarity':
            # Strengths: well-articulated aspects
            strengths.extend(json_data.get('well_articulated_aspects', []))
            if json_data.get('logical_flow'):
                strengths.append(f"Logical flow: {json_data['logical_flow']}")
            if json_data.get('completeness'):
                strengths.append(f"Completeness: {json_data['completeness']}")

            # Weaknesses: ambiguous aspects
            weaknesses.extend(json_data.get('ambiguous_aspects', []))

            # Suggestions: clarification needs
            suggestions.extend(json_data.get('clarification_needs', []))

        elif criterion == 'impact':
            # Strengths: contributions, applications, advancement
            strengths.extend(json_data.get('scientific_contributions', []))
            strengths.extend(json_data.get('practical_applications', []))
            if json_data.get('field_advancement'):
                strengths.append(f"Field advancement: {json_data['field_advancement']}")
            if json_data.get('broader_implications'):
                strengths.append(f"Broader implications: {json_data['broader_implications']}")

            # Weaknesses: limitations
            weaknesses.extend(json_data.get('limitations', []))

            # Suggestions: enhancement suggestions
            suggestions.extend(json_data.get('enhancement_suggestions', []))

        return strengths, weaknesses, suggestions

    def _parse_criterion_response(self, response: str, criterion: str) -> Dict:
        """Parse criterion evaluation response into structured format with flexible field mapping."""
        result = {
            'criterion': criterion,
            'score': None,
            'strengths': [],
            'weaknesses': [],
            'suggestions': [],
            'raw_data': {}
        }

        json_data = safe_json_parse(response)
        if json_data and isinstance(json_data, dict):
            # Extract score (required)
            result['score'] = float(json_data.get('score')) if json_data.get('score') is not None else None

            # Store raw data for reference
            result['raw_data'] = json_data

            # Intelligently map criterion-specific fields to strengths/weaknesses/suggestions
            result['strengths'], result['weaknesses'], result['suggestions'] = \
                self._map_criterion_fields(json_data, criterion)

        return result

    def _aggregate_assessment(self, stage1_result: Optional[Dict], stage2_result: Optional[Dict]) -> Dict:
        """Aggregate Stage 1 and Stage 2 into overall assessment using weighted combination."""
        assessment = {
            'overall_score': None,
            'key_strengths': [],
            'key_weaknesses': [],
            'actionable_suggestions': []
        }

        # Get aggregation weights from config
        agg_config = self.reviewer_config['aggregation']
        stage1_weight = agg_config['stage1_weight']
        stage2_weight = agg_config['stage2_weight']

        # Calculate weighted overall score
        stage1_score = stage1_result.get('overall_score') if stage1_result else None
        stage2_score = stage2_result.get('overall_detailed_score') if stage2_result else None

        if stage1_score is not None and stage2_score is not None:
            # Both stages available: weighted combination
            assessment['overall_score'] = round(stage1_score * stage1_weight + stage2_score * stage2_weight, 2)
        elif stage2_score is not None:
            # Only Stage 2 available: use it directly
            assessment['overall_score'] = stage2_score
        elif stage1_score is not None:
            # Only Stage 1 available: use it directly
            assessment['overall_score'] = stage1_score

        # Collect qualitative feedback (prioritize Stage 2)
        max_items = self.reviewer_config['aggregation']['max_items_per_criterion']
        if stage2_result:
            for _, result in stage2_result.get('criterion_reviews', {}).items():
                assessment['key_strengths'].extend(result.get('strengths', [])[:max_items])
                assessment['key_weaknesses'].extend(result.get('weaknesses', [])[:max_items])
                assessment['actionable_suggestions'].extend(result.get('suggestions', [])[:max_items])
        elif stage1_result:
            assessment['key_strengths'] = stage1_result.get('strengths', [])
            assessment['key_weaknesses'] = stage1_result.get('weaknesses', [])
            assessment['actionable_suggestions'] = stage1_result.get('suggestions', [])

        return assessment
