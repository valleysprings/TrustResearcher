"""
Internal Selector Agent

This agent provides multi-stage idea filtering and selection functionality.
It performs fast, lightweight evaluation to rank and select the most promising 
ideas from a large pool of generated ideas before external literature comparison.
"""

import asyncio
from typing import List, Dict, Tuple
import numpy as np
from .base_agent import BaseAgent
from .idea_generator import ResearchIdea
from ..utils.llm_interface import LLMInterface
from ..utils.async_utils import limit_async_func_call, retry_with_timeout, async_batch_processor
from ..utils.text_utils import (
    calculate_idea_similarity, find_most_similar_idea, extract_content_from_idea,
    safe_json_parse, parse_json_response
)
from ..prompts.selection.idea_selector_prompts import (
    QUICK_EVALUATION_SYSTEM_PROMPT, QUICK_EVALUATION_USER_PROMPT,
    COMPARATIVE_RANKING_SYSTEM_PROMPT, COMPARATIVE_RANKING_USER_PROMPT,
    IDEA_MERGING_SYSTEM_PROMPT, IDEA_MERGING_USER_PROMPT
)
import json
import re


class InternalSelector(BaseAgent):
    """
    Agent for internal idea filtering and selection from large pools of generated ideas.
    Performs lightweight evaluation and ranking to identify the most promising ideas.
    """
    
    def __init__(self, config: Dict = None, logger=None, llm_config: Dict = None):
        super().__init__("InternalSelector")
        self.config = config or {}
        self.logger = logger

        # Initialize LLM interface only if proper config is provided
        if llm_config and llm_config.get("api_key") and llm_config.get("model_name") and llm_config.get("base_url"):
            self.llm = LLMInterface(config=llm_config, logger=logger)
        else:
            self.llm = None
            if logger:
                logger.log_warning("LLM interface not initialized - missing configuration", "internal_selector")
        
        # Selection criteria weights - configurable
        default_criteria = {
            'novelty': 0.30,
            'feasibility': 0.25,
            'clarity': 0.20,
            'impact': 0.25
        }
        
        config_criteria = self.config.get('selection_criteria', {})
        if config_criteria:
            # Extract weights from config structure (config may have weight/description structure)
            self.selection_criteria = {}
            for criterion, value in config_criteria.items():
                if isinstance(value, dict) and 'weight' in value:
                    self.selection_criteria[criterion] = value['weight']
                else:
                    self.selection_criteria[criterion] = value
        else:
            self.selection_criteria = default_criteria
            
        # Get configurable parameters
        self.min_diversity_threshold = self.config.get('min_diversity_threshold', 0.3)
        self.batch_size = self.config.get('batch_size', 50)
        self.large_dataset_threshold = self.config.get('large_dataset_threshold', 50)
        
        if self.logger:
            self.logger.log_info(f"Initialized InternalSelector with criteria: {list(self.selection_criteria.keys())}", "internal_selector")

    def _debug_log_idea(self, idea: 'ResearchIdea', source_method: str = "unknown"):
        """Log full idea details in debug mode"""
        if self.logger and hasattr(self.logger, 'debug_mode') and self.logger.debug_mode:
            debug_info = f"""
========== RESEARCH IDEA PROCESSED BY SELECTOR ==========
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
=========================================================""".strip()
            self.logger.log_debug(debug_info, "internal_selector")

        # Also do a brief info log even without debug mode
        elif self.logger:
            self.logger.log_info(f"Processed idea: '{idea.topic}' via {source_method} ({idea.method})", "internal_selector")

    def _get_retry_params(self, operation_type: str = 'default') -> tuple:
        """Get retry parameters from config for different operations"""
        retry_config = self.config.get('retry', {}) if hasattr(self, 'config') else {}

        if operation_type == 'merging':
            max_retries = retry_config.get('idea_merging_retries', 3)
            timeout = self.config.get('llm', {}).get('timeouts', {}).get('merging', 120)
        else:
            max_retries = retry_config.get('max_retries', 5)
            timeout = self.config.get('llm', {}).get('timeouts', {}).get('default', 300)

        delay = retry_config.get('delay', 2)
        return max_retries, timeout, delay

    @limit_async_func_call(max_size=3)
    async def quick_evaluate_ideas(self, ideas: List[ResearchIdea]) -> List[Tuple[ResearchIdea, float]]:
        """
        Perform quick evaluation of ideas to get preliminary scores.
        This is faster than full review but provides good ranking information.
        """
        if self.logger:
            self.logger.log_info(f"Quick evaluating {len(ideas)} ideas", "internal_selector")
        
        # Process ideas in batches asynchronously for efficiency
        batch_size = min(self.batch_size, len(ideas))  # Use configurable batch size
        
        # Create batches and process them concurrently
        batches = [ideas[i:i + batch_size] for i in range(0, len(ideas), batch_size)]
        
        # Process batches concurrently
        batch_tasks = [self._evaluate_idea_batch_async(batch) for batch in batches]
        batch_results = await asyncio.gather(*batch_tasks)
        
        # Flatten results
        evaluated_ideas = []
        for batch, batch_scores in zip(batches, batch_results):
            for idea, score in zip(batch, batch_scores):
                evaluated_ideas.append((idea, score))
        
        if self.logger:
            avg_score = sum(score for _, score in evaluated_ideas) / len(evaluated_ideas) if evaluated_ideas else 0
            self.logger.log_info(f"Quick evaluation completed - Average score: {avg_score:.2f}", "internal_selector")
        
        return evaluated_ideas
    
    def _evaluate_idea_batch(self, ideas: List[ResearchIdea]) -> List[float]:
        """Evaluate a batch of ideas efficiently (sync version for fallback)"""
        # Simple scoring based on idea characteristics
        return [3.0 + (len(idea.topic) % 3) * 0.5 for idea in ideas]
    
    @retry_with_timeout(max_retries=5, timeout=300, delay=2)
    async def _evaluate_idea_batch_async(self, ideas: List[ResearchIdea]) -> List[float]:
        """Evaluate a batch of ideas efficiently (async version)"""
        
        # Check if LLM is available
        if not self.llm:
            # Fallback to simple scoring based on idea characteristics
            base_score = self.config.get('fallback_scoring', {}).get('base_score', 3.0)
            modifier = self.config.get('fallback_scoring', {}).get('topic_length_modifier', 0.5)
            return [base_score + (len(idea.topic) % 3) * modifier for idea in ideas]
        
        # Format ideas for batch evaluation
        ideas_text = []
        for i, idea in enumerate(ideas, 1):
            ideas_text.append(f"""
IDEA {i}:
Topic: {idea.topic}
Problem: {idea.problem_statement[:self.config.get('content_limits', {}).get('problem_statement_length', 200)]}...
Method: {idea.proposed_methodology[:self.config.get('content_limits', {}).get('methodology_length', 200)]}...
Validation: {idea.experimental_validation[:self.config.get('content_limits', {}).get('validation_length', 200)]}...
""")
        
        user_prompt = QUICK_EVALUATION_USER_PROMPT.format(
            ideas_batch="\n".join(ideas_text),
            criteria_descriptions=self._format_criteria_descriptions()
        )

        max_tokens = self.config.get('llm', {}).get('max_tokens', 16384)
        response = await self.llm.generate_with_system_prompt(
            QUICK_EVALUATION_SYSTEM_PROMPT,
            user_prompt,
            max_tokens=max_tokens,
            caller="internal_selector"
        )
        
        # Parse scores from response
        scores = self._parse_batch_scores(response, len(ideas))
        return scores
    
    def _format_criteria_descriptions(self) -> str:
        """Format selection criteria for prompt"""
        criteria_desc = []
        for criterion, weight in self.selection_criteria.items():
            criteria_desc.append(f"- {criterion.title()} ({weight*100:.0f}%)")
        return "\n".join(criteria_desc)
    
    def _parse_batch_scores(self, response: str, num_ideas: int) -> List[float]:
        """Parse scores from LLM response"""
        scores = []
        
        # Look for patterns like "IDEA 1: 4.2/5" or "Score: 3.8"
        score_patterns = [
            r'IDEA\s+\d+.*?(\d+\.?\d*)/5',
            r'IDEA\s+\d+.*?Score:\s*(\d+\.?\d*)',
            r'IDEA\s+\d+.*?(\d+\.?\d*)\s*$'
        ]
        
        for pattern in score_patterns:
            found_scores = re.findall(pattern, response, re.MULTILINE | re.IGNORECASE)
            if len(found_scores) >= num_ideas:
                scores = [float(s) for s in found_scores[:num_ideas]]
                break
        
        # Fallback: extract all numbers that look like scores
        if not scores:
            all_numbers = re.findall(r'(\d+\.?\d*)', response)
            potential_scores = [float(n) for n in all_numbers if 0 <= float(n) <= 5]
            scores = potential_scores[:num_ideas]
        
        # Final fallback: assign default scores
        default_score = self.config.get('fallback_scoring', {}).get('base_score', 3.0)
        while len(scores) < num_ideas:
            scores.append(default_score)  # Default neutral score
        
        return scores[:num_ideas]
    
    def select_top_ideas(self, evaluated_ideas: List[Tuple[ResearchIdea, float]], 
                        target_count: int, 
                        min_diversity: float = None) -> List[ResearchIdea]:
        """
        Select top ideas from evaluated pool, with smart merging for similar high-quality ideas.
        
        Args:
            evaluated_ideas: List of (idea, score) tuples
            target_count: Number of ideas to select
            min_diversity: Minimum diversity threshold between selected ideas (uses config if None)
        """
        # Use configured diversity threshold if not provided
        if min_diversity is None:
            min_diversity = self.min_diversity_threshold

        merge_threshold = self.config.get('default_merge_threshold', 0.85)  # Similarity threshold for merging (higher than diversity threshold)
        
        # For large datasets, use more efficient clustering-based approach
        if len(evaluated_ideas) > self.large_dataset_threshold:
            return self._select_top_ideas_efficient(evaluated_ideas, target_count, min_diversity, merge_threshold)
        else:
            return self._select_top_ideas_standard(evaluated_ideas, target_count, min_diversity, merge_threshold)
    
    def _select_top_ideas_standard(self, evaluated_ideas: List[Tuple[ResearchIdea, float]], 
                                  target_count: int, min_diversity: float, merge_threshold: float) -> List[ResearchIdea]:
        """Standard O(n²) approach for smaller datasets"""    
        if self.logger:
            self.logger.log_info(f"Using standard selection for {len(evaluated_ideas)} ideas (target: {target_count}, diversity: {min_diversity}, merge: {merge_threshold})", "internal_selector")
        
        # Sort by score descending
        sorted_ideas = sorted(evaluated_ideas, key=lambda x: x[1], reverse=True)
        
        selected_ideas = []
        merged_count = 0
        
        for idea, score in sorted_ideas:
            if len(selected_ideas) >= target_count:
                break
            
            # Check for potential merge candidate
            merge_candidate, similarity = self._find_merge_candidate(idea, selected_ideas, merge_threshold)
            
            if merge_candidate is not None:
                # Merge instead of discarding
                merged_idea = self._merge_ideas(merge_candidate, idea, similarity)
                # Replace the original idea in selected_ideas
                for i, selected in enumerate(selected_ideas):
                    if selected.topic == merge_candidate.topic:
                        selected_ideas[i] = merged_idea
                        merged_count += 1
                        if self.logger:
                            self.logger.log_info(f"Merged similar ideas: '{merge_candidate.topic}' + '{idea.topic}' -> '{merged_idea.topic}' (similarity: {similarity:.2f})", "internal_selector")
                        break
            elif self._is_sufficiently_diverse(idea, selected_ideas, min_diversity):
                selected_ideas.append(idea)
                if self.logger:
                    self.logger.log_info(f"Selected idea: {idea.topic} (score: {score:.2f})", "internal_selector")
            else:
                if self.logger:
                    self.logger.log_debug(f"Skipped similar idea: {idea.topic} (score: {score:.2f})", "internal_selector")
        
        # If we don't have enough distinct ideas, fill with highest scored remaining ideas
        if len(selected_ideas) < target_count:
            remaining_needed = target_count - len(selected_ideas)
            for idea, score in sorted_ideas:
                if idea not in selected_ideas and not any(idea.topic == sel.topic for sel in selected_ideas):
                    selected_ideas.append(idea)
                    remaining_needed -= 1
                    if remaining_needed <= 0:
                        break
        
        if self.logger:
            self.logger.log_info(f"Standard selection complete: {len(selected_ideas)} ideas chosen, {merged_count} merges performed", "internal_selector")
        
        return selected_ideas[:target_count]
    
    def _select_top_ideas_efficient(self, evaluated_ideas: List[Tuple[ResearchIdea, float]], 
                                   target_count: int, min_diversity: float, merge_threshold: float) -> List[ResearchIdea]:
        """
        Efficient O(n log n) approach using clustering for large datasets.
        First clusters similar ideas, then selects best from each cluster.
        """
        if self.logger:
            self.logger.log_info(f"Using efficient clustering-based selection for {len(evaluated_ideas)} ideas", "internal_selector")
        
        # Sort by score descending  
        sorted_ideas = sorted(evaluated_ideas, key=lambda x: x[1], reverse=True)
        ideas = [idea for idea, score in sorted_ideas]
        scores = [score for idea, score in sorted_ideas]
        
        # Build similarity matrix more efficiently using vectorized operations
        similarity_matrix = self._build_similarity_matrix_efficient(ideas)
        
        # Cluster similar ideas using a greedy approach
        clusters = self._cluster_ideas_greedy(ideas, scores, similarity_matrix, merge_threshold)
        
        if self.logger:
            self.logger.log_info(f"Formed {len(clusters)} clusters from {len(ideas)} ideas", "internal_selector")
        
        # Select best ideas from clusters, ensuring diversity
        selected_ideas = []
        merged_count = 0
        
        # Sort clusters by best score in each cluster
        clusters.sort(key=lambda cluster: max(scores[i] for i in cluster), reverse=True)
        
        for cluster_indices in clusters:
            if len(selected_ideas) >= target_count:
                break
                
            if len(cluster_indices) == 1:
                # Single idea cluster - just add it
                selected_ideas.append(ideas[cluster_indices[0]])
            else:
                # Multiple ideas in cluster - merge them
                cluster_ideas = [ideas[i] for i in cluster_indices]
                cluster_scores = [scores[i] for i in cluster_indices]
                
                # Create merged idea from top ideas in cluster
                merged_idea = self._merge_cluster_ideas(cluster_ideas, cluster_scores)
                selected_ideas.append(merged_idea)
                merged_count += len(cluster_indices) - 1
                
                if self.logger:
                    topics = [ideas[i].topic for i in cluster_indices]
                    self.logger.log_info(f"Merged cluster of {len(cluster_indices)} ideas: {topics} -> '{merged_idea.topic}'", "internal_selector")
        
        if self.logger:
            self.logger.log_info(f"Efficient selection complete: {len(selected_ideas)} ideas chosen, {merged_count} merges performed", "internal_selector")
        
        return selected_ideas[:target_count]
    
    def _build_similarity_matrix_efficient(self, ideas: List[ResearchIdea]) -> np.ndarray:
        """Build similarity matrix more efficiently using vectorized operations"""
        n = len(ideas)
        similarity_matrix = np.zeros((n, n))
        
        # Pre-compute word sets for all ideas
        word_sets = []
        for idea in ideas:
            content = f"{idea.topic} {idea.problem_statement} {idea.proposed_methodology}"
            words = set(content.lower().split())
            word_sets.append(words)
        
        # Calculate pairwise similarities
        for i in range(n):
            for j in range(i+1, n):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                similarity = intersection / union if union > 0 else 0.0
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity  # Symmetric matrix
        
        return similarity_matrix
    
    def _cluster_ideas_greedy(self, ideas: List[ResearchIdea], scores: List[float], 
                             similarity_matrix: np.ndarray, merge_threshold: float) -> List[List[int]]:
        """
        Greedy clustering algorithm to group similar ideas.
        Returns list of clusters, where each cluster is a list of idea indices.
        """
        n = len(ideas)
        clusters = []
        assigned = [False] * n
        
        # Process ideas in order of decreasing score
        for i in range(n):
            if assigned[i]:
                continue
                
            # Start new cluster with current idea
            current_cluster = [i]
            assigned[i] = True
            
            # Find other unassigned ideas that are similar enough to merge
            for j in range(i+1, n):
                if assigned[j]:
                    continue
                    
                # Check if idea j is similar to any idea in current cluster
                should_merge = False
                for cluster_idx in current_cluster:
                    if similarity_matrix[cluster_idx][j] >= merge_threshold:
                        should_merge = True
                        break
                
                if should_merge:
                    current_cluster.append(j)
                    assigned[j] = True
            
            clusters.append(current_cluster)
        
        return clusters
    
    def _merge_cluster_ideas(self, cluster_ideas: List[ResearchIdea], cluster_scores: List[float]) -> ResearchIdea:
        """
        Merge multiple ideas from a cluster into a single comprehensive idea.
        Uses the highest-scored idea as the base and incorporates content from others.
        """
        if len(cluster_ideas) == 1:
            return cluster_ideas[0]
        
        # Sort by score to get best ideas first
        sorted_pairs = sorted(zip(cluster_ideas, cluster_scores), key=lambda x: x[1], reverse=True)
        best_ideas = [pair[0] for pair in sorted_pairs]
        
        # Use best idea as base
        base_idea = best_ideas[0]
        
        # Incrementally merge other ideas
        merge_threshold = self.config.get('default_merge_threshold', 0.85)
        merged_idea = base_idea
        for other_idea in best_ideas[1:]:
            merged_idea = self._merge_ideas(merged_idea, other_idea, merge_threshold)  # Use config threshold
        
        return merged_idea
    
    def _is_sufficiently_diverse(self, new_idea: ResearchIdea, existing_ideas: List[ResearchIdea],
                                 min_threshold: float) -> bool:
        """Check if new idea is sufficiently diverse from existing selected ideas"""
        if not existing_ideas:
            return True

        for existing_idea in existing_ideas:
            similarity = calculate_idea_similarity(new_idea, existing_idea)
            if similarity > (1.0 - min_threshold):  # Convert distinctness to similarity threshold
                return False

        return True
    
    def _find_merge_candidate(self, new_idea: ResearchIdea, existing_ideas: List[ResearchIdea],
                             merge_threshold: float) -> Tuple[ResearchIdea, float]:
        """
        Find the best candidate for merging with new_idea from existing_ideas.

        Returns:
            Tuple of (merge_candidate, similarity_score) or (None, 0.0) if no suitable candidate
        """
        return find_most_similar_idea(new_idea, existing_ideas, merge_threshold)
    
    def _merge_ideas(self, idea1: ResearchIdea, idea2: ResearchIdea, similarity: float) -> ResearchIdea:
        """
        Merge two similar ideas into a single, more comprehensive idea.
        
        Args:
            idea1: First idea (already selected)
            idea2: Second idea (candidate for merging)
            similarity: Similarity score between the ideas
            
        Returns:
            Merged ResearchIdea combining the best aspects of both
        """
        # Create merged topic combining both ideas
        merged_topic = f"{idea1.topic.strip()} and {idea2.topic.strip()}"
        if len(merged_topic) > 150:  # Keep topic reasonable length
            merged_topic = f"{idea1.topic.strip()} (Enhanced)"
        
        # Merge problem statements
        merged_problem = self._merge_text_sections(
            idea1.problem_statement, idea2.problem_statement,
            "The problem encompasses"
        )
        
        # Merge methodologies
        merged_methodology = self._merge_text_sections(
            idea1.proposed_methodology, idea2.proposed_methodology,
            "The integrated approach combines"
        )
        
        # Merge experimental validation
        merged_validation = self._merge_text_sections(
            idea1.experimental_validation, idea2.experimental_validation,
            "Validation will include"
        )
        
        # Merge potential impact
        merged_impact = self._merge_text_sections(
            idea1.potential_impact, idea2.potential_impact,
            "The combined impact includes"
        )
        
        # Create merged idea
        merged_facets = {
            'Problem Statement': merged_problem,
            'Proposed Methodology': merged_methodology,
            'Experimental Validation': merged_validation,
            'Potential Impact': merged_impact
        }

        merged_idea = ResearchIdea(
            topic=merged_topic,
            facets=merged_facets,
            source="internal_selector",
            method="idea_merging"
        )
        # Debug log the merged idea
        self._debug_log_idea(merged_idea, "idea_merging")

        return merged_idea
    
    def _merge_text_sections(self, text1: str, text2: str, prefix: str) -> str:
        """
        Intelligently merge two text sections, avoiding redundancy.
        
        Args:
            text1: First text section
            text2: Second text section  
            prefix: Prefix for merged text
            
        Returns:
            Merged text combining unique aspects of both sections
        """
        if not text1 and not text2:
            return ""
        elif not text1:
            return text2
        elif not text2:
            return text1
        
        # Split into sentences for better merging
        sentences1 = [s.strip() for s in text1.split('.') if s.strip()]
        sentences2 = [s.strip() for s in text2.split('.') if s.strip()]
        
        # Find unique sentences (basic deduplication)
        all_sentences = sentences1.copy()
        for sentence in sentences2:
            # Check if this sentence is substantially different from existing ones
            is_unique = True
            for existing in sentences1:
                # Simple similarity check based on shared words
                sentence_dedup_threshold = self.config.get('sentence_dedup_threshold', 0.7)
                words1 = set(existing.lower().split())
                words2 = set(sentence.lower().split())
                if len(words1) > 0 and len(words2) > 0:
                    overlap = len(words1 & words2) / len(words1 | words2)
                    if overlap > sentence_dedup_threshold:  # Use config threshold
                        is_unique = False
                        break
            
            if is_unique:
                all_sentences.append(sentence)
        
        # Reconstruct merged text
        merged_text = '. '.join(all_sentences[:5])  # Limit to 5 sentences to avoid bloat
        if merged_text and not merged_text.endswith('.'):
            merged_text += '.'
            
        return merged_text
    
    @limit_async_func_call(max_size=2)
    @retry_with_timeout(max_retries=5, timeout=300, delay=2)
    async def rank_ideas_comparatively(self, ideas: List[ResearchIdea]) -> List[Tuple[ResearchIdea, Dict]]:
        """
        Perform comparative ranking of ideas to get relative rankings.
        More accurate than individual scoring for small sets of ideas.
        """
        if len(ideas) <= 1:
            return [(ideas[0], {'rank': 1, 'score': 4.0})] if ideas else []
        
        if self.logger:
            self.logger.log_info(f"Performing async comparative ranking of {len(ideas)} ideas", "internal_selector")
        
        # Format ideas for comparison - include complete content
        ideas_text = []
        for i, idea in enumerate(ideas, 1):
            ideas_text.append(f"""
IDEA {i}: {idea.topic}

Problem Statement:
{idea.problem_statement}

Proposed Methodology:
{idea.proposed_methodology}

Experimental Validation:
{idea.experimental_validation}

""".strip())
        
        user_prompt = COMPARATIVE_RANKING_USER_PROMPT.format(
            ideas_to_rank="\n".join(ideas_text),
            num_ideas=len(ideas)
        )
        
        # Check if LLM is available  
        if not self.llm:
            # Fallback: simple ranking by topic length as a proxy
            simple_ranking = [(idea, {'rank': i+1, 'score': 5.0 - i*0.5, 'comparative_evaluation': False})
                             for i, idea in enumerate(ideas)]
            return simple_ranking

        max_tokens = self.config.get('llm', {}).get('max_tokens', 16384)
        response = await self.llm.generate_with_system_prompt(
            COMPARATIVE_RANKING_SYSTEM_PROMPT,
            user_prompt,
            max_tokens=max_tokens,
            caller="internal_selector"
        )
        
        # Parse ranking results
        ranking_results = self._parse_comparative_ranking(response, ideas)
        
        if self.logger:
            self.logger.log_info(f"Async comparative ranking completed", "internal_selector")
        
        return ranking_results
    
    def _parse_comparative_ranking(self, response: str, ideas: List[ResearchIdea]) -> List[Tuple[ResearchIdea, Dict]]:
        """Parse comparative ranking results"""
        ranking_results = []
        
        # Look for ranking patterns like "1. IDEA 3" or "Rank 1: IDEA 2"
        ranking_pattern = r'(?:Rank\s+)?(\d+)[\.\:\s]+IDEA\s+(\d+)'
        matches = re.findall(ranking_pattern, response, re.IGNORECASE)
        
        idea_rankings = {}
        for rank_str, idea_num_str in matches:
            try:
                rank = int(rank_str)
                idea_idx = int(idea_num_str) - 1  # Convert to 0-based index
                if 0 <= idea_idx < len(ideas):
                    idea_rankings[idea_idx] = rank
            except (ValueError, IndexError):
                continue
        
        # Create results with rankings
        for i, idea in enumerate(ideas):
            rank = idea_rankings.get(i, len(ideas))  # Default to last rank if not found
            score = 5.0 - ((rank - 1) * 0.5)  # Convert rank to score (rank 1 = 5.0, rank 2 = 4.5, etc.)
            score = max(1.0, score)  # Ensure minimum score of 1.0
            
            ranking_results.append((idea, {
                'rank': rank,
                'score': score,
                'comparative_evaluation': True
            }))
        
        # Sort by rank to ensure proper ordering
        ranking_results.sort(key=lambda x: x[1]['rank'])
        
        return ranking_results
        
    async def select_ideas_with_llm_merging(self, evaluated_ideas: List[Tuple[ResearchIdea, float]],
                                           target_count: int,
                                           merge_threshold: float = None) -> List[ResearchIdea]:
        """
        Select ideas using iterative merging with LLM. Uses naive O(n²) Jaccard similarity 
        to find the most similar pair, then LLM to merge them when similarity exceeds threshold.
        Iteratively merges one pair per iteration until max similarity < threshold.
        
        Args:
            evaluated_ideas: List of (idea, score) tuples
            target_count: Desired number of final ideas
            merge_threshold: Jaccard similarity threshold for merging (0.0-1.0, default 0.85)
            
        Returns:
            List of selected/merged ideas
        """
        if merge_threshold is None:
            merge_threshold = self.config.get('default_merge_threshold', 0.85)

        if self.logger:
            self.logger.log_info(f"Starting LLM-based iterative merging for {len(evaluated_ideas)} ideas (target: {target_count}, threshold: {merge_threshold})", "internal_selector")
        
        # Sort by score descending and take top candidates
        sorted_ideas = sorted(evaluated_ideas, key=lambda x: x[1], reverse=True)
        
        # Start with more ideas than target to allow for merging
        initial_multiplier = self.config.get('initial_multiplier', 3)
        initial_count = min(target_count * initial_multiplier, len(sorted_ideas))  # Multiplier for merging room
        current_ideas = [idea for idea, score in sorted_ideas[:initial_count]]
        
        if self.logger:
            self.logger.log_info(f"Starting iterative merging with {len(current_ideas)} top-scored ideas", "internal_selector")
        
        merge_iteration = 0
        total_merges = 0
        
        # Iteratively merge until no similarities above threshold OR target count reached
        while True:
            merge_iteration += 1
            
            if self.logger:
                self.logger.log_info(f"Merge iteration {merge_iteration}: {len(current_ideas)} ideas remaining", "internal_selector")
            
            # Find the most similar pair using naive O(n²) Jaccard similarity  
            most_similar_pair, max_similarity = self._find_most_similar_pair_naive(current_ideas, merge_threshold)
            
            # Stop if no pair exceeds threshold
            if most_similar_pair is None or max_similarity < merge_threshold:
                if self.logger:
                    self.logger.log_info(f"No more similarities above threshold {merge_threshold} found (max: {max_similarity:.3f}). Stopping iteration.", "internal_selector")
                break
            
            # Stop if we've already reached target count or fewer
            if len(current_ideas) <= target_count:
                if self.logger:
                    self.logger.log_info(f"Reached target count {target_count}. Stopping iteration.", "internal_selector")
                break
            
            idea_a, idea_b = most_similar_pair
            
            # Merge the most similar pair using LLM
            if self.llm:
                merged_idea = await self._merge_ideas_with_llm(idea_a, idea_b)
            else:
                # Fallback to simple text merging if no LLM
                merged_idea = self._merge_ideas(idea_a, idea_b, max_similarity)
            
            # Replace the two ideas with the merged one
            current_ideas = [idea for idea in current_ideas if idea != idea_a and idea != idea_b]
            current_ideas.append(merged_idea)
            total_merges += 1
            
            if self.logger:
                self.logger.log_info(f"Merged '{idea_a.topic}' + '{idea_b.topic}' -> '{merged_idea.topic}' (Jaccard: {max_similarity:.3f})", "internal_selector")
            
            # Safety check to prevent infinite loops
            max_iterations = self.config.get('max_merge_iterations', 20)
            if merge_iteration > max_iterations:
                if self.logger:
                    self.logger.log_warning(f"Reached maximum merge iterations ({max_iterations}), stopping", "internal_selector")
                break
        
        # If we still have more ideas than target, select the best ones by simple truncation
        final_ideas = current_ideas[:target_count]
        
        if self.logger:
            self.logger.log_info(f"LLM-based iterative merging complete: {len(final_ideas)} final ideas after {total_merges} merges in {merge_iteration} iterations", "internal_selector")
        
        return final_ideas
    
    def _find_most_similar_pair_naive(self, ideas: List[ResearchIdea],
                                     min_threshold: float) -> Tuple[Tuple[ResearchIdea, ResearchIdea], float]:
        """
        Find the most similar pair of ideas using naive O(n²) approach.

        Args:
            ideas: List of ideas to compare
            min_threshold: Minimum threshold (used for filtering)

        Returns:
            Tuple of ((idea_a, idea_b), similarity_score) or (None, max_similarity)
        """
        if len(ideas) < 2:
            return None, 0.0

        max_similarity = 0.0
        best_pair = None

        # Naive O(n²) implementation - check all pairs
        for i in range(len(ideas)):
            for j in range(i+1, len(ideas)):
                idea_a, idea_b = ideas[i], ideas[j]
                similarity = calculate_idea_similarity(idea_a, idea_b)

                # Track the highest similarity pair
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_pair = (idea_a, idea_b)

        # Only return if above threshold
        if best_pair and max_similarity >= min_threshold:
            return best_pair, max_similarity
        else:
            return None, max_similarity  # Return actual max similarity even if below threshold for logging
    
    @retry_with_timeout(max_retries=3, timeout=120, delay=2)
    async def _merge_ideas_with_llm(self, idea_a: ResearchIdea, idea_b: ResearchIdea) -> ResearchIdea:
        """Merge two ideas using LLM analysis"""
        
        ideas_text = f"""
IDEA 1:
Topic: {idea_a.topic}
Problem Statement: {idea_a.problem_statement}
Proposed Methodology: {idea_a.proposed_methodology}
Experimental Validation: {idea_a.experimental_validation}
Potential Impact: {idea_a.potential_impact}

IDEA 2:
Topic: {idea_b.topic}
Problem Statement: {idea_b.problem_statement}
Proposed Methodology: {idea_b.proposed_methodology}
Experimental Validation: {idea_b.experimental_validation}
Potential Impact: {idea_b.potential_impact}
"""

        user_prompt = IDEA_MERGING_USER_PROMPT.format(ideas_to_merge=ideas_text)

        max_tokens = self.config.get('llm', {}).get('max_tokens', 16384)
        response = await self.llm.generate_with_system_prompt(
            IDEA_MERGING_SYSTEM_PROMPT,
            user_prompt,
            max_tokens=max_tokens,
            caller="internal_selector"
        )
        
        # Parse the merged idea from response
        merged_idea = self._parse_merged_idea(response, idea_a, idea_b)
        return merged_idea
    
    def _parse_merged_idea(self, response: str, fallback_a: ResearchIdea, fallback_b: ResearchIdea) -> ResearchIdea:
        """Parse merged idea from LLM response with fallback"""
        
        try:
            # Try to extract structured fields from response
            topic = self._extract_field(response, ["Topic", "Unified Topic", "Title"])
            problem = self._extract_field(response, ["Problem Statement", "Integrated Problem Statement", "Problem"])
            methodology = self._extract_field(response, ["Methodology", "Combined Methodology", "Method", "Approach"])
            validation = self._extract_field(response, ["Experimental Validation", "Validation", "Experiments"])
            impact = self._extract_field(response, ["Potential Impact", "Enhanced Potential Impact", "Impact"])
            
            # Create merged idea with extracted fields, fallback to combination if parsing fails
            merged_facets = {
                "Problem Statement": problem or self._merge_text_sections(fallback_a.problem_statement, fallback_b.problem_statement, "Combined problem"),
                "Proposed Methodology": methodology or self._merge_text_sections(fallback_a.proposed_methodology, fallback_b.proposed_methodology, "Integrated approach"),
                "Experimental Validation": validation or self._merge_text_sections(fallback_a.experimental_validation, fallback_b.experimental_validation, "Combined validation"),
                "Potential Impact": impact or self._merge_text_sections(fallback_a.potential_impact, fallback_b.potential_impact, "Enhanced impact")
            }

            merged_idea = ResearchIdea(
                topic=topic or f"{fallback_a.topic} and {fallback_b.topic} (Integrated)",
                facets=merged_facets,
                source="internal_selector",
                method="llm_merging"
            )
            # Debug log the merged idea
            self._debug_log_idea(merged_idea, "llm_merging")

            return merged_idea
            
        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"Failed to parse LLM-merged idea, using fallback: {e}", "internal_selector")
            
            # Fallback to simple text merging
            return self._merge_ideas(fallback_a, fallback_b, 0.8)
    
    def _extract_field(self, text: str, field_names: List[str]) -> str:
        """Extract a specific field from structured LLM response using JSON-first approach"""
        # Try JSON parsing first
        json_result = safe_json_parse(text)

        if json_result and isinstance(json_result, dict):
            for field_name in field_names:
                field_key = field_name.lower().replace(" ", "_")
                if field_key in json_result:
                    return str(json_result[field_key])
                if field_name in json_result:
                    return str(json_result[field_name])

        # Fallback to text extraction
        from ..utils.text_utils import extract_field_from_response
        return extract_field_from_response(text, field_names)
    
    # Implementation of BaseAgent abstract methods
    def gather_information(self):
        """Gather selection criteria and thresholds"""
        return {
            'selection_criteria': self.selection_criteria,
            'capabilities': ['quick_evaluation', 'comparative_ranking', 'diversity_filtering']
        }
    
    def generate_ideas(self):
        """IdeaSelector doesn't generate ideas, it selects from existing ones"""
        return {'message': 'IdeaSelector focuses on selection, not generation'}
    
    def critique_ideas(self):
        """Provide quick critique for filtering purposes"""
        return {'message': 'Use quick_evaluate_ideas() for lightweight critique'}
    
    def refine_ideas(self):
        """IdeaSelector focuses on selection rather than refinement"""
        return {'message': 'IdeaSelector focuses on selection, not refinement'}