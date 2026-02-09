"""
Selection Agent

This agent provides unified idea selection functionality:
- External selection: Filter ideas based on distinctness from existing literature
- Internal selection: Deduplicate and merge similar generated ideas

Both use the same similarity calculation methods (TF-IDF or embeddings).
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .retrieval_agent import Paper
from .idea_gen.research_idea import ResearchIdea
from ..utils.llm_interface import LLMInterface
from ..utils.async_utils import limit_async_func_call, retry_with_timeout
from ..skills.selection import (
    IDEA_MERGING_SYSTEM_PROMPT, IDEA_MERGING_USER_PROMPT,
    format_ideas_for_merge
)


@dataclass
class SimilarityResult:
    """Result of similarity analysis"""
    item_id: str  # idea topic or paper title
    max_similarity_score: float
    most_similar_item: str
    similarity_details: List[Dict]
    is_sufficiently_distinct: bool
    recommendations: List[str]

    def to_dict(self) -> Dict:
        return {
            'item_id': self.item_id,
            'max_similarity_score': self.max_similarity_score,
            'most_similar_item': self.most_similar_item,
            'similarity_details': self.similarity_details,
            'is_sufficiently_distinct': self.is_sufficiently_distinct,
            'recommendations': self.recommendations
        }


class SelectionAgent:
    """
    Unified selector for filtering and selecting research ideas.

    Supports:
    - External selection: filtering against existing literature (papers)
    - Internal selection: deduplication and merging among generated ideas
    """

    def __init__(self, selection_config: Dict = None, logger=None, llm_config: Dict = None):
        self.selection_config = selection_config or {}
        self.logger = logger

        # Initialize LLM for merging operations
        if llm_config and llm_config.get("api_key"):
            self.llm = LLMInterface(config=llm_config, logger=logger)
        else:
            self.llm = None

        # Similarity thresholds
        self.similarity_threshold = self.selection_config['similarity_threshold']
        self.high_similarity_threshold = self.selection_config['high_similarity_threshold']
        self.low_similarity_threshold = self.selection_config['low_similarity_threshold']

        # Similarity method
        self.similarity_method = self.selection_config['similarity_method']

        # TF-IDF vectorizer
        tfidf_config = self.selection_config['tfidf']
        self.vectorizer = TfidfVectorizer(
            max_features=tfidf_config['max_features'],
            stop_words=tfidf_config['stop_words'],
            ngram_range=tuple(tfidf_config['ngram_range']),
            min_df=tfidf_config['min_df'],
        )

        # Embedding models (lazy loaded)
        self._embedding_models = {}

        # Merging config
        self.merge_threshold = self.selection_config['merge_threshold']
        self.min_diversity_threshold = self.selection_config['min_diversity_threshold']
        self.max_similarity_details = self.selection_config['max_similarity_details']

        if self.logger:
            self.logger.log_info(
                f"Initialized SelectionAgent with {self.similarity_method} similarity",
                "idea_selector"
            )

    # ==================== Similarity Calculation ====================

    def calculate_similarities(self, query_texts: List[str], corpus_texts: List[str]) -> np.ndarray:
        """
        Calculate similarity matrix between query texts and corpus texts.

        Args:
            query_texts: List of query texts (e.g., idea summaries)
            corpus_texts: List of corpus texts (e.g., paper summaries)

        Returns:
            Similarity matrix of shape (len(query_texts), len(corpus_texts))
        """
        if not query_texts or not corpus_texts:
            return np.array([])

        try:
            if self.similarity_method == 'tfidf':
                return self._calculate_tfidf_similarities(query_texts, corpus_texts)
            elif self.similarity_method in ['bge', 'stella', 'jina']:
                return self._calculate_embedding_similarities(query_texts, corpus_texts)
            else:
                if self.logger:
                    self.logger.log_warning(
                        f"Unknown similarity method: {self.similarity_method}, using TF-IDF",
                        "idea_selector"
                    )
                return self._calculate_tfidf_similarities(query_texts, corpus_texts)
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error calculating similarities: {e}", "idea_selector", e)
            return np.zeros((len(query_texts), len(corpus_texts)))

    def _calculate_tfidf_similarities(self, query_texts: List[str], corpus_texts: List[str]) -> np.ndarray:
        """Calculate TF-IDF cosine similarities"""
        all_texts = query_texts + corpus_texts
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)

        query_vectors = tfidf_matrix[:len(query_texts)]
        corpus_vectors = tfidf_matrix[len(query_texts):]

        return cosine_similarity(query_vectors, corpus_vectors)

    def _calculate_embedding_similarities(self, query_texts: List[str], corpus_texts: List[str]) -> np.ndarray:
        """Calculate embedding-based cosine similarities"""
        model = self._get_embedding_model(self.similarity_method)

        all_texts = query_texts + corpus_texts
        embeddings = model.encode(all_texts, normalize_embeddings=True)

        query_embeddings = embeddings[:len(query_texts)]
        corpus_embeddings = embeddings[len(query_texts):]

        return cosine_similarity(query_embeddings, corpus_embeddings)

    def _get_embedding_model(self, model_name: str):
        """Get embedding model, loading it lazily"""
        if model_name not in self._embedding_models:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    f"sentence_transformers required for {model_name}. "
                    "Install with: pip install sentence-transformers"
                )

            model_map = {
                'bge': 'BAAI/bge-m3',
                'stella': 'dunzhang/stella_en_1.5B_v5',
                'jina': 'jinaai/jina-embeddings-v2-base-en'
            }

            model_id = model_map.get(model_name, 'all-MiniLM-L6-v2')
            if self.logger:
                self.logger.log_info(f"Loading embedding model: {model_id}", "idea_selector")

            try:
                self._embedding_models[model_name] = SentenceTransformer(model_id)
            except Exception:
                # Fallback to smaller model
                self._embedding_models[model_name] = SentenceTransformer('all-MiniLM-L6-v2')

        return self._embedding_models[model_name]

    # ==================== External Selection (vs Literature) ====================

    def filter_against_literature(
        self, ideas: List[ResearchIdea], papers: List[Paper]
    ) -> Tuple[List[ResearchIdea], List[SimilarityResult]]:
        """
        Filter ideas that are too similar to existing papers.

        Args:
            ideas: List of generated research ideas
            papers: List of papers from literature

        Returns:
            Tuple of (filtered_ideas, analysis_results)
        """
        if self.logger:
            self.logger.log_info(
                f"Filtering {len(ideas)} ideas against {len(papers)} papers",
                "idea_selector"
            )

        if not papers:
            # No papers to compare - all ideas pass
            results = [
                SimilarityResult(
                    item_id=idea.topic,
                    max_similarity_score=0.0,
                    most_similar_item="No papers to compare",
                    similarity_details=[],
                    is_sufficiently_distinct=True,
                    recommendations=["No existing papers found for comparison"]
                )
                for idea in ideas
            ]
            return ideas, results

        # Get summaries
        idea_summaries = [idea.summary for idea in ideas]
        paper_summaries = [paper.summary for paper in papers]

        # Calculate similarity matrix
        sim_matrix = self.calculate_similarities(idea_summaries, paper_summaries)

        # Analyze each idea
        results = []
        filtered_ideas = []

        for i, idea in enumerate(ideas):
            similarities = sim_matrix[i] if len(sim_matrix) > 0 else np.array([])
            result = self._analyze_idea_distinctness(idea, papers, similarities)
            results.append(result)

            if result.is_sufficiently_distinct:
                filtered_ideas.append(idea)

        if self.logger:
            self.logger.log_info(
                f"Filtered to {len(filtered_ideas)}/{len(ideas)} distinct ideas",
                "idea_selector"
            )

        return filtered_ideas, results

    def _analyze_idea_distinctness(
        self, idea: ResearchIdea, papers: List[Paper], similarities: np.ndarray
    ) -> SimilarityResult:
        """Analyze how distinct an idea is from papers"""
        if len(similarities) == 0:
            return SimilarityResult(
                item_id=idea.topic,
                max_similarity_score=0.0,
                most_similar_item="No papers",
                similarity_details=[],
                is_sufficiently_distinct=True,
                recommendations=[]
            )

        max_idx = np.argmax(similarities)
        max_sim = float(similarities[max_idx])
        most_similar = papers[max_idx].title

        # Build similarity details
        details = []
        for i, (paper, sim) in enumerate(zip(papers, similarities)):
            if sim > self.low_similarity_threshold:
                details.append({
                    'paper_title': paper.title,
                    'paper_id': paper.paper_id,
                    'similarity_score': float(sim),
                    'year': paper.year,
                    'overlap_type': self._classify_overlap(float(sim))
                })

        details.sort(key=lambda x: x['similarity_score'], reverse=True)
        details = details[:self.max_similarity_details]

        is_distinct = max_sim < self.similarity_threshold
        recommendations = self._generate_recommendations(max_sim)

        return SimilarityResult(
            item_id=idea.topic,
            max_similarity_score=max_sim,
            most_similar_item=most_similar,
            similarity_details=details,
            is_sufficiently_distinct=is_distinct,
            recommendations=recommendations
        )

    def _classify_overlap(self, similarity: float) -> str:
        """Classify overlap type based on similarity score"""
        if similarity >= self.high_similarity_threshold:
            return "high_overlap"
        elif similarity >= self.similarity_threshold:
            return "moderate_overlap"
        elif similarity >= self.low_similarity_threshold:
            return "low_overlap"
        return "minimal_overlap"

    def _generate_recommendations(self, max_similarity: float) -> List[str]:
        """Generate recommendations based on similarity level"""
        if max_similarity >= self.high_similarity_threshold:
            return [
                "CRITICAL: Very high similarity to existing work",
                "Consider a different approach or angle",
                "Review similar papers to find gaps"
            ]
        elif max_similarity >= self.similarity_threshold:
            return [
                "Moderate similarity to existing work",
                "Emphasize novel aspects more clearly",
                "Differentiate methodology from existing approaches"
            ]
        elif max_similarity >= self.low_similarity_threshold:
            return [
                "Good distinctness from existing work",
                "Minor overlaps - consider highlighting differences"
            ]
        return ["Excellent distinctness from existing literature"]

    # ==================== Internal Selection (Dedupe/Merge Ideas) ====================

    async def select_diverse_ideas(
        self,
        ideas: List[ResearchIdea],
        target_count: int,
        merge_similar: bool = True
    ) -> List[ResearchIdea]:
        """
        Select diverse ideas from pool, optionally merging similar ones.

        Args:
            ideas: List of generated research ideas
            target_count: Number of ideas to select
            merge_similar: Whether to merge similar ideas (True) or just filter (False)

        Returns:
            List of selected (and possibly merged) ideas
        """
        if self.logger:
            self.logger.log_info(
                f"Selecting {target_count} diverse ideas from {len(ideas)} (merge={merge_similar})",
                "idea_selector"
            )

        if len(ideas) <= target_count:
            return ideas

        # Get summaries and calculate pairwise similarities
        summaries = [idea.summary for idea in ideas]
        sim_matrix = self.calculate_similarities(summaries, summaries)

        if merge_similar:
            return await self._select_with_merging(ideas, sim_matrix, target_count)
        else:
            return self._select_without_merging(ideas, sim_matrix, target_count)

    def _select_without_merging(
        self, ideas: List[ResearchIdea], sim_matrix: np.ndarray, target_count: int
    ) -> List[ResearchIdea]:
        """Select diverse ideas without merging - just filter out similar ones"""
        selected = []
        selected_indices = []

        for i, idea in enumerate(ideas):
            if len(selected) >= target_count:
                break

            # Check if sufficiently diverse from already selected
            is_diverse = True
            for j in selected_indices:
                if sim_matrix[i][j] > (1.0 - self.min_diversity_threshold):
                    is_diverse = False
                    break

            if is_diverse:
                selected.append(idea)
                selected_indices.append(i)

        return selected

    async def _select_with_merging(
        self, ideas: List[ResearchIdea], sim_matrix: np.ndarray, target_count: int
    ) -> List[ResearchIdea]:
        """Select ideas with iterative merging of similar ones using LLM"""
        current_ideas = list(ideas)
        merge_count = 0
        max_iterations = self.selection_config['max_merge_iterations']

        for iteration in range(max_iterations):
            if len(current_ideas) <= target_count:
                break

            # Find most similar pair
            best_pair, best_sim = self._find_most_similar_pair(current_ideas)

            if best_pair is None or best_sim < self.merge_threshold:
                break

            # Merge the pair using LLM
            idea_a, idea_b = best_pair
            merged = await self._merge_ideas(idea_a, idea_b)

            # Replace pair with merged idea
            current_ideas = [
                idea for idea in current_ideas
                if idea != idea_a and idea != idea_b
            ]
            current_ideas.append(merged)
            merge_count += 1

            if self.logger:
                self.logger.log_info(
                    f"Merged: '{idea_a.topic[:30]}...' + '{idea_b.topic[:30]}...' (sim={best_sim:.2f})",
                    "idea_selector"
                )

        if self.logger:
            self.logger.log_info(
                f"Selection complete: {len(current_ideas)} ideas, {merge_count} merges",
                "idea_selector"
            )

        return current_ideas[:target_count]

    def _find_most_similar_pair(
        self, ideas: List[ResearchIdea]
    ) -> Tuple[Optional[Tuple[ResearchIdea, ResearchIdea]], float]:
        """Find the most similar pair of ideas"""
        if len(ideas) < 2:
            return None, 0.0

        summaries = [idea.summary for idea in ideas]
        sim_matrix = self.calculate_similarities(summaries, summaries)

        best_sim = 0.0
        best_pair = None

        for i in range(len(ideas)):
            for j in range(i + 1, len(ideas)):
                if sim_matrix[i][j] > best_sim:
                    best_sim = sim_matrix[i][j]
                    best_pair = (ideas[i], ideas[j])

        return best_pair, best_sim

    async def _merge_ideas(self, idea1: ResearchIdea, idea2: ResearchIdea) -> ResearchIdea:
        """Merge two similar ideas into one using LLM"""
        if not self.llm:
            # Fallback to simple merge if no LLM available
            return self._merge_ideas_simple(idea1, idea2)

        try:
            # Format ideas for LLM
            ideas_text = format_ideas_for_merge(idea1, idea2)

            # Call LLM to merge
            user_prompt = IDEA_MERGING_USER_PROMPT.format(ideas_to_merge=ideas_text)

            response = await self.llm.generate_with_system_prompt(
                IDEA_MERGING_SYSTEM_PROMPT,
                user_prompt,
                max_tokens=3000,
                caller="idea_merge"
            )

            # Parse LLM response into ResearchIdea
            merged_idea = self._parse_merged_idea(response)

            if self.logger:
                self.logger.log_info(
                    f"LLM merged: '{idea1.topic[:30]}...' + '{idea2.topic[:30]}...'",
                    "idea_selector"
                )

            return merged_idea

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"LLM merge failed, using fallback: {e}", "idea_selector", e)
            return self._merge_ideas_simple(idea1, idea2)

    def _parse_merged_idea(self, llm_response: str) -> ResearchIdea:
        """Parse LLM response into a ResearchIdea object"""
        # Extract sections from LLM response
        topic = self._extract_section(llm_response, ["Topic:", "Unified Topic:"])
        problem = self._extract_section(llm_response, ["Problem Statement:", "Integrated Problem Statement:"])
        methodology = self._extract_section(llm_response, ["Methodology:", "Proposed Methodology:", "Combined Methodology:"])
        validation = self._extract_section(llm_response, ["Experimental Validation:", "Comprehensive Experimental Validation:"])

        # Create merged idea
        return ResearchIdea(
            topic=topic or "Merged Research Idea",
            problem_statement=problem or llm_response[:500],
            proposed_methodology=methodology or "",
            experimental_validation=validation or "",
            source="idea_selector",
            method="llm_merged"
        )

    def _extract_section(self, text: str, headers: list) -> str:
        """Extract a section from LLM response by looking for headers"""
        for header in headers:
            if header in text:
                start_idx = text.find(header) + len(header)
                # Find next header or end of text
                remaining = text[start_idx:]

                # Look for next section header
                next_headers = ["Topic:", "Problem Statement:", "Methodology:", "Proposed Methodology:",
                               "Experimental Validation:", "Potential Impact:", "Enhanced Potential Impact:",
                               "Unified Topic:", "Integrated Problem Statement:", "Combined Methodology:",
                               "Comprehensive Experimental Validation:"]

                end_idx = len(remaining)
                for next_header in next_headers:
                    if next_header != header and next_header in remaining:
                        pos = remaining.find(next_header)
                        if pos < end_idx and pos > 0:
                            end_idx = pos

                return remaining[:end_idx].strip()

        return ""

    def _merge_ideas_simple(self, idea1: ResearchIdea, idea2: ResearchIdea) -> ResearchIdea:
        """Simple fallback merge without LLM"""
        # Create merged topic
        merged_topic = f"{idea1.topic.strip()} and {idea2.topic.strip()}"
        if len(merged_topic) > 150:
            merged_topic = f"{idea1.topic.strip()} (Enhanced)"

        # Merge facets using simple concatenation
        merged_facets = {}
        for key in ['Problem Statement', 'Proposed Methodology', 'Experimental Validation', 'Potential Impact']:
            text1 = idea1.facets.get(key, '')
            text2 = idea2.facets.get(key, '')
            merged_facets[key] = self._merge_text_sections(text1, text2)

        return ResearchIdea(
            topic=merged_topic,
            facets=merged_facets,
            source="idea_selector",
            method="simple_merged"
        )

    def _merge_text_sections(self, text1: str, text2: str) -> str:
        """Merge two text sections, avoiding redundancy"""
        if not text1:
            return text2
        if not text2:
            return text1

        # Split into sentences
        sentences1 = [s.strip() for s in text1.split('.') if s.strip()]
        sentences2 = [s.strip() for s in text2.split('.') if s.strip()]

        # Deduplicate
        all_sentences = sentences1.copy()
        threshold = self.selection_config['sentence_dedup_threshold']

        for sent in sentences2:
            is_unique = True
            words2 = set(sent.lower().split())
            for existing in sentences1:
                words1 = set(existing.lower().split())
                if words1 and words2:
                    overlap = len(words1 & words2) / len(words1 | words2)
                    if overlap > threshold:
                        is_unique = False
                        break
            if is_unique:
                all_sentences.append(sent)

        merged = '. '.join(all_sentences[:5])
        if merged and not merged.endswith('.'):
            merged += '.'
        return merged

    # ==================== Report Generation ====================

    def generate_selection_report(self, results: List[SimilarityResult]) -> Dict:
        """Generate comprehensive selection report"""
        if not results:
            return {'summary': {'total_analyzed': 0}}

        total = len(results)
        distinct = sum(1 for r in results if r.is_sufficiently_distinct)
        similarities = [r.max_similarity_score for r in results]

        # Categorize by overlap level
        overlap_dist = {
            'minimal_overlap': 0,
            'low_overlap': 0,
            'moderate_overlap': 0,
            'high_overlap': 0
        }
        for r in results:
            overlap_type = self._classify_overlap(r.max_similarity_score)
            overlap_dist[overlap_type] += 1

        report = {
            'summary': {
                'total_analyzed': total,
                'sufficiently_distinct': distinct,
                'distinctness_rate': distinct / total if total > 0 else 0,
                'average_similarity': float(np.mean(similarities)),
                'max_similarity': float(max(similarities)),
                'similarity_method': self.similarity_method
            },
            'overlap_distribution': overlap_dist,
            'detailed_results': [
                {
                    'item_id': r.item_id,
                    'max_similarity': r.max_similarity_score,
                    'is_distinct': r.is_sufficiently_distinct,
                    'most_similar': r.most_similar_item
                }
                for r in results
            ]
        }

        if self.logger:
            self.logger.log_info(
                f"Report: {distinct}/{total} distinct (rate={report['summary']['distinctness_rate']:.2f})",
                "idea_selector"
            )

        return report

    # ==================== Status and Information ====================

    def gather_information(self):
        """Get current agent status and configuration"""
        return {
            "component": "SelectionAgent",
            "similarity_method": self.similarity_method,
            "similarity_threshold": self.similarity_threshold,
            "status": "ready"
        }
