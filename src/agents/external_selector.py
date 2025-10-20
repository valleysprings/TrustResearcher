"""
External Selector

This agent selects research ideas based on their distinctness from existing literature.
It analyzes similarity between generated ideas and published papers to ensure novelty
and filters out ideas that are too similar to already published work.
"""

import json
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer

from .base_agent import BaseAgent
from .semantic_scholar_agent import Paper
from .idea_generator import ResearchIdea
from ..utils.text_utils import extract_content_from_idea


@dataclass
class SimilarityResult:
    """Result of similarity analysis between an idea and papers"""
    idea_topic: str
    max_similarity_score: float
    most_similar_paper: str
    similarity_details: List[Dict]
    is_sufficiently_distinct: bool
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convert SimilarityResult to JSON-serializable dictionary"""
        return {
            'idea_topic': self.idea_topic,
            'max_similarity_score': self.max_similarity_score,
            'most_similar_paper': self.most_similar_paper,
            'similarity_details': self.similarity_details,
            'is_sufficiently_distinct': self.is_sufficiently_distinct,
            'recommendations': self.recommendations
        }


class ExternalSelector(BaseAgent):
    """
    Agent that filters generated ideas by comparing them against existing literature
    """
    
    def __init__(self, config: Dict = None, logger=None):
        super().__init__("LiteratureSimilarityAgent")
        self.config = config or {}
        self.logger = logger
        
        # Thresholds for similarity analysis
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        self.high_similarity_threshold = self.config.get('high_similarity_threshold', 0.8)
        self.low_similarity_threshold = self.config.get('low_similarity_threshold', 0.3)
        
        # Similarity method configuration
        self.similarity_method = self.config.get('similarity_method', 'bge')  # 'tfidf', 'bge', 'stella', 'jina'

        # TF-IDF vectorizer for text similarity - use config parameters
        tfidf_config = self.config.get('tfidf', {})
        self.vectorizer = TfidfVectorizer(
            max_features=tfidf_config.get('max_features', 1000),
            stop_words=tfidf_config.get('stop_words', 'english'),
            ngram_range=tuple(tfidf_config.get('ngram_range', [1, 2])),
            min_df=tfidf_config.get('min_df', 1),
            max_df=tfidf_config.get('max_df', 0.95)
        )
        
        # Initialize embedding models lazily
        self._embedding_models = {}
        
        if self.logger:
            self.logger.log_info(f"Initialized DistractorModel with {self.similarity_method} similarity", "distractor_model")
    
    def analyze_idea_distinctness(self, idea: ResearchIdea, papers: List[Paper]) -> SimilarityResult:
        """
        Analyze how distinct a generated idea is from existing papers
        
        Args:
            idea: Generated research idea
            papers: List of relevant papers from literature
            
        Returns:
            SimilarityResult with analysis details
        """
        if self.logger:
            self.logger.log_info(f"Analyzing distinctness for idea: '{idea.topic}'", "distractor_model")
        
        if not papers:
            return SimilarityResult(
                idea_topic=idea.topic,
                max_similarity_score=0.0,
                most_similar_paper="No papers to compare",
                similarity_details=[],
                is_sufficiently_distinct=True,
                recommendations=["No existing papers found for comparison"]
            )
        
        # Prepare texts for comparison
        idea_text = self._prepare_idea_text(idea)
        paper_texts = [self._prepare_paper_text(paper) for paper in papers]
        
        # Calculate similarities
        similarities = self._calculate_text_similarities(idea_text, paper_texts)
        
        # Find most similar paper
        max_sim_idx = np.argmax(similarities)
        max_similarity = similarities[max_sim_idx]
        most_similar_paper = papers[max_sim_idx].title
        
        # Create detailed similarity results
        similarity_details = []
        for i, (paper, sim_score) in enumerate(zip(papers, similarities)):
            if sim_score > self.low_similarity_threshold:  # Only include notable similarities
                similarity_details.append({
                    'paper_title': paper.title,
                    'paper_id': paper.paper_id,
                    'similarity_score': float(sim_score),
                    'year': int(paper.year) if paper.year else None,
                    'citation_count': int(paper.citation_count) if paper.citation_count else 0,
                    'overlap_type': self._classify_overlap_type(float(sim_score))
                })
        
        # Sort by similarity score
        similarity_details.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Determine if idea is sufficiently distinct
        is_distinct = bool(max_similarity < self.similarity_threshold)
        
        # Generate recommendations
        recommendations = self._generate_distinctness_recommendations(
            idea, max_similarity, similarity_details
        )
        
        result = SimilarityResult(
            idea_topic=idea.topic,
            max_similarity_score=float(max_similarity),
            most_similar_paper=most_similar_paper,
            similarity_details=similarity_details[:self.config.get('max_similarity_details', 5)],  # Top N most similar
            is_sufficiently_distinct=is_distinct,
            recommendations=recommendations
        )
        
        if self.logger:
            self.logger.log_info(
                f"Distinctness analysis complete. Max similarity: {max_similarity:.3f}, "
                f"Distinct: {is_distinct}", "distractor_model"
            )
        
        return result
    
    def batch_analyze_distinctness(self, ideas: List[ResearchIdea], papers: List[Paper]) -> List[SimilarityResult]:
        """
        Analyze distinctness for multiple ideas at once
        
        Args:
            ideas: List of generated research ideas
            papers: List of relevant papers from literature
            
        Returns:
            List of SimilarityResult objects
        """
        if self.logger:
            self.logger.log_info(f"Batch analyzing {len(ideas)} ideas against {len(papers)} papers", "distractor_model")
        
        results = []
        for idea in ideas:
            result = self.analyze_idea_distinctness(idea, papers)
            results.append(result)
        
        # Log summary
        distinct_count = sum(1 for r in results if r.is_sufficiently_distinct)
        if self.logger:
            self.logger.log_info(
                f"Batch analysis complete. {distinct_count}/{len(ideas)} ideas are sufficiently distinct",
                "distractor_model"
            )
        
        return results
    
    def filter_distinct_ideas(self, ideas: List[ResearchIdea], papers: List[Paper]) -> Tuple[List[ResearchIdea], List[SimilarityResult]]:
        """
        Filter ideas to keep only those that are sufficiently distinct from existing papers
        
        Args:
            ideas: List of generated research ideas
            papers: List of relevant papers from literature
            
        Returns:
            Tuple of (filtered_ideas, analysis_results)
        """
        analysis_results = self.batch_analyze_distinctness(ideas, papers)
        
        filtered_ideas = []
        for idea, result in zip(ideas, analysis_results):
            if result.is_sufficiently_distinct:
                filtered_ideas.append(idea)
        
        if self.logger:
            self.logger.log_info(
                f"Filtered {len(ideas)} ideas down to {len(filtered_ideas)} distinct ideas",
                "distractor_model"
            )
        
        return filtered_ideas, analysis_results
    
    def _prepare_idea_text(self, idea: ResearchIdea) -> str:
        """Prepare idea text for similarity analysis"""
        return extract_content_from_idea(idea)
    
    def _prepare_paper_text(self, paper: Paper) -> str:
        """Prepare paper text for similarity analysis"""
        components = [
            paper.title or "",
            paper.abstract or "",
            " ".join(paper.keywords) if paper.keywords else ""
        ]
        return " ".join(comp for comp in components if comp).strip()
    
    def _get_embedding_model(self, model_name: str) -> SentenceTransformer:
        """Get embedding model, loading it lazily"""
        if model_name not in self._embedding_models:
            try:
                if model_name == 'bge':
                    # Try BGE-M3 first, fallback to smaller model if timeout/error
                    try:
                        model_id = 'BAAI/bge-m3'
                        if self.logger:
                            self.logger.log_info(f"Loading embedding model: {model_id}", "distractor_model")
                        self._embedding_models[model_name] = SentenceTransformer(model_id)
                    except:
                        # Fallback to smaller model for convenience 
                        if self.logger:
                            self.logger.log_info("BGE-M3 loading failed, using all-MiniLM-L6-v2 fallback", "distractor_model")
                        self._embedding_models[model_name] = SentenceTransformer('all-MiniLM-L6-v2')
                elif model_name == 'stella':
                    model_id = 'dunzhang/stella_en_1.5B_v5'
                elif model_name == 'jina':
                    model_id = 'jinaai/jina-embeddings-v2-base-en'
                else:
                    raise ValueError(f"Unknown embedding model: {model_name}")
                
                if model_name != 'bge':  # BGE already handled above
                    if self.logger:
                        self.logger.log_info(f"Loading embedding model: {model_id}", "distractor_model")
                    self._embedding_models[model_name] = SentenceTransformer(model_id)
                
            except Exception as e:
                if self.logger:
                    self.logger.log_error(f"Failed to load embedding model {model_name}: {e}", "distractor_model", e)
                # Fallback to a smaller model
                self._embedding_models[model_name] = SentenceTransformer('all-MiniLM-L6-v2')
        
        return self._embedding_models[model_name]
    
    def _calculate_text_similarities(self, idea_text: str, paper_texts: List[str]) -> np.ndarray:
        """Calculate similarities between idea and papers using configured method"""
        try:
            if self.similarity_method == 'tfidf':
                return self._calculate_tfidf_similarities(idea_text, paper_texts)
            elif self.similarity_method in ['bge', 'stella', 'jina']:
                return self._calculate_embedding_similarities(idea_text, paper_texts, self.similarity_method)
            else:
                if self.logger:
                    self.logger.log_warning(f"Unknown similarity method: {self.similarity_method}, falling back to TF-IDF", "distractor_model")
                return self._calculate_tfidf_similarities(idea_text, paper_texts)
                
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error calculating similarities: {e}", "distractor_model", e)
            # Return zero similarities if calculation fails
            return np.zeros(len(paper_texts))
    
    def _calculate_tfidf_similarities(self, idea_text: str, paper_texts: List[str]) -> np.ndarray:
        """Calculate TF-IDF cosine similarities between idea and papers"""
        all_texts = [idea_text] + paper_texts
        
        # Fit TF-IDF vectorizer and transform texts
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # Calculate cosine similarity between idea (first text) and all papers
        idea_vector = tfidf_matrix[0:1]  # First row (idea)
        paper_vectors = tfidf_matrix[1:]  # Remaining rows (papers)
        
        similarities = cosine_similarity(idea_vector, paper_vectors)[0]
        
        return similarities
    
    def _calculate_embedding_similarities(self, idea_text: str, paper_texts: List[str], model_name: str) -> np.ndarray:
        """Calculate embedding-based cosine similarities between idea and papers"""
        model = self._get_embedding_model(model_name)
        
        # Get embeddings for all texts
        all_texts = [idea_text] + paper_texts
        embeddings = model.encode(all_texts, normalize_embeddings=True)
        
        # Calculate cosine similarity between idea (first embedding) and all papers
        idea_embedding = embeddings[0:1]  # First embedding (idea)
        paper_embeddings = embeddings[1:]  # Remaining embeddings (papers)
        
        similarities = cosine_similarity(idea_embedding, paper_embeddings)[0]
        
        return similarities
    
    def _classify_overlap_type(self, similarity_score: float) -> str:
        """Classify the type of overlap based on similarity score"""
        if similarity_score >= self.high_similarity_threshold:
            return "high_overlap"
        elif similarity_score >= self.similarity_threshold:
            return "moderate_overlap"
        elif similarity_score >= self.low_similarity_threshold:
            return "low_overlap"
        else:
            return "minimal_overlap"
    
    def _generate_distinctness_recommendations(self, idea: ResearchIdea, max_similarity: float, 
                                            similarity_details: List[Dict]) -> List[str]:
        """Generate recommendations to improve idea distinctness"""
        recommendations = []
        
        if max_similarity >= self.high_similarity_threshold:
            recommendations.extend([
                "CRITICAL: Idea shows very high similarity to existing work",
                "Consider focusing on a different aspect or approach",
                "Identify unique angles or methodological innovations",
                "Review the most similar papers and find gaps to address"
            ])
        elif max_similarity >= self.similarity_threshold:
            recommendations.extend([
                "Idea shows moderate similarity to existing work",
                "Consider emphasizing novel aspects more clearly",
                "Differentiate methodology from existing approaches",
                "Focus on unique contributions and innovations"
            ])
        elif max_similarity >= self.low_similarity_threshold:
            recommendations.extend([
                "Good distinctness from existing work",
                "Minor overlaps detected - consider highlighting differences",
                "Ensure methodology clearly differentiates from similar work"
            ])
        else:
            recommendations.extend([
                "Excellent distinctness from existing literature",
                "Idea appears to be novel and unique",
                "Consider if the novelty is too radical (might need grounding)"
            ])
        
        # Add specific recommendations based on high-similarity papers
        if similarity_details:
            top_similar = similarity_details[0]
            if top_similar['similarity_score'] > self.similarity_threshold:
                recommendations.append(
                    f"Review '{top_similar['paper_title']}' to understand overlap and differentiate"
                )
        
        return recommendations
    
    def generate_distinctness_report(self, results: List[SimilarityResult]) -> Dict:
        """Generate a comprehensive distinctness report for a set of ideas"""
        total_ideas = len(results)
        distinct_ideas = sum(1 for r in results if r.is_sufficiently_distinct)
        
        # Calculate similarity statistics
        similarities = [r.max_similarity_score for r in results]
        avg_similarity = np.mean(similarities) if similarities else 0
        max_similarity = max(similarities) if similarities else 0
        
        # Categorize ideas by overlap level
        overlap_categories = {
            'minimal_overlap': 0,
            'low_overlap': 0,
            'moderate_overlap': 0,
            'high_overlap': 0
        }
        
        for result in results:
            overlap_type = self._classify_overlap_type(result.max_similarity_score)
            overlap_categories[overlap_type] += 1
        
        report = {
            'summary': {
                'total_ideas_analyzed': int(total_ideas),
                'sufficiently_distinct_ideas': int(distinct_ideas),
                'distinctness_rate': float(distinct_ideas / total_ideas if total_ideas > 0 else 0),
                'average_similarity_score': float(avg_similarity),
                'maximum_similarity_score': float(max_similarity),
                'similarity_model': self.similarity_method,
                'model_details': self._get_model_details()
            },
            'overlap_distribution': overlap_categories,
            'recommendations': self._generate_portfolio_recommendations(results),
            'detailed_results': [
                {
                    'idea_topic': r.idea_topic,
                    'max_similarity': float(r.max_similarity_score),
                    'is_distinct': bool(r.is_sufficiently_distinct),
                    'top_overlap': r.most_similar_paper
                }
                for r in results
            ]
        }
        
        if self.logger:
            self.logger.log_info(
                f"Generated distinctness report: {distinct_ideas}/{total_ideas} ideas are distinct",
                "distractor_model"
            )
        
        return report
    
    def _generate_portfolio_recommendations(self, results: List[SimilarityResult]) -> List[str]:
        """Generate recommendations for the entire portfolio of ideas"""
        total = len(results)
        distinct = sum(1 for r in results if r.is_sufficiently_distinct)

        # Get threshold ratios from config
        excellent_ratio = self.config.get('distinctness_excellent_ratio', 0.8)
        moderate_ratio = self.config.get('distinctness_moderate_ratio', 0.5)

        recommendations = []

        if distinct == total:
            recommendations.append("Excellent portfolio distinctness - all ideas are sufficiently novel")
        elif distinct >= total * excellent_ratio:
            recommendations.append("Good portfolio distinctness - most ideas are novel")
            recommendations.append("Consider refining the few similar ideas further")
        elif distinct >= total * moderate_ratio:
            recommendations.append("Moderate portfolio distinctness - about half the ideas are novel")
            recommendations.append("Focus on improving distinctness of similar ideas")
            recommendations.append("Consider generating additional novel ideas to replace similar ones")
        else:
            recommendations.append("Low portfolio distinctness - many ideas overlap with existing work")
            recommendations.append("Major revision needed to improve novelty")
            recommendations.append("Consider different research angles or methodological approaches")

        return recommendations

    def _get_model_details(self) -> Dict[str, str]:
        """Get details about the similarity model being used"""
        if self.similarity_method == 'tfidf':
            return {
                'model_type': 'TF-IDF Vectorizer',
                'model_id': 'sklearn.feature_extraction.text.TfidfVectorizer',
                'description': 'TF-IDF with cosine similarity (range: 0-1, no negative values due to non-negative term weights)'
            }
        elif self.similarity_method == 'bge':
            # Check which BGE model is actually loaded
            if 'bge' in self._embedding_models:
                try:
                    model_name = getattr(self._embedding_models['bge'], 'model_name', 'BAAI/bge-m3')
                    if 'bge-m3' in model_name.lower():
                        return {
                            'model_type': 'BGE-M3 Embedding Model',
                            'model_id': 'BAAI/bge-m3',
                            'description': 'Multilingual embedding model with strong semantic understanding'
                        }
                    else:
                        return {
                            'model_type': 'MiniLM Embedding Model (BGE fallback)',
                            'model_id': 'all-MiniLM-L6-v2',
                            'description': 'Lightweight embedding model (fallback from BGE-M3)'
                        }
                except:
                    return {
                        'model_type': 'BGE Embedding Model',
                        'model_id': 'BAAI/bge-m3',
                        'description': 'BGE embedding model for semantic similarity'
                    }
            else:
                return {
                    'model_type': 'BGE Embedding Model',
                    'model_id': 'BAAI/bge-m3',
                    'description': 'BGE embedding model for semantic similarity'
                }
        elif self.similarity_method == 'stella':
            return {
                'model_type': 'Stella Embedding Model',
                'model_id': 'dunzhang/stella_en_1.5B_v5',
                'description': 'High-performance English embedding model'
            }
        elif self.similarity_method == 'jina':
            return {
                'model_type': 'Jina Embedding Model',
                'model_id': 'jinaai/jina-embeddings-v2-base-en',
                'description': 'Jina AI embedding model for semantic similarity'
            }
        else:
            return {
                'model_type': 'Unknown',
                'model_id': self.similarity_method,
                'description': f'Similarity method: {self.similarity_method}'
            }
    
    def gather_information(self):
        """Required method from BaseAgent"""
        return {
            "component": "DistractorModel",
            "similarity_threshold": self.similarity_threshold,
            "status": "ready"
        }
    
    def generate_ideas(self):
        """Required method from BaseAgent - not used for this agent"""
        return []