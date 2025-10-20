"""
Semantic Scholar API Agent

This agent searches for relevant academic papers using the Semantic Scholar API
to provide guidance for idea generation while preventing idea overlap.
"""

import asyncio
import aiohttp
import requests
import json
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
import time
import itertools
import random

from .base_agent import BaseAgent
from ..utils.llm_interface import LLMInterface
from ..utils.async_utils import limit_async_func_call, rate_limited, retry_with_timeout, AsyncRateLimiter
from ..prompts.literature_search.semantic_scholar_agent_prompts import (
    TOPIC_DECOMPOSITION_SYSTEM_PROMPT, TOPIC_DECOMPOSITION_USER_PROMPT
)


@dataclass
class Paper:
    """Represents a research paper from Semantic Scholar"""
    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    year: Optional[int]
    venue: Optional[str]
    citation_count: int
    url: str
    keywords: List[str]

    def to_dict(self) -> Dict:
        """Convert Paper object to dictionary for JSON serialization"""
        return {
            'paper_id': self.paper_id,
            'title': self.title,
            'abstract': self.abstract,
            'authors': self.authors,
            'year': self.year,
            'venue': self.venue,
            'citation_count': self.citation_count,
            'url': self.url,
            'keywords': self.keywords
        }

class SemanticScholarAgent(BaseAgent):
    """Agent for searching and retrieving papers from Semantic Scholar API"""
    
    def __init__(self, api_key: Optional[str] = None, config: Dict = None, logger=None, llm_config: Dict = None):
        super().__init__("SemanticScholarAgent")
        self.api_key = api_key or "1gJxp7gKYB28yVbJhWNtr2Ud0nj8IBgS6MosRMHh"
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.config = config or {}
        self.logger = logger
        self.rate_limit_delay = self.config.get('rate_limit_delay', 1.0)  # seconds
        self.time_window = self.config.get('time_window', 60.0)  # time window in seconds
        self.max_calls = self.config.get('max_calls', 10)  # max calls per time window
        self.max_papers = self.config.get('max_papers_per_search', 10)
        
        # Create configurable rate limiter instance
        self.rate_limiter = AsyncRateLimiter(self.max_calls, self.time_window)
        
        # Initialize LLM interface for intelligent query generation
        self.llm = LLMInterface(config=llm_config, logger=logger)
        
        # Combinatorial search configuration
        self.max_concepts = self.config.get('max_concepts', 8)  # Increased from 5 to 8 for richer concept extraction
        self.max_queries_per_layer = self.config.get('max_queries_per_layer', 3)  # Sample limit per layer
        self.topic_decomposition_tokens = self.config.get('topic_decomposition_tokens', 200)
        self.combination_order = self.config.get('combination_order', 'rev')
        if self.combination_order not in ('rev', 'fwd'):
            if self.logger:
                self.logger.log_warning(
                    f"Invalid combination_order '{self.combination_order}' provided; defaulting to 'rev'",
                    "semantic_scholar_agent",
                )
            self.combination_order = 'rev'
        
        # Paper scoring configuration
        scoring_config = self.config.get('scoring', {})
        self.relevance_weight = scoring_config.get('relevance_weight', 0.7)
        self.citation_weight = scoring_config.get('citation_weight', 0.3)
        self.citation_normalization = scoring_config.get('citation_normalization', 1000)
        
        # Validate scoring weights
        total_weight = self.relevance_weight + self.citation_weight
        if abs(total_weight - 1.0) > 0.01:  # Allow small floating point differences
            if self.logger:
                self.logger.log_warning(f"Scoring weights don't sum to 1.0: relevance={self.relevance_weight}, citation={self.citation_weight}, total={total_weight}", "semantic_scholar_agent")
        
        # Headers for API requests
        self.headers = {
            'x-api-key': self.api_key,
            'Content-Type': 'application/json'
        }
        
        if self.logger:
            self.logger.log_info(f"Initialized SemanticScholarAgent with API key: {self.api_key[:10]}... Rate limit: {self.max_calls} calls per {self.time_window}s", "semantic_scholar_agent")
    
    # @retry_with_timeout(max_retries=3, timeout=30, delay=1.0)
    async def search_papers(self, query: str, num_papers: int = 10, fields: List[str] = None, session: Optional[aiohttp.ClientSession] = None) -> List[Paper]:
        """
        Search for papers using Semantic Scholar API (async version)
        
        Args:
            query: Search query string
            num_papers: Number of papers to retrieve (default: 10)
            fields: List of fields to include in response
            session: Optional aiohttp session for connection reuse
            
        Returns:
            List of Paper objects
        """
        # Apply configurable rate limiting
        await self.rate_limiter.acquire()
        
        if fields is None:
            fields = [
                'paperId', 'title', 'abstract', 'authors', 'year', 
                'venue', 'citationCount', 'url', 'fieldsOfStudy'
            ]
        
        if self.logger:
            self.logger.log_info(f"Async searching papers for query: '{query}'", "semantic_scholar_agent")
        
        # Create session if not provided
        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True
        
        try:
            # Prepare search parameters
            params = {
                'query': query,
                'limit': min(num_papers, self.max_papers),
                'fields': ','.join(fields)
            }
            
            # Make async API request
            async with session.get(
                f"{self.base_url}/paper/search",
                headers=self.headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                # Handle rate limiting by raising an exception that retry decorator will catch
                if response.status == 429:  # Rate limited
                    if self.logger:
                        self.logger.log_warning("Rate limited - retry decorator will handle", "semantic_scholar_agent")
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=429,
                        message="Rate limited"
                    )
                
                response.raise_for_status()
                data = await response.json()
                
                if self.logger:
                    self.logger.log_info(f"Async API response: status={response.status}, data_keys={list(data.keys())}", "semantic_scholar_agent")
                
                # Check if 'data' key exists in response
                if 'data' not in data:
                    if self.logger:
                        self.logger.log_error(f"API response missing 'data' key. Full response: {data}", "semantic_scholar_agent")
                    return []
                
                papers = []
                paper_data_list = data.get('data', [])
                
                if self.logger:
                    self.logger.log_info(f"Raw papers in response: {len(paper_data_list)}", "semantic_scholar_agent")
                    
                # Also log total count if available
                if 'total' in data:
                    self.logger.log_info(f"Total papers available for query: {data['total']}", "semantic_scholar_agent")
                
                for paper_data in paper_data_list:
                    try:
                        paper = self._parse_paper_data(paper_data)
                        if paper:
                            papers.append(paper)
                    except Exception as e:
                        if self.logger:
                            self.logger.log_warning(f"Failed to parse paper data: {e}", "semantic_scholar_agent")
                        continue
                
                if self.logger:
                    self.logger.log_info(f"Successfully parsed {len(papers)} papers from {len(paper_data_list)} raw entries", "semantic_scholar_agent")
                
                return papers
                
        except aiohttp.ClientError as e:
            if self.logger:
                self.logger.log_error(f"Async API request failed: {e}", "semantic_scholar_agent", e)
            return []
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Unexpected error in async search_papers: {e}", "semantic_scholar_agent", e)
            return []
        finally:
            if close_session:
                await session.close()
    
    def search_papers_sync(self, query: str, num_papers: int = 10, fields: List[str] = None) -> List[Paper]:
        """
        Synchronous wrapper for search_papers (for backwards compatibility)
        """
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.search_papers(query, num_papers, fields))
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(self.search_papers(query, num_papers, fields))
    
    async def decompose_topic_into_concepts(self, topic: str) -> List[str]:
        """
        Use LLM to decompose a research topic into fundamental concepts
        
        Args:
            topic: Research topic to decompose
            
        Returns:
            List of fundamental concepts
        """
        if self.logger:
            self.logger.log_info(f"Decomposing topic '{topic}' into fundamental concepts", "semantic_scholar_agent")
        
        user_prompt = TOPIC_DECOMPOSITION_USER_PROMPT.format(topic=topic)

        try:
            response = await self.llm.generate_with_system_prompt(
                system_prompt=TOPIC_DECOMPOSITION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=self.topic_decomposition_tokens,
                temperature=0.3,
                caller="semantic_scholar_agent_decompose_topic"
            )
            
            # Parse the response to extract concepts
            concepts = [concept.strip().lower() for concept in response.split(',')]
            concepts = [c for c in concepts if c and len(c) > 2]  # Filter out short/empty concepts
            
            # Limit to max_concepts
            concepts = concepts[:self.max_concepts]
            
            if self.logger:
                self.logger.log_info(f"Decomposed '{topic}' into concepts: {concepts}", "semantic_scholar_agent")
            
            return concepts
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Failed to decompose topic: {e}", "semantic_scholar_agent", e)
            # Fallback to simple word extraction
            simple_concepts = topic.lower().split()[:self.max_concepts]
            return simple_concepts
    
    def generate_combinatorial_queries(self, concepts: List[str]) -> List[str]:
        """
        Generate combinatorial search queries from concepts
        
        Strategy (optimized):
        1. For n >= 10 concepts: Start from n//2 combinations
        2. For n < 10 concepts: Start from full combination (all concepts)  
        3. Work down to individual concepts
        4. Sample to prevent explosion
        
        This optimization reduces search space for large concept sets while
        maintaining comprehensive coverage for smaller sets.
        
        Args:
            concepts: List of fundamental concepts
            
        Returns:
            List of search queries ordered by comprehensiveness
        """
        if not concepts:
            return []
            
        if self.logger:
            self.logger.log_info(f"Generating combinatorial queries from {len(concepts)} concepts: {concepts}", "semantic_scholar_agent")
        
        all_queries = []
        
        # Determine starting combination size based on number of concepts
        n_concepts = len(concepts)
        if n_concepts >= 10:
            start_r = n_concepts // 2  # Start from n//2 for large concept sets
        else:
            start_r = n_concepts  # Start from n for small concept sets
        
        if self.combination_order == 'rev':
            range_iter = range(start_r, 0, -1)
            order_description = f"{start_r} down to 1"
        else:
            range_iter = range(1, start_r + 1)
            order_description = f"1 up to {start_r}"

        if self.logger:
            self.logger.log_info(
                f"Starting combinatorial search ({self.combination_order}) from {order_description} (total concepts: {n_concepts})",
                "semantic_scholar_agent",
            )
        
        # Generate combinations from configured order
        for r in range_iter:  # r = combination size
            combinations = list(itertools.combinations(concepts, r))
            
            # Convert combinations to search queries
            layer_queries = []
            for combo in combinations:
                query = " ".join(combo)
                layer_queries.append(query)
            
            # Sample to prevent explosion
            if len(layer_queries) > self.max_queries_per_layer:
                sampled_queries = random.sample(layer_queries, self.max_queries_per_layer)
                if self.logger:
                    self.logger.log_info(f"Sampled {len(sampled_queries)} from {len(layer_queries)} queries for {r}-concept combinations", "semantic_scholar_agent")
                layer_queries = sampled_queries
            
            all_queries.extend(layer_queries)
            
            if self.logger:
                self.logger.log_info(f"Generated {len(layer_queries)} queries for {r}-concept combinations", "semantic_scholar_agent")
        
        if self.logger:
            self.logger.log_info(f"Total combinatorial queries generated: {len(all_queries)}", "semantic_scholar_agent")
            self.logger.log_info(f"Example queries: {all_queries[:3]}", "semantic_scholar_agent")
        
        return all_queries
    
    async def search_papers_by_topic(self, topic: str, num_papers: int = 10) -> List[Paper]:
        """
        Search for papers related to a specific research topic using intelligent combinatorial search
        
        Args:
            topic: Research topic string
            num_papers: Number of papers to retrieve
            
        Returns:
            List of relevant Paper objects
        """
        if self.logger:
            self.logger.log_info(f"🔄 Starting iterative search for topic: '{topic}' (requesting {num_papers} papers)", "semantic_scholar_agent")

        # Step 1: Decompose topic into concepts
        concepts = await self.decompose_topic_into_concepts(topic)
        if not concepts:
            if self.logger:
                self.logger.log_warning("No concepts extracted, falling back to direct search", "semantic_scholar_agent")
            return await self.search_papers(topic, num_papers)

        # Step 2: Generate combinatorial queries
        all_search_queries = self.generate_combinatorial_queries(concepts)
        if not all_search_queries:
            if self.logger:
                self.logger.log_warning("No queries generated, falling back to direct search", "semantic_scholar_agent")
            return await self.search_papers(topic, num_papers)

        # Step 3: Iterative search with real-time deduplication
        unique_papers = []
        seen_ids = set()
        seen_titles = set()
        queries_used = 0
        max_queries_per_iteration = self.config.get('search_limits', {}).get('max_queries_per_iteration', 4)
        papers_per_query = self.config.get('search_limits', {}).get('papers_per_query', 10)

        if self.logger:
            self.logger.log_info(f"🔄 Starting with {len(all_search_queries)} total queries available", "semantic_scholar_agent")

        async with aiohttp.ClientSession() as session:
            iteration_count = 0
            while len(unique_papers) < num_papers and queries_used < len(all_search_queries):
                iteration_count += 1
                remaining_papers_needed = num_papers - len(unique_papers) + 5  # Buffer to account for duplicates
                queries_this_iteration = min(
                    max_queries_per_iteration,
                    len(all_search_queries) - queries_used,
                    max(1, (remaining_papers_needed // papers_per_query) + 1)
                )

                current_queries = all_search_queries[queries_used:queries_used + queries_this_iteration]

                if self.logger:
                    self.logger.log_info(f"🚀 Iteration {iteration_count}: Running {queries_this_iteration} queries (have {len(unique_papers)}/{num_papers} papers)", "semantic_scholar_agent")

                # Execute queries in parallel with rate limiting
                @limit_async_func_call(max_size=2, waitting_time=8)
                async def search_single_query(query: str) -> List[Paper]:
                    return await self.search_papers(query, papers_per_query, session=session)

                search_tasks = [search_single_query(query) for query in current_queries]
                results = await asyncio.gather(*search_tasks, return_exceptions=True)

                # Process and deduplicate results
                iteration_papers_added = 0
                for result in results:
                    if isinstance(result, Exception):
                        if self.logger:
                            self.logger.log_warning(f"Query failed: {result}", "semantic_scholar_agent")
                        continue

                    for paper in result:
                        if paper.paper_id in seen_ids:
                            continue
                        normalized_title = self._normalize_title(paper.title)
                        if normalized_title in seen_titles:
                            continue

                        seen_ids.add(paper.paper_id)
                        seen_titles.add(normalized_title)
                        unique_papers.append(paper)
                        iteration_papers_added += 1

                        if len(unique_papers) >= num_papers:
                            break
                    if len(unique_papers) >= num_papers:
                        break

                queries_used += queries_this_iteration

                if self.logger:
                    self.logger.log_info(f"✅ Iteration {iteration_count}: Added {iteration_papers_added} new papers (total: {len(unique_papers)}/{num_papers})", "semantic_scholar_agent")

                if iteration_papers_added == 0:
                    if self.logger:
                        self.logger.log_info("No new papers found, stopping search", "semantic_scholar_agent")
                    break

        # Step 4: Score and rank papers
        scored_papers = self._score_papers_by_relevance(unique_papers, concepts)
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        result = [paper for paper, score in scored_papers[:num_papers]]

        if self.logger:
            self.logger.log_info(f"🎯 Final result: {len(result)} papers (used {queries_used}/{len(all_search_queries)} queries)", "semantic_scholar_agent")

        return result
        
    def _score_papers_by_relevance(self, papers: List[Paper], concepts: List[str]) -> List[Tuple[Paper, float]]:
        """
        Score papers based on concept relevance and citation impact
        
        Args:
            papers: List of papers to score
            concepts: List of fundamental concepts for relevance scoring
            
        Returns:
            List of (paper, score) tuples
        """
        scored_papers = []
        
        if self.logger:
            self.logger.log_info(f"Scoring {len(papers)} papers with relevance_weight={self.relevance_weight}, citation_weight={self.citation_weight}, citation_norm={self.citation_normalization}", "semantic_scholar_agent")
        
        for paper in papers:
            # Calculate concept relevance score
            relevance_score = self._calculate_concept_relevance(paper, concepts)
            
            # Calculate citation impact score (normalized using configurable parameter)
            citation_score = min(1.0, (paper.citation_count or 0) / self.citation_normalization)
            
            # Combined score using configurable weights
            combined_score = self.relevance_weight * relevance_score + self.citation_weight * citation_score
            
            scored_papers.append((paper, combined_score))
        
        return scored_papers
    
    def _normalize_title(self, title: str) -> str:
        """
        Normalize paper title for deduplication by removing common variations
        
        Args:
            title: Original paper title
            
        Returns:
            Normalized title string
        """
        if not title:
            return ""
        
        # Convert to lowercase and remove common punctuation
        normalized = title.lower()
        
        # Remove common punctuation and extra whitespace  
        import re
        normalized = re.sub(r'[^\w\s]', ' ', normalized)  # Replace punctuation with space instead of removing
        normalized = re.sub(r'\s+', ' ', normalized)      # Normalize whitespace
        normalized = normalized.strip()
        
        # Only remove very common articles, keep more content words
        exclude_articles = self.config.get('title_normalization', {}).get('exclude_articles', ['a', 'an', 'the'])  # Reduced list - keep prepositions and conjunctions
        words = normalized.split()
        filtered_words = [word for word in words if word not in exclude_articles]
        
        return ' '.join(filtered_words)
    
    def _calculate_concept_relevance(self, paper: Paper, concepts: List[str]) -> float:
        """
        Calculate how relevant a paper is to the given concepts
        
        Args:
            paper: Paper to score
            concepts: List of fundamental concepts
            
        Returns:
            Relevance score between 0 and 1
        """
        # Combine paper text for analysis
        paper_text = " ".join([
            paper.title or "",
            paper.abstract or "",
            " ".join(paper.keywords) if paper.keywords else ""
        ]).lower()
        
        if not paper_text.strip():
            return 0.0
        
        # Count concept matches
        matches = 0
        for concept in concepts:
            if concept.lower() in paper_text:
                matches += 1
        
        # Calculate relevance as percentage of concepts found
        relevance = matches / len(concepts) if concepts else 0.0
        
        return min(1.0, relevance)  # Cap at 1.0
    
    def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """
        Get detailed information about a specific paper
        
        Args:
            paper_id: Semantic Scholar paper ID
            
        Returns:
            Paper object with detailed information
        """
        try:
            fields = [
                'paperId', 'title', 'abstract', 'authors', 'year',
                'venue', 'citationCount', 'url', 'fieldsOfStudy', 'references'
            ]
            
            params = {'fields': ','.join(fields)}
            
            response = requests.get(
                f"{self.base_url}/paper/{paper_id}",
                headers=self.headers,
                params=params,
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            return self._parse_paper_data(data)
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Failed to get paper details for {paper_id}: {e}", "semantic_scholar_agent", e)
            return None
    
    def extract_key_concepts(self, papers: List[Paper]) -> Set[str]:
        """
        Extract key concepts and keywords from a list of papers
        
        Args:
            papers: List of Paper objects
            
        Returns:
            Set of key concepts/keywords
        """
        concepts = set()
        
        for paper in papers:
            # Add keywords
            concepts.update(paper.keywords)
            
            # Extract key terms from titles and abstracts
            if paper.title:
                # Simple extraction of important terms from title
                title_terms = self._extract_technical_terms(paper.title)
                concepts.update(title_terms)
            
            if paper.abstract:
                # Extract terms from abstract (first configurable chars for efficiency)
                abstract_length = self.config.get('search_limits', {}).get('abstract_truncate_length', 200)
                abstract_terms = self._extract_technical_terms(paper.abstract[:abstract_length])
                concepts.update(abstract_terms)
        
        # Filter out common words and short terms
        filtered_concepts = {
            concept for concept in concepts 
            if len(concept) > 3 and concept.lower() not in {
                'method', 'approach', 'system', 'model', 'algorithm', 'study', 
                'analysis', 'research', 'paper', 'work', 'using', 'based'
            }
        }
        
        return filtered_concepts
    
    def _parse_paper_data(self, paper_data: Dict) -> Optional[Paper]:
        """Parse paper data from API response"""
        try:
            # Extract authors
            authors = []
            if paper_data.get('authors'):
                authors = [author.get('name', '') for author in paper_data['authors']]
            
            # Extract keywords from fieldsOfStudy
            keywords = []
            if paper_data.get('fieldsOfStudy'):
                keywords = [field for field in paper_data['fieldsOfStudy'] if field]
            
            return Paper(
                paper_id=paper_data.get('paperId', ''),
                title=paper_data.get('title', ''),
                abstract=paper_data.get('abstract', ''),
                authors=authors,
                year=paper_data.get('year'),
                venue=paper_data.get('venue'),
                citation_count=paper_data.get('citationCount', 0),
                url=paper_data.get('url', ''),
                keywords=keywords
            )
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Failed to parse paper data: {e}", "semantic_scholar_agent", e)
            return None
    
    def _extract_technical_terms(self, text: str) -> Set[str]:
        """Extract technical terms from text"""
        if not text:
            return set()
        
        # Simple extraction based on capitalization and length
        words = text.split()
        technical_terms = set()
        
        for word in words:
            # Clean word
            clean_word = ''.join(c for c in word if c.isalnum())
            
            # Keep terms that are:
            # - Capitalized (likely technical terms)
            # - Longer than 4 characters
            # - Not common words
            if (len(clean_word) > 4 and 
                (word[0].isupper() or any(c.isupper() for c in clean_word[1:]))):
                technical_terms.add(clean_word.lower())
        
        return technical_terms
    
    def gather_information(self):
        """Required method from BaseAgent - gather information about available papers"""
        if self.logger:
            self.logger.log_info("Gathering information about Semantic Scholar API status", "semantic_scholar_agent")
        
        try:
            # Test API connectivity using sync wrapper
            test_papers = self.search_papers_sync('test', 1)
            
            if test_papers is not None:  # Empty list is valid, None indicates error
                return {"status": "connected", "api_accessible": True, "test_papers_count": len(test_papers)}
            else:
                return {"status": "error", "api_accessible": False, "error": "Failed to retrieve test papers"}
                
        except Exception as e:
            return {"status": "error", "api_accessible": False, "error": str(e)}

    def generate_ideas(self):
        """Required method from BaseAgent - not used for this agent"""
        return []
