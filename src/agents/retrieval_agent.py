"""
Retrieval Agent

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

from ..utils.llm_interface import LLMInterface
from ..utils.async_utils import limit_async_func_call, retry_with_timeout, AsyncRateLimiter
from ..skills.retrieval import (
    QUERY_GENERATION_SYSTEM_PROMPT, QUERY_GENERATION_USER_PROMPT
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

    @property
    def summary(self) -> str:
        """Get summary text for similarity comparison"""
        components = [
            self.title or "",
            self.abstract or "",
            " ".join(self.keywords) if self.keywords else ""
        ]
        return " ".join(comp for comp in components if comp).strip()

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

class RetrievalAgent:
    """Agent for searching and retrieving papers from Semantic Scholar API"""

    def __init__(self, api_key: Optional[str] = None, retrieval_config: Dict = None, logger=None, llm_config: Dict = None):
        self.retrieval_config = retrieval_config or {}
        self.api_key = api_key or self.retrieval_config['api_key']
        self.base_url = self.retrieval_config['base_url']
        self.logger = logger
        self.rate_limit_delay = self.retrieval_config['rate_limit_delay']
        self.time_window = self.retrieval_config['time_window']
        self.max_calls = self.retrieval_config['max_calls']
        self.max_papers = self.retrieval_config['max_papers_per_search']
        self.timeout = self.retrieval_config['timeout']

        # Create configurable rate limiter instance
        self.rate_limiter = AsyncRateLimiter(self.max_calls, self.time_window)

        # Initialize LLM interface for intelligent query generation
        self.llm = LLMInterface(config=llm_config, logger=logger)
        
        # Combinatorial search configuration (legacy fallback)
        self.max_concepts = self.retrieval_config['max_concepts']
        self.max_queries_per_layer = self.retrieval_config['max_queries_per_layer']
        self.combination_order = self.retrieval_config['combination_order']

        # Query generation configuration
        # Note: max_tokens now uses unified llm.max_tokens (16384) instead of per-task limits
        # Temperature removed - using LLM default
        if self.combination_order not in ('rev', 'fwd'):
            if self.logger:
                self.logger.log_warning(
                    f"Invalid combination_order '{self.combination_order}' provided; defaulting to 'rev'",
                    "semantic_scholar_agent",
                )
            self.combination_order = 'rev'
        
        # Paper scoring configuration
        scoring_config = self.retrieval_config['scoring']
        self.relevance_weight = scoring_config['relevance_weight']
        self.citation_weight = scoring_config['citation_weight']
        self.citation_normalization = scoring_config['citation_normalization']
        
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
            self.logger.log_info(f"Initialized RetrievalAgent with API key: {self.api_key[:10]}... Rate limit: {self.max_calls} calls per {self.time_window}s", "retrieval_agent")

    @retry_with_timeout()
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
                timeout=aiohttp.ClientTimeout(total=self.timeout)
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
    
    def _extract_simple_concepts(self, topic: str) -> List[str]:
        """
        Simple concept extraction from topic for fallback scenarios

        Args:
            topic: Research topic to extract concepts from

        Returns:
            List of simple concepts (words from topic)
        """
        # Simple word extraction as fallback
        concepts = [word.strip().lower() for word in topic.split() if len(word.strip()) > 2]
        concepts = concepts[:self.max_concepts]

        if self.logger:
            self.logger.log_info(f"Extracted simple concepts from topic: {concepts}", "semantic_scholar_agent")

        return concepts

    # UNUSED METHOD - Commented out for potential future use
    # def _parse_json_array_response(self, response: str):
    #     cleaned = response.strip()
    #     if cleaned.startswith("```json"):
    #         cleaned = cleaned[7:]
    #     if cleaned.startswith("```"):
    #         cleaned = cleaned[3:]
    #     if cleaned.endswith("```"):
    #         cleaned = cleaned[:-3]
    #     cleaned = cleaned.strip()
    #
    #     if not cleaned:
    #         raise ValueError("Empty response from LLM")
    #
    #     try:
    #         return json.loads(cleaned)
    #     except json.JSONDecodeError:
    #         start = cleaned.find("[")
    #         end = cleaned.rfind("]")
    #         if start != -1 and end != -1 and end > start:
    #             return json.loads(cleaned[start:end + 1])
    #         raise

    async def generate_queries_with_llm(self, topic: str, num_queries: int = 10, concepts: str = "") -> List[Dict[str, str]]:
        """
        Generate strategic search queries directly from the research topic.

        Args:
            topic: Research topic to generate queries for
            num_queries: Target number of queries to generate
            concepts: Optional comma-separated list of extracted concepts

        Returns:
            List of query dictionaries with 'query' field
        """
        if self.logger:
            self.logger.log_info(f"Generating queries for: '{topic}'", "semantic_scholar_agent")

        # Format the user prompt with topic and concepts
        user_prompt = QUERY_GENERATION_USER_PROMPT.format(
            topic=topic,
            concepts=concepts if concepts else "Not provided"
        )

        try:
            response = await self.llm.generate_with_system_prompt(
                system_prompt=QUERY_GENERATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                caller="semantic_scholar_agent_generate_queries"
            )

            # Parse comma-separated queries
            query_strings = [q.strip() for q in response.split(',') if q.strip()]

            # Convert to query dictionaries
            queries = [{'query': q} for q in query_strings[:num_queries]]

            if not queries:
                return self._generate_fallback_queries(topic)

            if self.logger:
                self.logger.log_info(f"Generated {len(queries)} queries", "semantic_scholar_agent")
                # Log first few queries for debugging
                for i, q in enumerate(queries[:3]):
                    self.logger.log_info(f"  Query {i+1}: {q['query']}", "semantic_scholar_agent")

            return queries

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error generating queries: {e}", "semantic_scholar_agent", e)
            return self._generate_fallback_queries(topic)

    def _generate_fallback_queries(self, topic: str) -> List[Dict[str, str]]:
        """
        Generate simple fallback queries when LLM generation fails

        Args:
            topic: Research topic

        Returns:
            List of basic query dictionaries
        """
        # Simple fallback: use the topic itself and variations
        words = topic.lower().split()

        fallback_queries = [
            {'query': topic, 'focus': 'direct', 'rationale': 'Direct topic search'},
        ]

        # Add partial queries if topic has multiple words
        if len(words) >= 3:
            fallback_queries.extend([
                {'query': ' '.join(words[:len(words)//2]), 'focus': 'broad', 'rationale': 'Broader search'},
                {'query': ' '.join(words[len(words)//2:]), 'focus': 'specific', 'rationale': 'Focused search'}
            ])

        if self.logger:
            self.logger.log_warning(f"Using fallback queries: {[q['query'] for q in fallback_queries]}", "semantic_scholar_agent")

        return fallback_queries

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
        Search for papers related to a specific research topic using LLM-generated queries

        Args:
            topic: Research topic string
            num_papers: Number of papers to retrieve

        Returns:
            List of relevant Paper objects
        """
        if self.logger:
            self.logger.log_info(f"🔄 Starting intelligent search for topic: '{topic}' (requesting {num_papers} papers)", "semantic_scholar_agent")

        # Step 1: Generate search queries using LLM
        use_llm_queries = self.retrieval_config['use_llm_query_generation']

        if use_llm_queries:
            default_num_queries = self.retrieval_config['default_num_queries']
            queries_data = await self.generate_queries_with_llm(topic, num_queries=default_num_queries)
            all_search_queries = [q['query'] for q in queries_data]

            # Extract keywords from topic for relevance scoring
            topic_keywords = topic.lower().split()
        else:
            # Fallback to old combinatorial approach
            if self.logger:
                self.logger.log_info("Using legacy combinatorial query generation", "semantic_scholar_agent")
            concepts = self._extract_simple_concepts(topic)
            if not concepts:
                if self.logger:
                    self.logger.log_warning("No concepts extracted, falling back to direct search", "semantic_scholar_agent")
                return await self.search_papers(topic, num_papers)
            all_search_queries = self.generate_combinatorial_queries(concepts)
            topic_keywords = concepts

        if not all_search_queries:
            if self.logger:
                self.logger.log_warning("No queries generated, falling back to direct search", "semantic_scholar_agent")
            return await self.search_papers(topic, num_papers)

        # Step 2: Iterative search with real-time deduplication
        unique_papers = []
        seen_ids = set()
        seen_titles = set()
        queries_used = 0
        max_queries_per_iteration = self.retrieval_config['search_limits']['max_queries_per_iteration']
        papers_per_query = self.retrieval_config['search_limits']['papers_per_query']

        if self.logger:
            self.logger.log_info(f"🔄 Starting with {len(all_search_queries)} queries available", "semantic_scholar_agent")

        async with aiohttp.ClientSession() as session:
            iteration_count = 0
            while len(unique_papers) < num_papers and queries_used < len(all_search_queries):
                iteration_count += 1
                paper_buffer = self.retrieval_config['search_limits']['paper_buffer_size']
                remaining_papers_needed = num_papers - len(unique_papers) + paper_buffer  # Buffer to account for duplicates
                queries_this_iteration = min(
                    max_queries_per_iteration,
                    len(all_search_queries) - queries_used,
                    max(1, (remaining_papers_needed // papers_per_query) + 1)
                )

                current_queries = all_search_queries[queries_used:queries_used + queries_this_iteration]

                if self.logger:
                    self.logger.log_info(f"🚀 Iteration {iteration_count}: Running {queries_this_iteration} queries (have {len(unique_papers)}/{num_papers} papers)", "semantic_scholar_agent")
                    self.logger.log_info(f"   Queries: {current_queries}", "semantic_scholar_agent")

                # Execute queries in parallel with rate limiting
                parallel_config = self.retrieval_config['parallel_search']
                max_concurrent = parallel_config['max_concurrent']

                @limit_async_func_call(max_size=max_concurrent)
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

        # Step 3: Score and rank papers
        scored_papers = self._score_papers_by_relevance(unique_papers, topic_keywords)
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        result = [paper for paper, score in scored_papers[:num_papers]]

        if self.logger:
            self.logger.log_info(f"🎯 Final result: {len(result)} papers (used {queries_used}/{len(all_search_queries)} queries)", "semantic_scholar_agent")

        return result
        
    def _score_papers_by_relevance(self, papers: List[Paper], keywords: List[str]) -> List[Tuple[Paper, float]]:
        """
        Score papers based on keyword relevance and citation impact

        Args:
            papers: List of papers to score
            keywords: List of keywords or concepts for relevance scoring

        Returns:
            List of (paper, score) tuples
        """
        scored_papers = []

        if self.logger:
            self.logger.log_info(f"Scoring {len(papers)} papers with relevance_weight={self.relevance_weight}, citation_weight={self.citation_weight}, citation_norm={self.citation_normalization}", "semantic_scholar_agent")

        for paper in papers:
            # Calculate keyword relevance score
            relevance_score = self._calculate_concept_relevance(paper, keywords)

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
        exclude_articles = self.retrieval_config['title_normalization']['exclude_articles']
        words = normalized.split()
        filtered_words = [word for word in words if word not in exclude_articles]
        
        return ' '.join(filtered_words)
    
    def _calculate_concept_relevance(self, paper: Paper, keywords: List[str]) -> float:
        """
        Calculate how relevant a paper is to the given keywords/concepts

        Uses weighted scoring based on where keywords appear:
        - Title matches: highest weight
        - Abstract matches: medium weight
        - Keywords field matches: lower weight
        - Partial word matches: reduced score

        Args:
            paper: Paper to score
            keywords: List of keywords or concepts to match

        Returns:
            Relevance score between 0 and 1
        """
        if not keywords:
            return 0.0

        # Prepare text fields
        title_text = (paper.title or "").lower()
        abstract_text = (paper.abstract or "").lower()
        keywords_text = " ".join(paper.keywords).lower() if paper.keywords else ""

        # Scoring weights for different fields (from config)
        scoring_config = self.retrieval_config['relevance_scoring']
        title_weight = scoring_config['title_weight']
        abstract_weight = scoring_config['abstract_weight']
        keywords_weight = scoring_config['keywords_weight']

        total_score = 0.0
        max_possible_score = 0.0

        for keyword in keywords:
            keyword_lower = keyword.lower()
            keyword_score = 0.0

            # Exact match in title (highest value)
            if keyword_lower in title_text:
                keyword_score += title_weight

            # Match in abstract
            if keyword_lower in abstract_text:
                keyword_score += abstract_weight

            # Match in keywords field
            if keyword_lower in keywords_text:
                keyword_score += keywords_weight

            # Partial word matching (for multi-word keywords)
            if len(keyword_lower.split()) > 1:
                # For multi-word keywords, check if all words appear
                words = keyword_lower.split()
                words_in_title = sum(1 for w in words if w in title_text)
                words_in_abstract = sum(1 for w in words if w in abstract_text)

                # Partial credit for partial matches (using config weight)
                partial_weight = scoring_config['partial_match_weight']
                if words_in_title > 0:
                    keyword_score += (words_in_title / len(words)) * title_weight * partial_weight
                if words_in_abstract > 0:
                    keyword_score += (words_in_abstract / len(words)) * abstract_weight * partial_weight

            total_score += keyword_score
            # Maximum possible score per keyword is if it appears in all fields
            max_possible_score += (title_weight + abstract_weight + keywords_weight)

        # Normalize to 0-1 range
        if max_possible_score > 0:
            relevance = total_score / max_possible_score
        else:
            relevance = 0.0

        return min(1.0, relevance)  # Cap at 1.0

    # UNUSED METHOD - Commented out for potential future use
    # def get_paper_details(self, paper_id: str) -> Optional[Paper]:
    #     """
    #     Get detailed information about a specific paper
    #
    #     Args:
    #         paper_id: Semantic Scholar paper ID
    #
    #     Returns:
    #         Paper object with detailed information
    #     """
    #     try:
    #         fields = [
    #             'paperId', 'title', 'abstract', 'authors', 'year',
    #             'venue', 'citationCount', 'url', 'fieldsOfStudy', 'references'
    #         ]
    #
    #         params = {'fields': ','.join(fields)}
    #
    #         response = requests.get(
    #             f"{self.base_url}/paper/{paper_id}",
    #             headers=self.headers,
    #             params=params,
    #             timeout=self.timeout
    #         )
    #
    #         response.raise_for_status()
    #         data = response.json()
    #
    #         return self._parse_paper_data(data)
    #
    #     except Exception as e:
    #         if self.logger:
    #             self.logger.log_error(f"Failed to get paper details for {paper_id}: {e}", "semantic_scholar_agent", e)
    #         return None


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

    # UNUSED METHOD - Commented out for potential future use
    # def _extract_technical_terms(self, text: str) -> Set[str]:
    #     """Extract technical terms from text"""
    #     if not text:
    #         return set()
    #
    #     # Simple extraction based on capitalization and length
    #     words = text.split()
    #     technical_terms = set()
    #
    #     for word in words:
    #         # Clean word
    #         clean_word = ''.join(c for c in word if c.isalnum())
    #
    #         # Keep terms that are:
    #         # - Capitalized (likely technical terms)
    #         # - Longer than 4 characters
    #         # - Not common words
    #         if (len(clean_word) > 4 and
    #             (word[0].isupper() or any(c.isupper() for c in clean_word[1:]))):
    #             technical_terms.add(clean_word.lower())
    #
    #     return technical_terms

    # UNUSED METHOD - Commented out for potential future use
    # def gather_information(self):
    #     """Required method from BaseAgent - gather information about available papers"""
    #     if self.logger:
    #         self.logger.log_info("Gathering information about Semantic Scholar API status", "semantic_scholar_agent")
    #
    #     try:
    #         # Test API connectivity using sync wrapper
    #         test_papers = self.search_papers_sync('test', 1)
    #
    #         if test_papers is not None:  # Empty list is valid, None indicates error
    #             return {"status": "connected", "api_accessible": True, "test_papers_count": len(test_papers)}
    #         else:
    #             return {"status": "error", "api_accessible": False, "error": "Failed to retrieve test papers"}
    #
    #     except Exception as e:
    #         return {"status": "error", "api_accessible": False, "error": str(e)}
