import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Tuple
from .debug_logger import init_debug_logger
from .async_utils import limit_async_func_call, retry_with_timeout, rate_limited
from .token_cost_tracker import TokenCostTracker
from ..prompts.interface_prompts import (
    ENTITY_EXTRACTION_SYSTEM_PROMPT, ENTITY_EXTRACTION_USER_PROMPT
)


class LLMInterface:
    def __init__(self, config: dict = None, logger=None):
        # Validate config is provided
        if not config:
            raise ValueError("Configuration is required and cannot be None")
        
        # Validate required fields
        if not config.get("api_key"):
            raise ValueError("API key is required in LLM configuration")
        if not config.get("model_name"):
            raise ValueError("Model name is required in LLM configuration")
        if not config.get("base_url"):
            raise ValueError("Base URL is required in LLM configuration")
        
        self.model_name = config["model_name"]
        self.api_key = config["api_key"]
        self.base_url = config["base_url"]
        self.default_max_tokens = config.get("max_tokens", 16384)  # Default context window
        self.default_temperature = config.get("temperature", 0.7)
        
        # Use unified max_tokens instead of per-task context windows
        self.max_tokens = config.get("max_tokens", 16384)
        
        # Async session will be initialized when needed
        self._session = None
        self.logger = logger or init_debug_logger()
        
        # Rate limiting configuration
        self.max_concurrent_calls = config.get("max_concurrent_calls", 5)
        self.rate_limit_calls = config.get("rate_limit_calls", 10)
        self.rate_limit_window = config.get("rate_limit_window", 60)  # seconds
        
        # Token cost tracking
        self.cost_tracker = TokenCostTracker(self.model_name, self.logger)
        
        # Don't log sensitive information
        self.logger.log_info(f"LLM Interface initialized - Model: {self.model_name}", "llm_interface")

    async def get_session(self):
        """Get or create async session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=120)  # 2 minute timeout
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )
        return self._session

    async def close_session(self):
        """Close the async session properly"""
        if self._session and not self._session.closed:
            try:
                # Give any pending operations a moment to complete
                await asyncio.sleep(0.05)
                await self._session.close()
                # Wait a brief moment for the connection to fully close
                await asyncio.sleep(0.05)
                self._session = None
                if self.logger:
                    self.logger.log_info("Async session closed successfully", "llm_interface")
            except Exception as e:
                if self.logger:
                    self.logger.log_warning(f"Error closing async session: {e}", "llm_interface")
                self._session = None

    def __del__(self):
        """Destructor to ensure session cleanup"""
        if self._session and not self._session.closed:
            import asyncio
            import warnings
            try:
                # Try to close gracefully if we're in an event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close_session())
                else:
                    asyncio.run(self.close_session())
            except:
                # If we can't close gracefully, suppress warnings and detach
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        # Detach the session to prevent cleanup warnings
                        if hasattr(self._session, '_connector'):
                            self._session._connector = None
                        self._session = None
                    except:
                        pass

    def get_context_window(self, task_type: str = "default") -> int:
        """Get the context window size (unified for all tasks)"""
        return self.max_tokens

    async def generate_text(self, prompt: str, max_tokens: int = None, temperature: float = None, task_type: str = "default") -> str:
        """Generate text using chat completions API"""
        messages = [{"role": "user", "content": prompt}]
        max_tokens = max_tokens or self.get_context_window(task_type)
        temperature = temperature or self.default_temperature
        response, _ = await self.chat_completion(messages, max_tokens, temperature, caller="general")
        return response

    @limit_async_func_call(max_size=8)
    @rate_limited(max_calls=20, time_window=60)
    @retry_with_timeout(max_retries=3, timeout=300, delay=2)
    async def chat_completion(
        self,
        messages: List[Dict],
        max_tokens: int = None,
        temperature: float = None,
        caller: str = "unknown"
    ) -> Tuple[str, Optional[Dict]]:
        """Generate response using OpenAI-compatible chat completions API and cost tracking"""
        try:
            max_tokens = max_tokens or self.default_max_tokens
            temperature = temperature or self.default_temperature
            self.logger.log_debug(f"Making async API call for {caller}", "llm_interface")
            
            session = await self.get_session()
            
            async with session.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            ) as response:
                response.raise_for_status()
                result = await response.json()
                response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # Track token usage and costs
                cost_info = self.cost_tracker.track_conversation(caller, messages, response_text)
                
                self.logger.log_debug(
                    f"Async API call successful for {caller} - Response length: {len(response_text)}, "
                    f"Tokens: {cost_info['tokens']['total_tokens']}, Cost: ${cost_info['costs_usd']['total_cost']:.6f}", 
                    "llm_interface"
                )
                return response_text, cost_info
                
        except Exception as e:
            self.logger.log_error(f"Async API call failed for {caller}: {e}", "llm_interface", e)
            return "", None

    async def generate_with_system_prompt(self, system_prompt: str, user_prompt: str, 
                                         max_tokens: int = None, temperature: float = None, 
                                         caller: str = "unknown", task_type: str = None) -> str:
        """Generate text with both system and user prompts"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Auto-detect task type from caller if not specified
        if not task_type:
            task_type = self._detect_task_type_from_caller(caller)
        
        max_tokens = max_tokens or self.get_context_window(task_type)
        temperature = temperature or self.default_temperature
        response, cost_info = await self.chat_completion(messages, max_tokens, temperature, caller)

        # Note: Cost tracking is already handled in chat_completion method

        # Log the full conversation
        self.logger.log_llm_conversation(
            agent_name=caller,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=response,
            metadata={
                "model": self.model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "async": True,
                "cost_info": cost_info
            }
        )

        return response

    def _detect_task_type_from_caller(self, caller: str) -> str:
        """Auto-detect task type from caller name"""
        caller_lower = caller.lower()
        
        if "idea_generation" in caller_lower or "ideagenerator" in caller_lower:
            return "idea_generation"
        elif "planning" in caller_lower:
            return "planning"  
        elif "faceted_decomposition" in caller_lower:
            return "faceted_decomposition"
        elif "gap_analysis" in caller_lower:
            return "gap_analysis"
        elif "kg_builder" in caller_lower or "extract_entities" in caller_lower:
            return "kg_extraction"
        elif "variant" in caller_lower:
            return "variant_generation"
        else:
            return "default"

    async def extract_entities_and_relationships(self, text: str) -> str:
        """Extract entities and relationships from research text for knowledge graph construction"""
        user_prompt = ENTITY_EXTRACTION_USER_PROMPT.format(text=text)
        
        return await self.generate_with_system_prompt(ENTITY_EXTRACTION_SYSTEM_PROMPT, user_prompt, caller="KGBuilder")


    async def get_model_info(self) -> dict:
        """Retrieve information about available models"""
        try:
            session = await self.get_session()
            async with session.get(f"{self.base_url}/models") as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            self.logger.log_error(f"Error fetching model info: {e}", "llm_interface", e)
            return {"error": str(e)}

    def get_session_cost_summary(self) -> Dict:
        """Get comprehensive cost summary for the current session"""
        return self.cost_tracker.get_session_summary()
    
    async def close_session(self):
        """Close the async session"""
        if self._session and not self._session.closed:
            try:
                # Log final cost summary before closing
                self.cost_tracker.log_session_summary()
                
                # Give any pending operations a moment to complete
                await asyncio.sleep(0.05)
                await self._session.close()
                # Wait a brief moment for the connection to fully close  
                await asyncio.sleep(0.05)
                
                self.logger.log_info("Async LLM session closed", "llm_interface")
            except Exception as e:
                self.logger.log_warning(f"Error closing LLM session: {e}", "llm_interface")
            finally:
                self._session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_session()
