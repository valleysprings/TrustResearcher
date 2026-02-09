import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Tuple
from .debug_logger import init_debug_logger
from .async_utils import limit_async_func_call
from .token_cost_tracker import TokenCostTracker
from ..skills.ideagen.graphop import (
    ENTITY_EXTRACTION_SYSTEM_PROMPT, ENTITY_EXTRACTION_USER_PROMPT
)


class LLMInterface:
    def __init__(self, config: dict = None, logger=None):
        # Validate config is provided
        if not config:
            raise ValueError("Configuration is required and cannot be None")

        # Validate required fields
        required_fields = [
            "api_key", "model_name", "base_url", "max_tokens", "temperature",
            "max_concurrent_calls", "rate_limit_calls", "rate_limit_window"
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"{field} is required in LLM configuration")

        # Validate timeouts configuration
        if "timeouts" not in config:
            raise ValueError("timeouts configuration is required")
        if "session_timeout" not in config["timeouts"]:
            raise ValueError("timeouts.session_timeout is required in LLM configuration")

        self.model_name = config["model_name"]
        self.api_key = config["api_key"]
        self.base_url = config["base_url"]
        self.default_max_tokens = config["max_tokens"]
        self.default_temperature = config["temperature"]

        # Use unified max_tokens instead of per-task context windows
        self.max_tokens = config["max_tokens"]

        # Async session will be initialized when needed
        self._session = None
        self.logger = logger or init_debug_logger()

        # Rate limiting configuration - all from config file
        self.max_concurrent_calls = config["max_concurrent_calls"]
        self.rate_limit_calls = config["rate_limit_calls"]
        self.rate_limit_window = config["rate_limit_window"]

        # Timeout configuration
        self.session_timeout = config["timeouts"]["session_timeout"]

        # Retry configuration - read from config
        if "retry" not in config:
            raise ValueError("retry configuration is required in LLM configuration")
        self.max_retries = config["retry"]["max_retries"]
        self.retry_delay = config["retry"]["delay"]
        self.retry_timeout = config["retry"]["timeout"]

        # Extract custom pricing if available in config
        custom_pricing = None
        if "token_cost" in config and "custom_pricing" in config["token_cost"]:
            custom_pricing = config["token_cost"]["custom_pricing"]

        # Token cost tracking
        self.cost_tracker = TokenCostTracker(self.model_name, self.logger, custom_pricing=custom_pricing)

        # Don't log sensitive information
        self.logger.log_info(f"LLM Interface initialized - Model: {self.model_name}", "llm_interface")

    async def get_session(self):
        """Get or create async session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.session_timeout)
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

    def _extract_response_text(self, result: Dict) -> str:
        """Extract content text from various OpenAI-compatible or Gemini-like responses"""
        if not isinstance(result, dict):
            return ""

        if isinstance(result.get("data"), dict):
            return self._extract_response_text(result["data"])

        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0] or {}
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
                if isinstance(choice.get("text"), str):
                    return choice["text"]

        candidates = result.get("candidates")
        if isinstance(candidates, list) and candidates:
            candidate = candidates[0] or {}
            if isinstance(candidate, dict):
                content = candidate.get("content")
                if isinstance(content, dict):
                    parts = content.get("parts")
                    if isinstance(parts, list):
                        text_parts = []
                        for part in parts:
                            if isinstance(part, dict) and isinstance(part.get("text"), str):
                                text_parts.append(part["text"])
                            elif isinstance(part, str):
                                text_parts.append(part)
                        joined = "".join(text_parts).strip()
                        if joined:
                            return joined
                for key in ("text", "output", "content"):
                    if isinstance(candidate.get(key), str):
                        return candidate[key]

        for key in ("output_text", "text", "content"):
            if isinstance(result.get(key), str):
                return result[key]

        return ""

    @limit_async_func_call(max_size=8)
    async def chat_completion(
        self,
        messages: List[Dict],
        max_tokens: int = None,
        temperature: float = None,
        caller: str = "unknown"
    ) -> Tuple[str, Optional[Dict]]:
        """Generate response using OpenAI-compatible chat completions API and cost tracking"""
        last_exception = None

        # Retry loop using config values
        for attempt in range(self.max_retries):
            try:
                # Wrap entire operation in timeout from config
                async def _make_request():
                    max_tokens_val = max_tokens or self.default_max_tokens
                    temperature_val = temperature or self.default_temperature
                    self.logger.log_debug(f"Making async API call for {caller} (attempt {attempt + 1}/{self.max_retries})", "llm_interface")

                    session = await self.get_session()

                    request_payload = {
                        "model": self.model_name,
                        "messages": messages,
                        "max_tokens": max_tokens_val,
                        "temperature": temperature_val,
                        "stop": None  # Explicitly disable stop sequences to prevent premature stopping
                    }

                    # Log the request for debugging
                    self.logger.log_debug(
                        f"Request for {caller}: model={self.model_name}, max_tokens={max_tokens_val}, "
                        f"messages={len(messages)} items, temp={temperature_val}",
                        "llm_interface"
                    )

                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        json=request_payload
                    ) as response:
                        response.raise_for_status()
                        try:
                            result = await response.json(content_type=None)
                        except Exception as e:
                            raw_text = await response.text()
                            self.logger.log_error(
                                f"Non-JSON response for {caller}: {raw_text[:500]}",
                                "llm_interface",
                                e
                            )
                            return "", None

                        # Log usage details for debugging
                        usage = result.get("usage", {})
                        completion_details = usage.get("completion_tokens_details", {})
                        self.logger.log_debug(
                            f"Response for {caller}: "
                            f"prompt_tokens={usage.get('prompt_tokens', 0)}, "
                            f"completion_tokens={usage.get('completion_tokens', 0)}, "
                            f"reasoning_tokens={completion_details.get('reasoning_tokens', 0)}, "
                            f"content_tokens={completion_details.get('content_tokens', 0)}",
                            "llm_interface"
                        )

                        response_text = self._extract_response_text(result)
                        if not response_text:
                            # Log more details about the empty response
                            choices_info = "N/A"
                            if "choices" in result and result["choices"]:
                                first_choice = result["choices"][0]
                                choices_info = f"choices[0] keys: {list(first_choice.keys()) if isinstance(first_choice, dict) else 'not a dict'}"
                                if isinstance(first_choice, dict) and "message" in first_choice:
                                    msg = first_choice["message"]
                                    choices_info += f", message keys: {list(msg.keys()) if isinstance(msg, dict) else 'not a dict'}"
                                    if isinstance(msg, dict):
                                        choices_info += f", content='{msg.get('content', 'N/A')[:100]}'"

                            self.logger.log_warning(
                                f"Empty response content for {caller}; response keys: {list(result.keys())}, {choices_info}",
                                "llm_interface"
                            )

                            # Log full response for debugging (only if DEBUG level)
                            import json
                            self.logger.log_debug(
                                f"Full empty response for {caller}: {json.dumps(result, indent=2)}",
                                "llm_interface"
                            )

                        # Track token usage and costs
                        cost_info = self.cost_tracker.track_conversation(caller, messages, response_text)

                        self.logger.log_debug(
                            f"Async API call successful for {caller} - Response length: {len(response_text)}, "
                            f"Tokens: {cost_info['tokens']['total_tokens']}, Cost: ${cost_info['costs_usd']['total_cost']:.6f}",
                            "llm_interface"
                        )
                        return response_text, cost_info

                # Apply timeout from config
                result = await asyncio.wait_for(_make_request(), timeout=self.retry_timeout)
                return result

            except asyncio.TimeoutError:
                last_exception = asyncio.TimeoutError(f"Request timed out after {self.retry_timeout}s")
                if attempt < self.max_retries - 1:
                    self.logger.log_warning(
                        f"Async API call timed out for {caller} (attempt {attempt + 1}/{self.max_retries}) after {self.retry_timeout}s. Retrying in {self.retry_delay}s...",
                        "llm_interface"
                    )
                    await asyncio.sleep(self.retry_delay)
                else:
                    self.logger.log_error(
                        f"Async API call timed out for {caller} after {self.max_retries} attempts ({self.retry_timeout}s each)",
                        "llm_interface",
                        last_exception
                    )

            except Exception as e:
                import traceback
                last_exception = e
                error_details = {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }

                if attempt < self.max_retries - 1:
                    self.logger.log_warning(
                        f"Async API call failed for {caller} (attempt {attempt + 1}/{self.max_retries}): {error_details['type']} - {error_details['message']}. Retrying in {self.retry_delay}s...",
                        "llm_interface"
                    )
                    await asyncio.sleep(self.retry_delay)
                else:
                    self.logger.log_error(
                        f"Async API call failed for {caller} after {self.max_retries} attempts: {error_details['type']} - {error_details['message']}",
                        "llm_interface",
                        e
                    )
                    self.logger.log_debug(f"Full traceback for {caller}:\n{error_details['traceback']}", "llm_interface")

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
