"""
Async Utilities for Rate Limiting and Retry Strategy

This module provides decorators for controlling async function call rates and implementing
retry strategies with exponential backoff for robust API interactions.
"""

import asyncio
import functools
import http.client
import requests
from functools import wraps
from typing import Any, Callable


def limit_async_func_call(max_size: int = None, config_path: str = None):
    """
    Add restriction of maximum async calling times for a async func.

    Args:
        max_size: Maximum concurrent calls. If None, will read from config_path.
        config_path: Dot-notation path to config value (e.g., 'idea_generation.async_func_max_size').
                    Only used if max_size is None and the decorated function's first arg has a 'config' attribute.
    """
    # Use a dict to store semaphores per effective_max_size
    _semaphores = {}

    def decro(func):
        @wraps(func)
        async def wait_func(*args, **kwargs):
            # Determine effective max_size
            effective_max_size = max_size
            if effective_max_size is None and config_path and args:
                instance = args[0]
                if hasattr(instance, 'config'):
                    config = instance.config
                    for key in config_path.split('.'):
                        config = config.get(key, {})
                        if not isinstance(config, dict):
                            effective_max_size = config
                            break
                    else:
                        effective_max_size = 8  # Fallback default
                else:
                    effective_max_size = 8  # Fallback default
            elif effective_max_size is None:
                effective_max_size = 8  # Fallback default

            # Get or create semaphore for this max_size
            if effective_max_size not in _semaphores:
                _semaphores[effective_max_size] = asyncio.Semaphore(effective_max_size)

            sem = _semaphores[effective_max_size]
            async with sem:
                return await func(*args, **kwargs)

        return wait_func

    return decro


def retry_with_timeout(max_retries: int = None, timeout: int = None, delay: float = None):
    """
    Retry decorator with timeout and exponential backoff for async functions.

    Args:
        max_retries: Maximum number of retry attempts (None = read from instance config)
        timeout: Timeout in seconds for each attempt (None = read from instance config)
        delay: Initial delay between retries (None = read from instance config)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Try to get config from instance (args[0] is self for instance methods)
            retries = max_retries
            timeout_val = timeout
            delay_val = delay

            if retries is None or timeout_val is None or delay_val is None:
                try:
                    # Try to read from instance's llm config
                    if args and hasattr(args[0], 'llm'):
                        llm = args[0].llm
                        if retries is None and hasattr(llm, 'max_retries'):
                            retries = llm.max_retries
                        if timeout_val is None and hasattr(llm, 'retry_timeout'):
                            timeout_val = llm.retry_timeout
                        if delay_val is None and hasattr(llm, 'retry_delay'):
                            delay_val = llm.retry_delay
                except:
                    pass

            # Fall back to defaults if still None
            retries = retries if retries is not None else 5
            timeout_val = timeout_val if timeout_val is not None else 60
            delay_val = delay_val if delay_val is not None else 1

            for attempt in range(retries):
                try:
                    return await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout_val
                    )
                except Exception as e:
                    # Handle common retry-able exceptions
                    retryable_exceptions = (
                        asyncio.TimeoutError,
                        ConnectionError,
                        OSError,  # Network errors
                    )
                    
                    # Add requests exceptions if available
                    try:
                        import requests
                        retryable_exceptions += (
                            requests.exceptions.ProxyError,
                            requests.exceptions.RequestException,
                        )
                    except ImportError:
                        pass
                    
                    # Add aiohttp exceptions if available
                    try:
                        import aiohttp
                        retryable_exceptions += (
                            aiohttp.ClientError,
                            aiohttp.ServerTimeoutError,
                            aiohttp.ClientResponseError,
                        )
                    except ImportError:
                        pass
                    
                    # Add http client exceptions
                    try:
                        retryable_exceptions += (http.client.RemoteDisconnected,)
                    except:
                        pass
                    
                    if not isinstance(e, retryable_exceptions):
                        # Not a retryable exception, re-raise immediately
                        raise
                    if attempt == retries - 1:
                        print(f"All {retries} attempts failed. Final error: {str(e)}")
                        raise
                    print(f"Attempt {attempt + 1}/{retries} failed: {str(e)}")
                    await asyncio.sleep(delay_val * (2 ** attempt))  # Exponential backoff
            # This should never be reached due to the loop logic, but adding safety
            raise RuntimeError(f"Failed after {retries} attempts")
        return wrapper
    return decorator


def async_batch_processor(batch_size: int = 5, delay_between_batches: float = 0.1):
    """
    Process items in batches asynchronously with optional delay between batches.
    
    Args:
        batch_size: Number of items to process in each batch
        delay_between_batches: Delay in seconds between processing batches
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(items, *args, **kwargs):
            results = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_tasks = [func(item, *args, **kwargs) for item in batch]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                results.extend(batch_results)
                
                # Add delay between batches if there are more batches to process
                if i + batch_size < len(items) and delay_between_batches > 0:
                    await asyncio.sleep(delay_between_batches)
                    
            return results
        return wrapper
    return decorator


class AsyncRateLimiter:
    """
    Rate limiter for async operations with configurable rates.
    """
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a call, waiting if necessary"""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            
            # Remove calls outside the time window
            self.calls = [call_time for call_time in self.calls 
                         if now - call_time < self.time_window]
            
            # Wait if we've hit the limit
            while len(self.calls) >= self.max_calls:
                sleep_time = self.time_window - (now - self.calls[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                now = asyncio.get_event_loop().time()
                self.calls = [call_time for call_time in self.calls 
                             if now - call_time < self.time_window]
            
            # Record this call
            self.calls.append(now)


def rate_limited(max_calls: int, time_window: float):
    """
    Rate limiting decorator using AsyncRateLimiter.
    
    Args:
        max_calls: Maximum number of calls allowed
        time_window: Time window in seconds
    """
    rate_limiter = AsyncRateLimiter(max_calls, time_window)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            await rate_limiter.acquire()
            return await func(*args, **kwargs)
        return wrapper
    return decorator