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


def limit_async_func_call(max_size: int = None, waitting_time: float = 0.0001, config_path: str = None):
    """
    Add restriction of maximum async calling times for a async func.

    Args:
        max_size: Maximum concurrent calls. If None, will read from config_path.
        waitting_time: Sleep time when waiting for available slot.
        config_path: Dot-notation path to config value (e.g., 'idea_generation.async_func_max_size').
                    Only used if max_size is None and the decorated function's first arg has a 'config' attribute.
    """

    def decro(func):
        """Not using async.Semaphore to avoid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size

            # Determine effective max_size
            effective_max_size = max_size
            if effective_max_size is None and config_path and args:
                # Try to read from config of the first argument (usually self)
                instance = args[0]
                if hasattr(instance, 'config'):
                    config = instance.config
                    # Navigate through config path
                    for key in config_path.split('.'):
                        config = config.get(key, {})
                        if not isinstance(config, dict):
                            effective_max_size = config
                            break
                    else:
                        effective_max_size = 2  # Fallback default
                else:
                    effective_max_size = 2  # Fallback default
            elif effective_max_size is None:
                effective_max_size = 2  # Fallback default

            while __current_size >= effective_max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return decro


def retry_with_timeout(max_retries: int = 5, timeout: int = 60, delay: float = 1):
    """
    Retry decorator with timeout and exponential backoff for async functions.
    
    Args:
        max_retries: Maximum number of retry attempts
        timeout: Timeout in seconds for each attempt
        delay: Initial delay between retries (exponentially increased)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries):
                try:
                    return await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout
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
                    if attempt == max_retries - 1:
                        print(f"All {max_retries} attempts failed. Final error: {str(e)}")
                        raise
                    print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                    await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
            # This should never be reached due to the loop logic, but adding safety
            raise RuntimeError(f"Failed after {max_retries} attempts")
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