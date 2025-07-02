"""Retry and error recovery utilities for DoDHaluEval."""

import asyncio
import functools
import random
import time
from typing import (
    Callable, 
    TypeVar, 
    Union, 
    List, 
    Type, 
    Optional, 
    Any, 
    Awaitable,
    Tuple
)
from datetime import datetime, timedelta

from dodhalueval.utils.exceptions import (
    DoDHaluEvalError,
    APIError,
    RateLimitError,
    TimeoutError as DoDTimeoutError,
    AuthenticationError
)
from dodhalueval.utils.logger import get_logger, OperationContext

T = TypeVar('T')
logger = get_logger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_backoff: bool = True,
        jitter: bool = True,
        retry_on_exceptions: Optional[Tuple[Type[Exception], ...]] = None
    ):
        """Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            exponential_backoff: Use exponential backoff
            jitter: Add random jitter to delays
            retry_on_exceptions: Tuple of exception types to retry on
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter
        self.retry_on_exceptions = retry_on_exceptions or (
            APIError, 
            RateLimitError, 
            DoDTimeoutError, 
            ConnectionError,
            OSError
        )
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        if self.exponential_backoff:
            delay = self.base_delay * (2 ** attempt)
        else:
            delay = self.base_delay
        
        # Apply maximum delay
        delay = min(delay, self.max_delay)
        
        # Add jitter
        if self.jitter:
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an exception should trigger a retry.
        
        Args:
            exception: Exception that occurred
            attempt: Current attempt number (0-based)
            
        Returns:
            True if should retry, False otherwise
        """
        # Check if we've exceeded max attempts
        if attempt >= self.max_attempts:
            return False
        
        # Never retry authentication errors
        if isinstance(exception, AuthenticationError):
            return False
        
        # Check if exception type is retryable
        return isinstance(exception, self.retry_on_exceptions)


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_backoff: bool = True,
    jitter: bool = True,
    retry_on_exceptions: Optional[Tuple[Type[Exception], ...]] = None
) -> Callable:
    """Decorator for adding retry logic to functions.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_backoff: Use exponential backoff
        jitter: Add random jitter to delays
        retry_on_exceptions: Tuple of exception types to retry on
        
    Returns:
        Decorated function with retry logic
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_backoff=exponential_backoff,
        jitter=jitter,
        retry_on_exceptions=retry_on_exceptions
    )
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):
            return _async_retry_wrapper(func, config)
        else:
            return _sync_retry_wrapper(func, config)
    
    return decorator


def _sync_retry_wrapper(func: Callable[..., T], config: RetryConfig) -> Callable[..., T]:
    """Synchronous retry wrapper."""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        last_exception = None
        
        for attempt in range(config.max_attempts):
            try:
                with OperationContext(
                    logger, 
                    f"{func.__name__}_attempt_{attempt + 1}",
                    attempt=attempt + 1,
                    max_attempts=config.max_attempts
                ):
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                if not config.should_retry(e, attempt):
                    logger.error(
                        f"Function {func.__name__} failed permanently",
                        function=func.__name__,
                        attempt=attempt + 1,
                        error=str(e)
                    )
                    raise
                
                delay = config.calculate_delay(attempt)
                
                logger.warning(
                    f"Function {func.__name__} failed, retrying in {delay:.2f}s",
                    function=func.__name__,
                    attempt=attempt + 1,
                    delay=delay,
                    error=str(e)
                )
                
                # Special handling for rate limit errors
                if isinstance(e, RateLimitError) and e.retry_after:
                    delay = max(delay, e.retry_after)
                
                time.sleep(delay)
        
        # If we get here, all attempts failed
        logger.error(
            f"Function {func.__name__} failed after {config.max_attempts} attempts",
            function=func.__name__,
            max_attempts=config.max_attempts,
            final_error=str(last_exception)
        )
        raise last_exception
    
    return wrapper


def _async_retry_wrapper(
    func: Callable[..., Awaitable[T]], 
    config: RetryConfig
) -> Callable[..., Awaitable[T]]:
    """Asynchronous retry wrapper."""
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        last_exception = None
        
        for attempt in range(config.max_attempts):
            try:
                with OperationContext(
                    logger, 
                    f"{func.__name__}_attempt_{attempt + 1}",
                    attempt=attempt + 1,
                    max_attempts=config.max_attempts
                ):
                    return await func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                if not config.should_retry(e, attempt):
                    logger.error(
                        f"Async function {func.__name__} failed permanently",
                        function=func.__name__,
                        attempt=attempt + 1,
                        error=str(e)
                    )
                    raise
                
                delay = config.calculate_delay(attempt)
                
                logger.warning(
                    f"Async function {func.__name__} failed, retrying in {delay:.2f}s",
                    function=func.__name__,
                    attempt=attempt + 1,
                    delay=delay,
                    error=str(e)
                )
                
                # Special handling for rate limit errors
                if isinstance(e, RateLimitError) and e.retry_after:
                    delay = max(delay, e.retry_after)
                
                await asyncio.sleep(delay)
        
        # If we get here, all attempts failed
        logger.error(
            f"Async function {func.__name__} failed after {config.max_attempts} attempts",
            function=func.__name__,
            max_attempts=config.max_attempts,
            final_error=str(last_exception)
        )
        raise last_exception
    
    return wrapper


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting recovery
            expected_exception: Exception type that triggers circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = 'closed'  # closed, open, half-open
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to apply circuit breaker to a function."""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if self.state == 'open':
                if self._should_attempt_reset():
                    self.state = 'half-open'
                    logger.info("Circuit breaker attempting reset", state=self.state)
                else:
                    logger.warning("Circuit breaker is open, failing fast")
                    raise DoDHaluEvalError("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful function execution."""
        self.failure_count = 0
        if self.state == 'half-open':
            self.state = 'closed'
            logger.info("Circuit breaker reset to closed", state=self.state)
    
    def _on_failure(self) -> None:
        """Handle failed function execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
            logger.warning(
                "Circuit breaker opened due to failures",
                failure_count=self.failure_count,
                threshold=self.failure_threshold,
                state=self.state
            )


class GracefulDegradation:
    """Implements graceful degradation patterns."""
    
    @staticmethod
    def fallback_on_error(
        fallback_func: Callable[..., T],
        exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ) -> Callable:
        """Decorator to provide fallback functionality on errors.
        
        Args:
            fallback_func: Function to call if primary function fails
            exceptions: Exception types that trigger fallback
            
        Returns:
            Decorated function with fallback logic
        """
        def decorator(primary_func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(primary_func)
            def wrapper(*args, **kwargs) -> T:
                try:
                    return primary_func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(
                        f"Primary function {primary_func.__name__} failed, using fallback",
                        primary_function=primary_func.__name__,
                        fallback_function=fallback_func.__name__,
                        error=str(e)
                    )
                    return fallback_func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    @staticmethod
    def timeout_with_partial_results(
        timeout_seconds: float,
        partial_result_func: Optional[Callable[..., T]] = None
    ) -> Callable:
        """Decorator to return partial results on timeout.
        
        Args:
            timeout_seconds: Timeout in seconds
            partial_result_func: Function to generate partial results
            
        Returns:
            Decorated function with timeout and partial results
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                import signal
                
                def timeout_handler(signum, frame):
                    if partial_result_func:
                        logger.warning(
                            f"Function {func.__name__} timed out, returning partial results",
                            timeout=timeout_seconds
                        )
                        raise TimeoutError("Partial results available")
                    else:
                        raise DoDTimeoutError(
                            f"Function {func.__name__} timed out after {timeout_seconds}s",
                            operation=func.__name__,
                            timeout_seconds=timeout_seconds
                        )
                
                # Set up timeout handler
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout_seconds))
                
                try:
                    result = func(*args, **kwargs)
                    signal.alarm(0)  # Cancel alarm
                    return result
                    
                except TimeoutError:
                    if partial_result_func:
                        return partial_result_func(*args, **kwargs)
                    raise
                    
                finally:
                    signal.signal(signal.SIGALRM, old_handler)
            
            return wrapper
        return decorator


def safe_execute(
    func: Callable[..., T],
    *args,
    default: Optional[T] = None,
    log_errors: bool = True,
    **kwargs
) -> Optional[T]:
    """Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        default: Default value to return on error
        log_errors: Whether to log errors
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default value on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(
                f"Safe execution of {func.__name__} failed",
                function=func.__name__,
                error=str(e)
            )
        return default


async def safe_execute_async(
    func: Callable[..., Awaitable[T]],
    *args,
    default: Optional[T] = None,
    log_errors: bool = True,
    **kwargs
) -> Optional[T]:
    """Safely execute an async function with error handling.
    
    Args:
        func: Async function to execute
        *args: Function arguments
        default: Default value to return on error
        log_errors: Whether to log errors
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default value on error
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(
                f"Safe async execution of {func.__name__} failed",
                function=func.__name__,
                error=str(e)
            )
        return default