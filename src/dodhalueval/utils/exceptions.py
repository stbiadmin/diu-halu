"""Custom exception classes for DoDHaluEval."""

from typing import Optional, Any, Dict


class DoDHaluEvalError(Exception):
    """Base exception for all DoDHaluEval errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """String representation of the exception."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(DoDHaluEvalError):
    """Raised when configuration is invalid or missing."""
    pass


class PDFProcessingError(DoDHaluEvalError):
    """Raised when PDF processing fails."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, page_num: Optional[int] = None):
        """Initialize PDF processing error.
        
        Args:
            message: Error message
            file_path: Path to the problematic PDF file
            page_num: Page number where error occurred
        """
        details = {}
        if file_path:
            details['file_path'] = file_path
        if page_num is not None:
            details['page_num'] = page_num
        
        super().__init__(message, details)
        self.file_path = file_path
        self.page_num = page_num


class APIError(DoDHaluEvalError):
    """Raised when API calls fail."""
    
    def __init__(
        self, 
        message: str, 
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None
    ):
        """Initialize API error.
        
        Args:
            message: Error message
            provider: API provider name
            status_code: HTTP status code
            response_text: API response text
        """
        details = {}
        if provider:
            details['provider'] = provider
        if status_code is not None:
            details['status_code'] = status_code
        if response_text:
            details['response_text'] = response_text[:500]  # Truncate long responses
        
        super().__init__(message, details)
        self.provider = provider
        self.status_code = status_code
        self.response_text = response_text


class RateLimitError(APIError):
    """Raised when API rate limits are exceeded."""
    
    def __init__(
        self, 
        message: str, 
        provider: Optional[str] = None,
        retry_after: Optional[int] = None
    ):
        """Initialize rate limit error.
        
        Args:
            message: Error message
            provider: API provider name
            retry_after: Seconds to wait before retrying
        """
        super().__init__(message, provider)
        if retry_after is not None:
            self.details['retry_after'] = retry_after
        self.retry_after = retry_after


class AuthenticationError(APIError):
    """Raised when API authentication fails."""
    pass


class ModelNotFoundError(APIError):
    """Raised when requested model is not available."""
    
    def __init__(self, message: str, provider: Optional[str] = None, model: Optional[str] = None):
        """Initialize model not found error.
        
        Args:
            message: Error message
            provider: API provider name
            model: Model name that was not found
        """
        super().__init__(message, provider)
        if model:
            self.details['model'] = model
        self.model = model


class ValidationError(DoDHaluEvalError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        """Initialize validation error.
        
        Args:
            message: Error message
            field: Field name that failed validation
            value: Value that failed validation
        """
        details = {}
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = str(value)[:100]  # Truncate long values
        
        super().__init__(message, details)
        self.field = field
        self.value = value


class DataProcessingError(DoDHaluEvalError):
    """Raised when data processing operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None, data_type: Optional[str] = None):
        """Initialize data processing error.
        
        Args:
            message: Error message
            operation: Type of operation that failed
            data_type: Type of data being processed
        """
        details = {}
        if operation:
            details['operation'] = operation
        if data_type:
            details['data_type'] = data_type
        
        super().__init__(message, details)
        self.operation = operation
        self.data_type = data_type


class CacheError(DoDHaluEvalError):
    """Raised when cache operations fail."""
    
    def __init__(self, message: str, cache_type: Optional[str] = None, cache_key: Optional[str] = None):
        """Initialize cache error.
        
        Args:
            message: Error message
            cache_type: Type of cache (pdf, llm, etc.)
            cache_key: Cache key that caused the error
        """
        details = {}
        if cache_type:
            details['cache_type'] = cache_type
        if cache_key:
            details['cache_key'] = cache_key
        
        super().__init__(message, details)
        self.cache_type = cache_type
        self.cache_key = cache_key


class EvaluationError(DoDHaluEvalError):
    """Raised when evaluation operations fail."""
    
    def __init__(
        self, 
        message: str, 
        method: Optional[str] = None,
        prompt_id: Optional[str] = None,
        response_id: Optional[str] = None
    ):
        """Initialize evaluation error.
        
        Args:
            message: Error message
            method: Evaluation method that failed
            prompt_id: ID of prompt being evaluated
            response_id: ID of response being evaluated
        """
        details = {}
        if method:
            details['method'] = method
        if prompt_id:
            details['prompt_id'] = prompt_id
        if response_id:
            details['response_id'] = response_id
        
        super().__init__(message, details)
        self.method = method
        self.prompt_id = prompt_id
        self.response_id = response_id


class PromptGenerationError(DoDHaluEvalError):
    """Raised when prompt generation fails."""
    
    def __init__(
        self, 
        message: str, 
        strategy: Optional[str] = None,
        document_id: Optional[str] = None
    ):
        """Initialize prompt generation error.
        
        Args:
            message: Error message
            strategy: Generation strategy that failed
            document_id: ID of document being processed
        """
        details = {}
        if strategy:
            details['strategy'] = strategy
        if document_id:
            details['document_id'] = document_id
        
        super().__init__(message, details)
        self.strategy = strategy
        self.document_id = document_id


class FileSystemError(DoDHaluEvalError):
    """Raised when file system operations fail."""
    
    def __init__(self, message: str, path: Optional[str] = None, operation: Optional[str] = None):
        """Initialize file system error.
        
        Args:
            message: Error message
            path: File or directory path
            operation: File operation that failed (read, write, delete, etc.)
        """
        details = {}
        if path:
            details['path'] = path
        if operation:
            details['operation'] = operation
        
        super().__init__(message, details)
        self.path = path
        self.operation = operation


class ResourceError(DoDHaluEvalError):
    """Raised when system resources are insufficient."""
    
    def __init__(
        self, 
        message: str, 
        resource_type: Optional[str] = None,
        current_usage: Optional[str] = None,
        limit: Optional[str] = None
    ):
        """Initialize resource error.
        
        Args:
            message: Error message
            resource_type: Type of resource (memory, disk, etc.)
            current_usage: Current resource usage
            limit: Resource limit
        """
        details = {}
        if resource_type:
            details['resource_type'] = resource_type
        if current_usage:
            details['current_usage'] = current_usage
        if limit:
            details['limit'] = limit
        
        super().__init__(message, details)
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit


class TimeoutError(DoDHaluEvalError):
    """Raised when operations timeout."""
    
    def __init__(
        self, 
        message: str, 
        operation: Optional[str] = None,
        timeout_seconds: Optional[float] = None
    ):
        """Initialize timeout error.
        
        Args:
            message: Error message
            operation: Operation that timed out
            timeout_seconds: Timeout duration in seconds
        """
        details = {}
        if operation:
            details['operation'] = operation
        if timeout_seconds is not None:
            details['timeout_seconds'] = timeout_seconds
        
        super().__init__(message, details)
        self.operation = operation
        self.timeout_seconds = timeout_seconds