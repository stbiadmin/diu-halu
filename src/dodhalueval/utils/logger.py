"""Logging utilities for DoDHaluEval with Rich formatting."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict, Union
from logging.handlers import RotatingFileHandler
import json

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as rich_traceback_install
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from dodhalueval.models.config import LoggingConfig


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields from the record
        extra_fields = {
            k: v for k, v in record.__dict__.items()
            if k not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                        'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                        'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                        'thread', 'threadName', 'processName', 'process', 'getMessage']
        }
        if extra_fields:
            log_data['extra'] = extra_fields
        
        return json.dumps(log_data, default=str)


class DoDHaluEvalLogger:
    """Enhanced logger for DoDHaluEval with Rich formatting and context management."""
    
    def __init__(self, name: str, config: Optional[LoggingConfig] = None):
        """Initialize the logger.
        
        Args:
            name: Logger name
            config: Logging configuration
        """
        self.name = name
        self.config = config or LoggingConfig()
        self.logger = logging.getLogger(name)
        self.console = Console()
        self._progress: Optional[Progress] = None
        self._current_task = None
        
        # Install rich traceback handler
        rich_traceback_install(console=self.console, show_locals=True)
        
        # Configure logger
        self._configure_logger()
    
    def _configure_logger(self) -> None:
        """Configure the logger with handlers and formatters."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        level = getattr(logging, self.config.level.upper())
        self.logger.setLevel(level)
        
        # Console handler with Rich formatting
        if self.config.console_output:
            console_handler = RichHandler(
                console=self.console,
                show_time=True,
                show_path=True,
                rich_tracebacks=True,
                markup=True
            )
            console_handler.setLevel(level)
            
            if not self.config.structured_logging:
                console_formatter = logging.Formatter(
                    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                console_handler.setFormatter(console_formatter)
            
            self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.config.log_file:
            log_path = Path(self.config.log_dir) / self.config.log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            
            if self.config.structured_logging:
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_formatter = logging.Formatter(
                    fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                file_handler.setFormatter(file_formatter)
            
            self.logger.addHandler(file_handler)
        
        # Prevent duplicate logging
        self.logger.propagate = False
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self.logger.critical(message, extra=kwargs)
    
    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception message with traceback."""
        self.logger.exception(message, extra=kwargs)
    
    def log_operation_start(self, operation: str, **context: Any) -> None:
        """Log the start of an operation."""
        self.info(f"Starting operation: {operation}", operation=operation, **context)
    
    def log_operation_end(self, operation: str, success: bool = True, **context: Any) -> None:
        """Log the end of an operation."""
        status = "completed" if success else "failed"
        level_func = self.info if success else self.error
        level_func(f"Operation {status}: {operation}", operation=operation, success=success, **context)
    
    def log_api_call(
        self,
        provider: str,
        endpoint: str,
        method: str = "POST",
        status_code: Optional[int] = None,
        duration: Optional[float] = None,
        **context: Any
    ) -> None:
        """Log API call details."""
        self.info(
            f"API call to {provider}",
            provider=provider,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            duration=duration,
            **context
        )
    
    def log_processing_stats(
        self,
        operation: str,
        total_items: int,
        processed_items: int,
        failed_items: int = 0,
        duration: Optional[float] = None,
        **context: Any
    ) -> None:
        """Log processing statistics."""
        success_rate = (processed_items / total_items * 100) if total_items > 0 else 0
        self.info(
            f"Processing stats for {operation}",
            operation=operation,
            total_items=total_items,
            processed_items=processed_items,
            failed_items=failed_items,
            success_rate=f"{success_rate:.1f}%",
            duration=duration,
            **context
        )
    
    def start_progress(self, description: str, total: Optional[int] = None) -> None:
        """Start a progress bar."""
        if self._progress is None:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
                transient=True
            )
            self._progress.start()
        
        self._current_task = self._progress.add_task(description, total=total)
    
    def update_progress(self, advance: int = 1, description: Optional[str] = None) -> None:
        """Update the current progress bar."""
        if self._progress and self._current_task is not None:
            self._progress.update(self._current_task, advance=advance)
            if description:
                self._progress.update(self._current_task, description=description)
    
    def stop_progress(self) -> None:
        """Stop the current progress bar."""
        if self._progress:
            self._progress.stop()
            self._progress = None
            self._current_task = None
    
    def print_status(self, message: str, style: str = "bold blue") -> None:
        """Print a status message with Rich formatting."""
        self.console.print(f"[{style}]{message}[/{style}]")
    
    def print_success(self, message: str) -> None:
        """Print a success message."""
        self.console.print(f"[bold green]✓[/bold green] {message}")
    
    def print_error(self, message: str) -> None:
        """Print an error message."""
        self.console.print(f"[bold red]✗[/bold red] {message}")
    
    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        self.console.print(f"[bold yellow]⚠[/bold yellow] {message}")


class OperationContext:
    """Context manager for logging operations with automatic start/end logging."""
    
    def __init__(
        self,
        logger: DoDHaluEvalLogger,
        operation: str,
        **context: Any
    ):
        """Initialize operation context.
        
        Args:
            logger: Logger instance
            operation: Operation name
            **context: Additional context data
        """
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time: Optional[float] = None
    
    def __enter__(self) -> 'OperationContext':
        """Enter the operation context."""
        import time
        self.start_time = time.time()
        self.logger.log_operation_start(self.operation, **self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the operation context."""
        import time
        duration = time.time() - self.start_time if self.start_time else None
        success = exc_type is None
        
        context = self.context.copy()
        if duration:
            context['duration'] = f"{duration:.2f}s"
        
        self.logger.log_operation_end(self.operation, success=success, **context)
        
        if not success and exc_val:
            self.logger.exception(f"Operation failed: {self.operation}", error=str(exc_val))


# Global logger instance
_loggers: Dict[str, DoDHaluEvalLogger] = {}


def get_logger(name: str, config: Optional[LoggingConfig] = None) -> DoDHaluEvalLogger:
    """Get or create a logger instance.
    
    Args:
        name: Logger name
        config: Logging configuration
        
    Returns:
        Logger instance
    """
    if name not in _loggers:
        _loggers[name] = DoDHaluEvalLogger(name, config)
    return _loggers[name]


def setup_logging(config: LoggingConfig) -> None:
    """Setup global logging configuration.
    
    Args:
        config: Logging configuration
    """
    # Clear existing loggers
    _loggers.clear()
    
    # Set root logger level
    logging.getLogger().setLevel(getattr(logging, config.level.upper()))
    
    # Create log directory
    if config.log_file:
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)