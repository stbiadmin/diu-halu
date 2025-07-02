"""Command-line interface for DoDHaluEval."""

import sys
from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console
from rich.table import Table

from dodhalueval.utils.config import load_config, get_default_config
from dodhalueval.utils.logger import get_logger, setup_logging
from dodhalueval.data.pdf_processor import PDFProcessor
from dodhalueval.models.schemas import BenchmarkDataset
from dodhalueval.utils.exceptions import DoDHaluEvalError

console = Console()
logger = get_logger(__name__)


@click.group()
@click.option(
    '--config', '-c',
    type=click.Path(exists=True, path_type=Path),
    help='Configuration file path'
)
@click.option(
    '--environment', '-e',
    default='development',
    type=click.Choice(['development', 'production', 'testing']),
    help='Environment configuration to use'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose logging'
)
@click.option(
    '--quiet', '-q',
    is_flag=True,
    help='Suppress output except errors'
)
@click.pass_context
def cli(ctx: click.Context, config: Optional[Path], environment: str, verbose: bool, quiet: bool) -> None:
    """DoDHaluEval - DoD Hallucination Evaluation Benchmark
    
    A comprehensive benchmark for evaluating large language model hallucinations
    in the Department of Defense knowledge domain.
    """
    # Initialize context
    ctx.ensure_object(dict)
    
    try:
        # Load configuration
        if config:
            ctx.obj['config'] = load_config(str(config))
        else:
            ctx.obj['config'] = load_config(environment=environment)
        
        # Setup logging
        if verbose:
            ctx.obj['config'].logging.level = 'DEBUG'
        elif quiet:
            ctx.obj['config'].logging.level = 'ERROR'
            ctx.obj['config'].logging.console_output = False
        
        setup_logging(ctx.obj['config'].logging)
        
        logger.info(f"DoDHaluEval initialized with {environment} configuration")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to initialize: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    '--input', '-i',
    'input_dir',
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help='Input directory containing PDF documents'
)
@click.option(
    '--output', '-o',
    'output_dir',
    required=True,
    type=click.Path(path_type=Path),
    help='Output directory for processed documents'
)
@click.option(
    '--max-pages',
    type=int,
    help='Maximum pages to process per document (for testing)'
)
@click.option(
    '--chunk-size',
    type=int,
    default=1000,
    help='Target size for text chunks'
)
@click.option(
    '--chunk-overlap',
    type=int,
    default=200,
    help='Overlap between chunks'
)
@click.option(
    '--force',
    is_flag=True,
    help='Force reprocessing even if cached results exist'
)
@click.pass_context
def process_docs(
    ctx: click.Context,
    input_dir: Path,
    output_dir: Path,
    max_pages: Optional[int],
    chunk_size: int,
    chunk_overlap: int,
    force: bool
) -> None:
    """Process DoD PDF documents for hallucination evaluation.
    
    Extracts text, creates chunks, and prepares documents for prompt generation.
    """
    try:
        config = ctx.obj['config']
        
        # Initialize PDF processor
        processor = PDFProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            cache_enabled=config.cache.enabled and not force,
            cache_dir=config.cache.pdf_cache_dir,
            max_pages=max_pages
        )
        
        # Find PDF files
        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            console.print(f"[yellow]No PDF files found in {input_dir}[/yellow]")
            return
        
        console.print(f"Found {len(pdf_files)} PDF files to process")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each PDF
        processed_docs = []
        
        with console.status("[bold green]Processing documents...") as status:
            for i, pdf_file in enumerate(pdf_files):
                status.update(f"Processing {pdf_file.name} ({i+1}/{len(pdf_files)})")
                
                try:
                    result = processor.process_document(pdf_file)
                    processed_docs.append(result)
                    
                    # Save individual document result
                    output_file = output_dir / f"{pdf_file.stem}_processed.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        import json
                        json.dump(result, f, indent=2, default=str)
                    
                    logger.info(f"Processed {pdf_file.name}: {len(result['chunks'])} chunks")
                    
                except Exception as e:
                    logger.error(f"Failed to process {pdf_file.name}: {e}")
                    continue
        
        # Create summary
        total_chunks = sum(len(doc['chunks']) for doc in processed_docs)
        total_pages = sum(len(doc['pages']) for doc in processed_docs)
        
        console.print("\n[bold green]Processing Complete![/bold green]")
        console.print(f"Documents processed: {len(processed_docs)}")
        console.print(f"Total pages: {total_pages}")
        console.print(f"Total chunks: {total_chunks}")
        console.print(f"Output directory: {output_dir}")
        
    except DoDHaluEvalError as e:
        logger.error(f"Processing failed: {e}")
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error during processing")
        console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.option(
    '--config-file', '-c',
    type=click.Path(path_type=Path),
    help='Configuration file to validate'
)
@click.option(
    '--environment', '-e',
    type=click.Choice(['development', 'production', 'testing']),
    help='Environment configuration to validate'
)
@click.pass_context
def validate_config(ctx: click.Context, config_file: Optional[Path], environment: Optional[str]) -> None:
    """Validate configuration file or environment settings."""
    try:
        if config_file:
            config = load_config(str(config_file))
            console.print(f"[green]✓[/green] Configuration file {config_file} is valid")
        elif environment:
            config = load_config(environment=environment)
            console.print(f"[green]✓[/green] Environment '{environment}' configuration is valid")
        else:
            # Validate current context config
            config = ctx.obj['config']
            console.print("[green]✓[/green] Current configuration is valid")
        
        # Display configuration summary
        table = Table(title="Configuration Summary")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Environment", config.environment)
        table.add_row("Version", config.version)
        table.add_row("Batch Size", str(config.batch_size))
        table.add_row("Cache Enabled", str(config.cache.enabled))
        table.add_row("Log Level", config.logging.level)
        table.add_row("API Providers", ", ".join(config.api_configs.keys()))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Configuration validation failed: {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """Display system information and current configuration."""
    try:
        config = ctx.obj['config']
        
        # System information
        console.print("[bold blue]DoDHaluEval System Information[/bold blue]\n")
        
        info_table = Table(title="System Details")
        info_table.add_column("Component", style="cyan")
        info_table.add_column("Information", style="green")
        
        # Python and system info
        import platform
        info_table.add_row("Python Version", platform.python_version())
        info_table.add_row("Platform", f"{platform.system()} {platform.release()}")
        info_table.add_row("Architecture", platform.machine())
        
        # Package info
        try:
            import dodhalueval
            info_table.add_row("DoDHaluEval Version", dodhalueval.__version__)
        except:
            info_table.add_row("DoDHaluEval Version", "Development")
        
        # Configuration info
        info_table.add_row("Environment", config.environment)
        info_table.add_row("Config Version", config.version)
        info_table.add_row("Data Directory", config.data_dir)
        info_table.add_row("Cache Enabled", "Yes" if config.cache.enabled else "No")
        
        console.print(info_table)
        
        # API Providers
        if config.api_configs:
            console.print("\n[bold blue]Configured API Providers[/bold blue]")
            provider_table = Table()
            provider_table.add_column("Provider", style="cyan")
            provider_table.add_column("Model", style="green")
            provider_table.add_column("Status", style="yellow")
            
            for provider, api_config in config.api_configs.items():
                status = "Configured" if api_config.api_key else "Missing API Key"
                provider_table.add_row(provider, api_config.model, status)
            
            console.print(provider_table)
        
        # Evaluation Methods
        if config.evaluation_methods:
            console.print("\n[bold blue]Evaluation Methods[/bold blue]")
            eval_table = Table()
            eval_table.add_column("Method", style="cyan")
            eval_table.add_column("Enabled", style="green")
            eval_table.add_column("Batch Size", style="yellow")
            
            for method_config in config.evaluation_methods:
                enabled = "Yes" if method_config.enabled else "No"
                eval_table.add_row(
                    method_config.method,
                    enabled,
                    str(method_config.batch_size)
                )
            
            console.print(eval_table)
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to retrieve system information: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    '--output', '-o',
    'output_file',
    required=True,
    type=click.Path(path_type=Path),
    help='Output file for default configuration'
)
@click.option(
    '--environment', '-e',
    default='development',
    type=click.Choice(['development', 'production', 'testing']),
    help='Environment template to generate'
)
@click.option(
    '--overwrite',
    is_flag=True,
    help='Overwrite existing configuration file'
)
def generate_config(output_file: Path, environment: str, overwrite: bool) -> None:
    """Generate a default configuration file."""
    try:
        if output_file.exists() and not overwrite:
            console.print(f"[yellow]Configuration file {output_file} already exists. Use --overwrite to replace it.[/yellow]")
            return
        
        # Load template configuration
        config = load_config(environment=environment)
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        from dodhalueval.utils.config import ConfigLoader
        loader = ConfigLoader()
        loader.save_config(config, str(output_file))
        
        console.print(f"[green]✓[/green] Generated {environment} configuration: {output_file}")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to generate configuration: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    '--dataset', '-d',
    'dataset_file',
    type=click.Path(exists=True, path_type=Path),
    help='Dataset file to analyze'
)
@click.option(
    '--format',
    type=click.Choice(['json', 'jsonl']),
    default='jsonl',
    help='Dataset file format'
)
def analyze_dataset(dataset_file: Optional[Path], format: str) -> None:
    """Analyze a benchmark dataset and display statistics."""
    try:
        if not dataset_file:
            console.print("[yellow]No dataset file specified[/yellow]")
            return
        
        # Load dataset (simplified for Phase 1)
        console.print(f"[blue]Analyzing dataset:[/blue] {dataset_file}")
        
        # Basic file analysis
        file_size = dataset_file.stat().st_size
        console.print(f"File size: {file_size:,} bytes")
        
        if format == 'jsonl':
            # Count lines in JSONL file
            with open(dataset_file, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            console.print(f"Number of entries: {line_count}")
        
        console.print("[green]✓[/green] Dataset analysis complete")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to analyze dataset: {e}")
        sys.exit(1)


@cli.command()
def version() -> None:
    """Display version information."""
    try:
        import dodhalueval
        version = dodhalueval.__version__
    except:
        version = "Development"
    
    console.print(f"DoDHaluEval version: {version}")


if __name__ == '__main__':
    cli()