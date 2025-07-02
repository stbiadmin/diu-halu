#!/usr/bin/env python3
"""
Model Utilities for DoDHaluEval

Command-line utilities for model discovery, validation, and configuration.
"""

import sys
from pathlib import Path
import click
from typing import Optional

# Add the src directory to the path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

from dodhalueval.models.model_registry import get_model_registry, ModelCapability


def print_model_table(models, title="Models"):
    """Print models in a formatted table."""
    if not models:
        print(f"No {title.lower()} found.")
        return
    
    print(f"\n{title}")
    print("=" * len(title))
    
    for model in models:
        status = ""
        if model.is_default:
            status += " 🌟"
        if model.is_deprecated:
            status += " ⚠️"
        
        cost = f"${model.output_cost_per_token:.4f}/1k"
        context = f"{model.context_length:,}"
        
        print(f"\n📝 {model.display_name}{status}")
        print(f"   ID: {model.id}")
        print(f"   Provider: {model.provider}")
        print(f"   Description: {model.description}")
        print(f"   Context: {context} tokens | Cost: {cost}")
        print(f"   Capabilities: {', '.join([cap.value for cap in model.capabilities])}")
        if model.recommended_for:
            print(f"   Best for: {', '.join(model.recommended_for)}")


@click.group()
def cli():
    """DoDHaluEval Model Utilities - Discover, validate, and configure models."""
    pass


@cli.command()
@click.option('--provider', type=str, help='Filter by provider (openai, fireworks, mock)')
@click.option('--include-deprecated', is_flag=True, help='Include deprecated models')
def list_models(provider: Optional[str], include_deprecated: bool):
    """List all available models."""
    registry = get_model_registry()
    
    if provider:
        models = registry.get_models_by_provider(provider, include_deprecated)
        title = f"{provider.upper()} Models"
    else:
        # Get all models
        models = []
        for p in registry.list_providers():
            models.extend(registry.get_models_by_provider(p, include_deprecated))
        title = "All Available Models"
    
    print_model_table(models, title)
    
    if not include_deprecated:
        deprecated_count = sum(1 for m in models if m.is_deprecated)
        if deprecated_count > 0:
            print(f"\n💡 Use --include-deprecated to see {deprecated_count} deprecated models")


@cli.command()
def list_providers():
    """List all supported providers."""
    registry = get_model_registry()
    providers = registry.list_providers()
    
    print("\n🔧 Supported Providers")
    print("=" * 20)
    
    for provider in providers:
        models = registry.get_models_by_provider(provider)
        default = registry.get_default_model(provider)
        
        print(f"\n📡 {provider.upper()}")
        print(f"   Models available: {len(models)}")
        if default:
            print(f"   Default model: {default.display_name}")
        
        # Show model count by capability
        capabilities = {}
        for model in models:
            for cap in model.capabilities:
                capabilities[cap] = capabilities.get(cap, 0) + 1
        
        if capabilities:
            print("   Capabilities:")
            for cap, count in capabilities.items():
                print(f"     - {cap.value}: {count} models")


@cli.command()
@click.argument('model_id')
@click.option('--provider', type=str, help='Expected provider for validation')
def validate_model(model_id: str, provider: Optional[str]):
    """Validate if a model exists and get detailed information."""
    registry = get_model_registry()
    
    model = registry.get_model(model_id)
    
    if not model:
        print(f"❌ Model '{model_id}' not found in registry")
        
        # Suggest similar models
        all_models = []
        for p in registry.list_providers():
            all_models.extend(registry.get_models_by_provider(p))
        
        # Simple similarity check
        suggestions = []
        for m in all_models:
            if any(word in m.id.lower() for word in model_id.lower().split('-')):
                suggestions.append(m)
        
        if suggestions:
            print("\n💡 Similar models found:")
            for suggestion in suggestions[:3]:
                print(f"   - {suggestion.id} ({suggestion.provider})")
        return
    
    # Validate provider if specified
    if provider:
        is_valid, message = registry.validate_model(model_id, provider)
        if is_valid:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")
            return
    else:
        print(f"✅ Model '{model_id}' found")
    
    # Show detailed information
    print_model_table([model], f"Model Details: {model.display_name}")


@cli.command()
@click.argument('use_case')
@click.option('--provider', type=str, help='Filter by provider')
@click.option('--max-cost', type=float, help='Maximum cost per 1k output tokens')
def recommend(use_case: str, provider: Optional[str], max_cost: Optional[float]):
    """Get model recommendations for a specific use case."""
    registry = get_model_registry()
    
    models = registry.get_recommended_models(use_case, provider, max_cost)
    
    if not models:
        print(f"❌ No models found for use case: '{use_case}'")
        print("\n💡 Available use cases:")
        
        # Show available use cases
        all_models = []
        for p in registry.list_providers():
            all_models.extend(registry.get_models_by_provider(p))
        
        use_cases = set()
        for model in all_models:
            use_cases.update(model.recommended_for)
        
        for uc in sorted(use_cases):
            print(f"   - {uc}")
        return
    
    title = f"Recommended Models for '{use_case}'"
    if provider:
        title += f" ({provider})"
    if max_cost:
        title += f" (max cost: ${max_cost:.4f})"
    
    print_model_table(models, title)


@cli.command()
@click.argument('capability', type=click.Choice([cap.value for cap in ModelCapability]))
@click.option('--provider', type=str, help='Filter by provider')
def by_capability(capability: str, provider: Optional[str]):
    """List models by capability."""
    registry = get_model_registry()
    
    cap_enum = ModelCapability(capability)
    models = registry.get_models_by_capability(cap_enum, provider)
    
    title = f"Models with {capability.title()} Capability"
    if provider:
        title += f" ({provider})"
    
    print_model_table(models, title)


@cli.command()
@click.option('--provider', type=str, help='Show defaults for specific provider')
def defaults(provider: Optional[str]):
    """Show default models for each provider."""
    registry = get_model_registry()
    
    if provider:
        providers = [provider]
    else:
        providers = registry.list_providers()
    
    print("\n⭐ Default Models")
    print("=" * 15)
    
    for p in providers:
        default = registry.get_default_model(p)
        if default:
            print(f"\n🔧 {p.upper()}")
            print(f"   {default.display_name}")
            print(f"   ID: {default.id}")
            print(f"   Cost: ${default.output_cost_per_token:.4f}/1k tokens")
            print(f"   Use: {', '.join(default.recommended_for[:2])}")


@cli.command()
@click.argument('model_id')
def summary(model_id: str):
    """Get a one-line summary of a model."""
    registry = get_model_registry()
    summary_text = registry.get_model_summary(model_id)
    print(summary_text)


@cli.command()
def export_config():
    """Export model configuration for easy copy-paste into .env files."""
    registry = get_model_registry()
    
    print("\n📋 Model Configuration for .env")
    print("=" * 35)
    
    for provider in registry.list_providers():
        if provider == "mock":
            continue
            
        default = registry.get_default_model(provider)
        if default:
            env_var = f"DEFAULT_MODEL_{provider.upper()}"
            print(f"\n# {provider.title()} Configuration")
            print(f"{env_var}={default.id}")
            print(f"# Description: {default.description}")
            print(f"# Cost: ${default.output_cost_per_token:.4f}/1k tokens")


if __name__ == "__main__":
    cli()