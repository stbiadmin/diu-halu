#!/usr/bin/env python3
"""
Visualization script for DoDHaluEval test results.

This script creates visualizations for evaluation prompts and response results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from typing import Dict, List, Any
import numpy as np

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results_csv(filepath: Path) -> pd.DataFrame:
    """Load results from CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"📊 Loaded {len(df)} results from {filepath}")
        return df
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return pd.DataFrame()

def load_prompts_json(filepath: Path) -> List[Dict]:
    """Load prompts from JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        print(f"📝 Loaded {len(prompts)} prompts from {filepath}")
        return prompts
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return []

def create_hallucination_rate_chart(df: pd.DataFrame, output_dir: Path):
    """Create a chart showing hallucination rates by provider."""
    if df.empty:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Hallucination rate by provider
    halluc_by_provider = df.groupby('Provider')['Contains_Hallucination'].agg(['count', 'sum']).reset_index()
    halluc_by_provider['Rate'] = halluc_by_provider['sum'] / halluc_by_provider['count'] * 100
    
    bars1 = ax1.bar(halluc_by_provider['Provider'], halluc_by_provider['Rate'])
    ax1.set_title('Hallucination Rate by Provider')
    ax1.set_ylabel('Hallucination Rate (%)')
    ax1.set_xlabel('Provider')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Response count by provider
    response_counts = df['Provider'].value_counts()
    bars2 = ax2.bar(response_counts.index, response_counts.values)
    ax2.set_title('Total Responses by Provider')
    ax2.set_ylabel('Number of Responses')
    ax2.set_xlabel('Provider')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    output_file = output_dir / 'hallucination_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Saved hallucination analysis to {output_file}")
    plt.close()

def create_processing_time_chart(df: pd.DataFrame, output_dir: Path):
    """Create a chart showing processing times by provider."""
    if df.empty or 'Processing_Time' not in df.columns:
        return
    
    # Convert processing time to numeric and handle errors
    df['Processing_Time'] = pd.to_numeric(df['Processing_Time'], errors='coerce')
    df_clean = df.dropna(subset=['Processing_Time'])
    
    if df_clean.empty:
        print("⚠️ No valid processing time data found")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot of processing times
    providers = df_clean['Provider'].unique()
    processing_data = [df_clean[df_clean['Provider'] == p]['Processing_Time'].values 
                      for p in providers]
    
    box_plot = ax1.boxplot(processing_data, labels=providers, patch_artist=True)
    ax1.set_title('Processing Time Distribution by Provider')
    ax1.set_ylabel('Processing Time (seconds)')
    ax1.set_xlabel('Provider')
    
    # Color the boxes
    colors = sns.color_palette("husl", len(providers))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    # Average processing time by provider
    avg_times = df_clean.groupby('Provider')['Processing_Time'].mean()
    bars = ax2.bar(avg_times.index, avg_times.values)
    ax2.set_title('Average Processing Time by Provider')
    ax2.set_ylabel('Average Time (seconds)')
    ax2.set_xlabel('Provider')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    output_file = output_dir / 'processing_time_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Saved processing time analysis to {output_file}")
    plt.close()

def create_hallucination_types_chart(df: pd.DataFrame, output_dir: Path):
    """Create a chart showing distribution of hallucination types."""
    if df.empty or 'Hallucination_Types' not in df.columns:
        return
    
    # Extract hallucination types
    halluc_types = []
    for types_str in df['Hallucination_Types'].dropna():
        if types_str and types_str.strip():
            types = [t.strip() for t in types_str.split(';') if t.strip()]
            halluc_types.extend(types)
    
    if not halluc_types:
        print("⚠️ No hallucination types found in data")
        return
    
    # Count occurrences
    type_counts = pd.Series(halluc_types).value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(type_counts.index, type_counts.values)
    ax.set_title('Distribution of Hallucination Types')
    ax.set_ylabel('Count')
    ax.set_xlabel('Hallucination Type')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    output_file = output_dir / 'hallucination_types.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Saved hallucination types chart to {output_file}")
    plt.close()

def create_word_count_analysis(df: pd.DataFrame, output_dir: Path):
    """Create analysis of response word counts."""
    if df.empty or 'Word_Count' not in df.columns:
        return
    
    df['Word_Count'] = pd.to_numeric(df['Word_Count'], errors='coerce')
    df_clean = df.dropna(subset=['Word_Count'])
    
    if df_clean.empty:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram of word counts
    ax1.hist(df_clean['Word_Count'], bins=20, alpha=0.7, edgecolor='black')
    ax1.set_title('Distribution of Response Word Counts')
    ax1.set_xlabel('Word Count')
    ax1.set_ylabel('Frequency')
    ax1.axvline(df_clean['Word_Count'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df_clean["Word_Count"].mean():.1f}')
    ax1.legend()
    
    # Word count by provider
    avg_words = df_clean.groupby('Provider')['Word_Count'].mean()
    bars = ax2.bar(avg_words.index, avg_words.values)
    ax2.set_title('Average Word Count by Provider')
    ax2.set_ylabel('Average Word Count')
    ax2.set_xlabel('Provider')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    output_file = output_dir / 'word_count_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Saved word count analysis to {output_file}")
    plt.close()

def create_summary_report(df: pd.DataFrame, prompts: List[Dict], output_dir: Path):
    """Create a text summary report."""
    report_lines = []
    report_lines.append("🎯 DoDHaluEval Test Results Summary")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    if prompts:
        report_lines.append(f"📝 Total prompts tested: {len(prompts)}")
        
        # Prompt analysis
        halluc_types = [p.get('hallucination_type', 'unknown') for p in prompts]
        type_counts = pd.Series(halluc_types).value_counts()
        report_lines.append("Prompt breakdown by hallucination type:")
        for htype, count in type_counts.items():
            report_lines.append(f"  - {htype}: {count}")
        report_lines.append("")
    
    if not df.empty:
        report_lines.append(f"🤖 Total responses generated: {len(df)}")
        report_lines.append(f"📊 Providers tested: {', '.join(df['Provider'].unique())}")
        
        # Hallucination analysis
        total_halluc = df['Contains_Hallucination'].sum()
        halluc_rate = total_halluc / len(df) * 100
        report_lines.append(f"🎭 Overall hallucination rate: {halluc_rate:.1f}% ({total_halluc}/{len(df)})")
        
        # Provider breakdown
        report_lines.append("\\nProvider Performance:")
        for provider in df['Provider'].unique():
            provider_df = df[df['Provider'] == provider]
            provider_halluc = provider_df['Contains_Hallucination'].sum()
            provider_rate = provider_halluc / len(provider_df) * 100
            avg_time = provider_df['Processing_Time'].mean() if 'Processing_Time' in df.columns else 0
            report_lines.append(f"  {provider}:")
            report_lines.append(f"    - Responses: {len(provider_df)}")
            report_lines.append(f"    - Hallucination rate: {provider_rate:.1f}%")
            if avg_time > 0:
                report_lines.append(f"    - Avg processing time: {avg_time:.2f}s")
        
        report_lines.append("")
    
    # Write report
    report_file = output_dir / 'test_summary.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\\n'.join(report_lines))
    
    print(f"📄 Saved summary report to {report_file}")
    
    # Also print to console
    print("\\n" + "\\n".join(report_lines))

def main():
    parser = argparse.ArgumentParser(description='Visualize DoDHaluEval test results')
    parser.add_argument('--results-dir', type=str, default='test_outputs',
                       help='Directory containing test results')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"🔍 Looking for results in: {results_dir}")
    print(f"💾 Saving visualizations to: {output_dir}")
    
    # Find result files
    csv_files = list(results_dir.glob('*.csv'))
    json_files = list(results_dir.glob('*prompts*.json'))
    
    if not csv_files:
        print("❌ No CSV result files found")
        return
    
    # Load most recent results
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
    df = load_results_csv(latest_csv)
    
    prompts = []
    if json_files:
        latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
        prompts = load_prompts_json(latest_json)
    
    print(f"\\n📈 Creating visualizations...")
    
    # Create all visualizations
    create_hallucination_rate_chart(df, output_dir)
    create_processing_time_chart(df, output_dir)
    create_hallucination_types_chart(df, output_dir)
    create_word_count_analysis(df, output_dir)
    create_summary_report(df, prompts, output_dir)
    
    print(f"\\n✅ Visualizations complete! Check {output_dir} for results.")

if __name__ == "__main__":
    main()