#!/usr/bin/env python3
"""
Plot comprehensive benchmark results comparing PD and PDV across different configurations.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def load_results(results_dir):
    """Load all benchmark results."""
    results = {}
    results_path = Path(results_dir)
    
    for summary_file in results_path.glob("*_summary.csv"):
        config_name = summary_file.stem.replace("_summary", "")
        df = pd.read_csv(summary_file)
        results[config_name] = df
        print(f"Loaded: {config_name} ({len(df)} rows)")
    
    return results

def plot_throughput_comparison(results, output_dir):
    """Plot throughput comparison across configurations."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (config_name, df) in enumerate(results.items()):
        ax = axes[idx]
        
        # Separate PD and PDV data
        pd_data = df[df['Engine'] == 'PD']
        pdv_data = df[df['Engine'] == 'PDV']
        
        # Plot
        ax.plot(pd_data['Concurrency'], pd_data['Throughput (TPS)'], 
                marker='o', linewidth=2, markersize=8, label='PD', color='#1f77b4')
        ax.plot(pdv_data['Concurrency'], pdv_data['Throughput (TPS)'], 
                marker='s', linewidth=2, markersize=8, label='PDV', color='#ff7f0e')
        
        ax.set_xlabel('Concurrency Level', fontsize=11, fontweight='bold')
        ax.set_ylabel('Throughput (TPS)', fontsize=11, fontweight='bold')
        ax.set_title(f'{config_name}\nThroughput vs Concurrency', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'throughput_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_latency_comparison(results, output_dir):
    """Plot latency comparison across configurations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for idx, (config_name, df) in enumerate(results.items()):
        # Average latency
        ax_avg = axes[0, idx]
        pd_data = df[df['Engine'] == 'PD']
        pdv_data = df[df['Engine'] == 'PDV']
        
        ax_avg.plot(pd_data['Concurrency'], pd_data['Avg Request Latency (ms)'], 
                   marker='o', linewidth=2, markersize=8, label='PD', color='#1f77b4')
        ax_avg.plot(pdv_data['Concurrency'], pdv_data['Avg Request Latency (ms)'], 
                   marker='s', linewidth=2, markersize=8, label='PDV', color='#ff7f0e')
        
        ax_avg.set_xlabel('Concurrency Level', fontsize=11, fontweight='bold')
        ax_avg.set_ylabel('Avg Latency (ms)', fontsize=11, fontweight='bold')
        ax_avg.set_title(f'{config_name}\nAverage Latency', fontsize=12, fontweight='bold')
        ax_avg.legend(loc='best', fontsize=10)
        ax_avg.grid(True, alpha=0.3)
        ax_avg.set_xscale('log', base=2)
        ax_avg.set_yscale('log')
        
        # P99 latency
        ax_p99 = axes[1, idx]
        ax_p99.plot(pd_data['Concurrency'], pd_data['Latency P99 (ms)'], 
                   marker='o', linewidth=2, markersize=8, label='PD', color='#1f77b4')
        ax_p99.plot(pdv_data['Concurrency'], pdv_data['Latency P99 (ms)'], 
                   marker='s', linewidth=2, markersize=8, label='PDV', color='#ff7f0e')
        
        ax_p99.set_xlabel('Concurrency Level', fontsize=11, fontweight='bold')
        ax_p99.set_ylabel('P99 Latency (ms)', fontsize=11, fontweight='bold')
        ax_p99.set_title(f'{config_name}\nP99 Latency', fontsize=12, fontweight='bold')
        ax_p99.legend(loc='best', fontsize=10)
        ax_p99.grid(True, alpha=0.3)
        ax_p99.set_xscale('log', base=2)
        ax_p99.set_yscale('log')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'latency_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_gpu_utilization(results, output_dir):
    """Plot GPU utilization comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (config_name, df) in enumerate(results.items()):
        ax = axes[idx]
        
        pd_data = df[df['Engine'] == 'PD']
        pdv_data = df[df['Engine'] == 'PDV']
        
        ax.plot(pd_data['Concurrency'], pd_data['GPU Util %'], 
                marker='o', linewidth=2, markersize=8, label='PD', color='#1f77b4')
        ax.plot(pdv_data['Concurrency'], pdv_data['GPU Util %'], 
                marker='s', linewidth=2, markersize=8, label='PDV', color='#ff7f0e')
        
        ax.set_xlabel('Concurrency Level', fontsize=11, fontweight='bold')
        ax.set_ylabel('GPU Utilization (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{config_name}\nGPU Utilization', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_ylim([0, 100])
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'gpu_utilization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_acceptance_rate(results, output_dir):
    """Plot acceptance rate comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (config_name, df) in enumerate(results.items()):
        ax = axes[idx]
        
        pd_data = df[df['Engine'] == 'PD']
        pdv_data = df[df['Engine'] == 'PDV']
        
        # Convert percentage strings to float
        pd_accept = pd_data['Acceptance Rate'].str.rstrip('%').astype(float)
        pdv_accept = pdv_data['Acceptance Rate'].str.rstrip('%').astype(float)
        
        ax.plot(pd_data['Concurrency'], pd_accept, 
                marker='o', linewidth=2, markersize=8, label='PD', color='#1f77b4')
        ax.plot(pdv_data['Concurrency'], pdv_accept, 
                marker='s', linewidth=2, markersize=8, label='PDV', color='#ff7f0e')
        
        ax.set_xlabel('Concurrency Level', fontsize=11, fontweight='bold')
        ax.set_ylabel('Acceptance Rate (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{config_name}\nToken Acceptance Rate', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'acceptance_rate.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_improvement_heatmap(results, output_dir):
    """Plot heatmap showing PDV improvement over PD."""
    configs = list(results.keys())
    concurrency_levels = sorted(results[configs[0]]['Concurrency'].unique())
    
    # Calculate improvements for throughput and latency
    throughput_improvements = np.zeros((len(configs), len(concurrency_levels)))
    latency_improvements = np.zeros((len(configs), len(concurrency_levels)))
    
    for i, config_name in enumerate(configs):
        df = results[config_name]
        for j, concurrency in enumerate(concurrency_levels):
            pd_row = df[(df['Engine'] == 'PD') & (df['Concurrency'] == concurrency)]
            pdv_row = df[(df['Engine'] == 'PDV') & (df['Concurrency'] == concurrency)]
            
            if not pd_row.empty and not pdv_row.empty:
                pd_tps = pd_row['Throughput (TPS)'].values[0]
                pdv_tps = pdv_row['Throughput (TPS)'].values[0]
                throughput_improvements[i, j] = ((pdv_tps - pd_tps) / pd_tps) * 100
                
                pd_lat = pd_row['Avg Request Latency (ms)'].values[0]
                pdv_lat = pdv_row['Avg Request Latency (ms)'].values[0]
                latency_improvements[i, j] = ((pd_lat - pdv_lat) / pd_lat) * 100
    
    # Plot heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Throughput improvement
    im1 = axes[0].imshow(throughput_improvements, cmap='RdYlGn', aspect='auto', vmin=-30, vmax=60)
    axes[0].set_xticks(range(len(concurrency_levels)))
    axes[0].set_xticklabels(concurrency_levels)
    axes[0].set_yticks(range(len(configs)))
    axes[0].set_yticklabels(configs)
    axes[0].set_xlabel('Concurrency Level', fontsize=11, fontweight='bold')
    axes[0].set_title('PDV Throughput Improvement over PD (%)', fontsize=12, fontweight='bold')
    
    for i in range(len(configs)):
        for j in range(len(concurrency_levels)):
            text = axes[0].text(j, i, f'{throughput_improvements[i, j]:.1f}%',
                              ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im1, ax=axes[0], label='Improvement (%)')
    
    # Latency improvement
    im2 = axes[1].imshow(latency_improvements, cmap='RdYlGn', aspect='auto', vmin=-30, vmax=30)
    axes[1].set_xticks(range(len(concurrency_levels)))
    axes[1].set_xticklabels(concurrency_levels)
    axes[1].set_yticks(range(len(configs)))
    axes[1].set_yticklabels(configs)
    axes[1].set_xlabel('Concurrency Level', fontsize=11, fontweight='bold')
    axes[1].set_title('PDV Latency Reduction over PD (%)', fontsize=12, fontweight='bold')
    
    for i in range(len(configs)):
        for j in range(len(concurrency_levels)):
            text = axes[1].text(j, i, f'{latency_improvements[i, j]:.1f}%',
                              ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im2, ax=axes[1], label='Improvement (%)')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'improvement_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def generate_summary_table(results, output_dir):
    """Generate summary statistics table."""
    summary_data = []
    
    for config_name, df in results.items():
        for engine in ['PD', 'PDV']:
            engine_df = df[df['Engine'] == engine]
            
            summary_data.append({
                'Configuration': config_name,
                'Engine': engine,
                'Avg Throughput (TPS)': engine_df['Throughput (TPS)'].mean(),
                'Max Throughput (TPS)': engine_df['Throughput (TPS)'].max(),
                'Avg Latency (ms)': engine_df['Avg Request Latency (ms)'].mean(),
                'Min Latency (ms)': engine_df['Avg Request Latency (ms)'].min(),
                'Avg GPU Util (%)': engine_df['GPU Util %'].mean(),
                'Avg Acceptance Rate (%)': engine_df['Acceptance Rate'].str.rstrip('%').astype(float).mean()
            })
    
    summary_df = pd.DataFrame(summary_data)
    output_path = Path(output_dir) / 'summary_statistics.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    return summary_df

def main():
    results_dir = Path("benchmark_results")
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("PLOTTING BENCHMARK RESULTS")
    print("=" * 80)
    
    # Load results
    results = load_results(results_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"\nGenerating plots for {len(results)} configurations...")
    
    # Generate all plots
    plot_throughput_comparison(results, plots_dir)
    plot_latency_comparison(results, plots_dir)
    plot_gpu_utilization(results, plots_dir)
    plot_acceptance_rate(results, plots_dir)
    plot_improvement_heatmap(results, plots_dir)
    
    # Generate summary table
    summary_df = generate_summary_table(results, plots_dir)
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("ALL PLOTS GENERATED")
    print("=" * 80)
    print(f"Plots saved in: {plots_dir.absolute()}")

if __name__ == "__main__":
    main()

