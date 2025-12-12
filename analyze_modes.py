#!/usr/bin/env python3
"""
Analyze success and failure modes of PDV vs PD across different configurations.
"""
import pandas as pd
from pathlib import Path
import numpy as np

def analyze_modes(results_dir):
    """Analyze success and failure modes."""
    results_path = Path(results_dir)
    
    print("=" * 80)
    print("FAILURE AND SUCCESS MODE ANALYSIS")
    print("=" * 80)
    
    success_modes = []
    failure_modes = []
    
    for summary_file in results_path.glob("*_summary.csv"):
        config_name = summary_file.stem.replace("_summary", "")
        df = pd.read_csv(summary_file)
        
        print(f"\n{config_name}:")
        print("-" * 80)
        
        for concurrency in sorted(df['Concurrency'].unique()):
            pd_row = df[(df['Engine'] == 'PD') & (df['Concurrency'] == concurrency)]
            pdv_row = df[(df['Engine'] == 'PDV') & (df['Concurrency'] == concurrency)]
            
            if pd_row.empty or pdv_row.empty:
                continue
            
            pd_tps = pd_row['Throughput (TPS)'].values[0]
            pdv_tps = pdv_row['Throughput (TPS)'].values[0]
            tps_improvement = ((pdv_tps - pd_tps) / pd_tps) * 100
            
            pd_lat = pd_row['Avg Request Latency (ms)'].values[0]
            pdv_lat = pdv_row['Avg Request Latency (ms)'].values[0]
            lat_improvement = ((pd_lat - pdv_lat) / pd_lat) * 100
            
            pd_gpu = pd_row['GPU Util %'].values[0]
            pdv_gpu = pdv_row['GPU Util %'].values[0]
            
            status = ""
            if tps_improvement > 10 and lat_improvement > 5:
                status = "MAJOR SUCCESS"
                success_modes.append({
                    'Config': config_name,
                    'Concurrency': concurrency,
                    'TPS Improvement': f"{tps_improvement:.1f}%",
                    'Latency Improvement': f"{lat_improvement:.1f}%",
                    'PD GPU Util': f"{pd_gpu:.1f}%",
                    'PDV GPU Util': f"{pdv_gpu:.1f}%",
                    'Reason': 'High concurrency + efficient parallelization'
                })
            elif tps_improvement > 5 or lat_improvement > 5:
                status = "SUCCESS"
                success_modes.append({
                    'Config': config_name,
                    'Concurrency': concurrency,
                    'TPS Improvement': f"{tps_improvement:.1f}%",
                    'Latency Improvement': f"{lat_improvement:.1f}%",
                    'PD GPU Util': f"{pd_gpu:.1f}%",
                    'PDV GPU Util': f"{pdv_gpu:.1f}%",
                    'Reason': 'Moderate improvement'
                })
            elif tps_improvement < -5 or lat_improvement < -5:
                status = "FAILURE"
                failure_modes.append({
                    'Config': config_name,
                    'Concurrency': concurrency,
                    'TPS Degradation': f"{tps_improvement:.1f}%",
                    'Latency Degradation': f"{lat_improvement:.1f}%",
                    'PD GPU Util': f"{pd_gpu:.1f}%",
                    'PDV GPU Util': f"{pdv_gpu:.1f}%",
                    'Reason': 'Overhead exceeds parallelization benefits'
                })
            else:
                status = "NEUTRAL"
            
            print(f"  C={concurrency:2d}: TPS {tps_improvement:+6.1f}%, "
                  f"Latency {lat_improvement:+6.1f}% [{status}]")
    
    print("\n" + "=" * 80)
    print("SUCCESS MODES (PDV outperforms PD)")
    print("=" * 80)
    if success_modes:
        success_df = pd.DataFrame(success_modes)
        print(success_df.to_string(index=False))
    else:
        print("No clear success modes identified.")
    
    print("\n" + "=" * 80)
    print("FAILURE MODES (PDV underperforms PD)")
    print("=" * 80)
    if failure_modes:
        failure_df = pd.DataFrame(failure_modes)
        print(failure_df.to_string(index=False))
    else:
        print("No significant failure modes identified.")
    
    # Save to files
    if success_modes:
        success_df = pd.DataFrame(success_modes)
        success_df.to_csv(results_path / 'success_modes.csv', index=False)
        print(f"\nSaved: {results_path / 'success_modes.csv'}")
    
    if failure_modes:
        failure_df = pd.DataFrame(failure_modes)
        failure_df.to_csv(results_path / 'failure_modes.csv', index=False)
        print(f"Saved: {results_path / 'failure_modes.csv'}")
    
    return success_modes, failure_modes

def generate_recommendations(success_modes, failure_modes):
    """Generate deployment recommendations."""
    print("\n" + "=" * 80)
    print("DEPLOYMENT RECOMMENDATIONS")
    print("=" * 80)
    
    if success_modes:
        print("\nPDV is recommended for:")
        for mode in success_modes:
            print(f"  - {mode['Config']}, Concurrency >= {mode['Concurrency']}")
            print(f"    ({mode['Reason']})")
    
    if failure_modes:
        print("\nPD is recommended for:")
        for mode in failure_modes:
            print(f"  - {mode['Config']}, Concurrency = {mode['Concurrency']}")
            print(f"    ({mode['Reason']})")
    
    print("\nGeneral Guidelines:")
    print("  1. Use PDV for high-concurrency workloads (typically C >= 16)")
    print("  2. Use PDV when verifier model is computationally heavy")
    print("  3. Use PD for low-concurrency, latency-sensitive workloads (C < 8)")
    print("  4. Monitor GPU utilization - PDV shines when GPU is well-utilized")

if __name__ == "__main__":
    results_dir = Path("benchmark_results")
    success_modes, failure_modes = analyze_modes(results_dir)
    generate_recommendations(success_modes, failure_modes)

