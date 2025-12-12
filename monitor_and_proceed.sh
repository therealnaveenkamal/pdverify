#!/bin/bash
# Monitor benchmark completion and automatically proceed with analysis

echo "Monitoring benchmark progress..."
echo "Started at: $(date)"

# Wait for benchmark to complete
while ps aux | grep -q "[r]un_comprehensive_analysis.py"; do
    sleep 30
    echo "$(date): Still running..."
done

echo "$(date): Benchmarks completed!"

# Check if results exist
if [ -d "benchmark_results" ] && [ "$(ls -A benchmark_results)" ]; then
    echo "Results found. Proceeding with analysis..."
    
    # Run plotting
    python plot_results.py
    
    # Run mode analysis
    python analyze_modes.py
    
    echo "Analysis complete!"
else
    echo "No results found!"
    exit 1
fi

