# Graph Analysis Tools

This directory contains tools for analyzing the graph data exported by tycho-searcher.

## graph_analyzer.py

A comprehensive Python script that analyzes the `largest_component.json` or `full_graph.json` files.

### Features

- **Basic Statistics**: Node/edge counts, token distribution
- **Pool Analysis**: TVL statistics, pool type distribution  
- **Connectivity Analysis**: Most connected tokens, degree distribution
- **Visualizations**: Degree plots, TVL histograms, pool type charts, graph layouts

### Usage

```bash
# Basic analysis (requires largest_component.json in current directory)
python tools/graph_analyzer.py

# Analyze specific file
python tools/graph_analyzer.py --file full_graph.json

# Skip visualizations
python tools/graph_analyzer.py --no-viz

# Custom output directory
python tools/graph_analyzer.py --output-dir my_plots/
```

### Installation

Install required dependencies:
```bash
pip install matplotlib networkx pandas
```

### Output

The script generates:
1. **Console output**: Detailed statistics about the graph
2. **Plot files** (in `tools/plots/` by default):
   - `degree_distribution.png`: Token connectivity histogram
   - `tvl_distribution.png`: Pool TVL distribution
   - `pool_types.png`: Pool type frequency chart
   - `graph_layout.png`: Visual graph layout (for small graphs)

### Example Output

```
=== Basic Graph Statistics ===
Total nodes (tokens): 156
Total edges (pools): 342
Block number: 21234567

=== Token Distribution ===
Unique token symbols: 143

Top 10 most connected tokens:
  WETH: 89 connections
  USDC: 67 connections
  USDT: 45 connections
  ...

=== Pool Analysis ===
Pool types distribution:
  uniswap_v3: 198 pools
  uniswap_v2: 87 pools
  balancer_v2: 43 pools
  ...

=== TVL Statistics ===
Total TVL: $12,456,789.23
Average TVL: $36,432.01
Max TVL: $1,234,567.89
...
```

This tool helps understand the structure and characteristics of the DEX graph that tycho-searcher is analyzing for arbitrage opportunities.