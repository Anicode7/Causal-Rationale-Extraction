#!/bin/bash
# Quick script to run the integrated pipeline

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         INTEGRATED PIPELINE: TOP 20 → DIALOG2FLOW             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Running pipeline with default settings..."
echo "Query: 'flight delay compensation'"
echo "Domain: Flight"
echo ""

python integrated_pipeline.py \
  --query "flight delay compensation" \
  --domain Flight

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    PIPELINE COMPLETE!                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Output files created:"
echo "  📁 data/top_20/*.txt - Top 20 conversations with metadata"
echo "  📁 data/example/*.txt - Dialog2Flow format"
echo "  📁 output/trajectories_with_metadata.json - Trajectories"
echo "  📁 output/graph_with_metadata.json - Graph data"
echo "  🌐 output/graph_visualization.html - Interactive visualization"
echo ""
echo "Next steps:"
echo "  1. Open visualization: firefox output/graph_visualization.html"
echo "  2. Inspect metadata: cat output/graph_with_metadata.json | jq '.nodes[0]'"
echo "  3. Run test: python test_integrated_pipeline.py"
echo ""
