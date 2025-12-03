#!/bin/bash

###############################################################################
# Quick Test Script - Verify Pipeline Works
#
# This script runs a quick test to verify the pipeline is working correctly.
# Run this after setup to ensure everything is configured properly.
#
# Usage:
#   bash TEST_PIPELINE.sh
###############################################################################

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Dialog2Flow Pipeline - Quick Test${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}✗ Virtual environment not activated${NC}"
    echo "Please run: source .venv/bin/activate"
    exit 1
fi
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Check Ollama
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${RED}✗ Ollama not running${NC}"
    echo "Please start Ollama: ollama serve &"
    exit 1
fi
echo -e "${GREEN}✓ Ollama is running${NC}"

# Check data file
if [ ! -f "data/final_json_for_d2f.json" ]; then
    echo -e "${RED}✗ Data file not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Data file exists${NC}"

# Run pipeline with minimal test
echo -e "\n${BLUE}Running pipeline test...${NC}\n"

python3 integrated_pipeline.py \
  --query "escalation issues" \
  --domain "Banking" \
  --distance-threshold 0.4 \
  --formats json \
  -l -lm llama3:8b

# Check outputs
echo -e "\n${BLUE}Verifying outputs...${NC}\n"

if [ -f "output/trajectories_with_metadata.json" ]; then
    echo -e "${GREEN}✓ Trajectories file created${NC}"
else
    echo -e "${RED}✗ Trajectories file missing${NC}"
    exit 1
fi

if [ -f "output/graph_with_metadata.json" ]; then
    echo -e "${GREEN}✓ Graph file created${NC}"
else
    echo -e "${RED}✗ Graph file missing${NC}"
    exit 1
fi

# Check statistics
python3 << 'EOF'
import json
import sys

try:
    with open('output/graph_with_metadata.json') as f:
        data = json.load(f)
    
    nodes = len(data['nodes'])
    edges = len(data['edges'])
    
    print(f"\n✓ Graph has {nodes} nodes and {edges} edges")
    
    with open('output/trajectories_with_metadata.json') as f:
        traj = json.load(f)
    
    labels = traj.get('cluster_labels', {})
    print(f"✓ Generated {len(labels)} cluster labels")
    
    if len(labels) > 0:
        print("\nSample labels:")
        for cid, label in list(labels.items())[:3]:
            print(f"  {cid}: {label}")
    
except Exception as e:
    print(f"\n✗ Error checking outputs: {e}")
    sys.exit(1)
EOF

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo "Pipeline is working correctly!"
echo "You can now run the full pipeline with HTML visualization:"
echo ""
echo "  python3 integrated_pipeline.py \\"
echo "    --query \"escalation issues\" \\"
echo "    --domain \"Banking\" \\"
echo "    --distance-threshold 0.4 \\"
echo "    --formats json graphml html \\"
echo "    -l -lm llama3:8b"
echo ""
