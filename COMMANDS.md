# Complete Command Reference for Reproducibility

This document contains ALL commands needed to reproduce the Dialog2Flow pipeline results from a fresh clone of the repository.

---

## 🎯 One-Line Complete Setup

```bash
git clone <your-repo-url> && cd dialog2flow && bash SETUP_ENVIRONMENT.sh && source .venv/bin/activate && python3 integrated_pipeline.py --query "escalation issues" --domain "Banking" --distance-threshold 0.4 --formats json graphml html -l -lm llama3:8b
```

---

## 📋 Step-by-Step Commands

### 1. Clone Repository

```bash
# Clone the repository
git clone <your-repo-url>

# Navigate to directory
cd dialog2flow

# Check repository structure
ls -la
```

**Expected output:** You should see files like `integrated_pipeline.py`, `requirements.txt`, `SETUP_ENVIRONMENT.sh`, etc.

---

### 2. System Dependencies (Ubuntu/Debian/WSL)

```bash
# Update package lists
sudo apt-get update

# Install required system packages
sudo apt-get install -y \
  python3 \
  python3-pip \
  python3-venv \
  git \
  curl \
  wget \
  graphviz \
  graphviz-dev \
  build-essential

# Verify Python version (should be 3.10+)
python3 --version
```

**Expected output:** `Python 3.10.x` or higher

---

### 3. Install Ollama

#### Linux/WSL:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service in background
ollama serve &

# Wait for service to start
sleep 5

# Verify Ollama is running
curl -s http://localhost:11434/api/tags > /dev/null && echo "Ollama is running" || echo "Ollama failed to start"

# Pull required model (this downloads ~4.7GB)
ollama pull llama3:8b

# Verify model is downloaded
ollama list
```

**Expected output:**
```
NAME            ID              SIZE      MODIFIED
llama3:8b       365c0bd3c000    4.7 GB    X minutes ago
```

#### macOS:
```bash
# Install Ollama via Homebrew
brew install ollama

# Start Ollama
ollama serve &

# Wait and verify
sleep 5
curl -s http://localhost:11434/api/tags > /dev/null && echo "Ollama is running"

# Pull model
ollama pull llama3:8b
ollama list
```

---

### 4. Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# OR on Windows:
# .venv\Scripts\activate

# Verify activation (you should see (.venv) in prompt)
which python3

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Verify pip version
pip --version
```

**Expected output:** `pip 24.x.x from /path/to/dialog2flow/.venv/lib/python3.xx/site-packages/pip (python 3.xx)`

---

### 5. Install Python Dependencies

```bash
# Install all requirements from requirements.txt
pip install -r requirements.txt

# Install additional critical packages
pip install faiss-cpu
pip install ollama
pip install networkx

# Verify installations
python3 << 'EOF'
import torch
import sentence_transformers
import faiss
import networkx
import ollama
print(f"PyTorch: {torch.__version__}")
print(f"SentenceTransformers: {sentence_transformers.__version__}")
print(f"FAISS: OK")
print(f"NetworkX: {networkx.__version__}")
print(f"Ollama: OK")
EOF
```

**Expected output:**
```
PyTorch: 2.x.x
SentenceTransformers: 5.x.x
FAISS: OK
NetworkX: 3.x
Ollama: OK
```

---

### 6. Download AI Models (Pre-download to avoid delays)

```bash
# Download SentenceTransformer models
python3 << 'EOF'
from sentence_transformers import SentenceTransformer
print("Downloading all-mpnet-base-v2...")
model1 = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
print("Downloaded all-mpnet-base-v2")

print("Downloading dialog2flow-joint-bert-base...")
model2 = SentenceTransformer('sergioburdisso/dialog2flow-joint-bert-base')
print("Downloaded dialog2flow-joint-bert-base")

print("All models ready!")
EOF
```

**Expected output:**
```
Downloading all-mpnet-base-v2...
Downloaded all-mpnet-base-v2
Downloading dialog2flow-joint-bert-base...
Downloaded dialog2flow-joint-bert-base
All models ready!
```

---

### 7. Verify Data File

```bash
# Check if data file exists
ls -lh data/final_json_for_d2f.json

# Count transcripts
python3 -c "import json; print(f\"Transcripts: {len(json.load(open('data/final_json_for_d2f.json')))}\")"

# Show available domains
python3 -c "import json; domains = sorted(set(t.get('domain', 'N/A') for t in json.load(open('data/final_json_for_d2f.json')))); print('Domains:', ', '.join(domains))"
```

**Expected output:**
```
-rw-r--r-- 1 user user 2.5M Dec 1 10:00 data/final_json_for_d2f.json
Transcripts: 18
Domains: Banking, Flight, Hotel, Insurance, Retail, Telecom
```

---

### 8. Run the Pipeline

#### Basic Run (No LLM labels)

```bash
python3 integrated_pipeline.py \
  --query "escalation issues" \
  --domain "Banking" \
  --distance-threshold 0.4 \
  --formats json graphml html
```

#### Full Run (With LLM labels) - RECOMMENDED

```bash
python3 integrated_pipeline.py \
  --query "escalation issues" \
  --domain "Banking" \
  --distance-threshold 0.4 \
  --formats json graphml html \
  -l \
  -lm llama3:8b
```

#### All Options Specified

```bash
python3 integrated_pipeline.py \
  --query "escalation issues" \
  --domain "Banking" \
  --data-path data/final_json_for_d2f.json \
  --top20-dir data/top_K \
  --example-dir data/example \
  --output-dir output \
  --model sergioburdisso/dialog2flow-joint-bert-base \
  --distance-threshold 0.4 \
  --formats json graphml html \
  -l \
  -lm llama3:8b
```

**Expected runtime:** 30-60 seconds (with LLM labels)

**Expected output (last few lines):**
```
✓ Dialog2Flow graph: 42 nodes, 92 edges
✓ Exported to /path/to/dialog2flow/output

================================================================================
PIPELINE COMPLETED SUCCESSFULLY
================================================================================
✓ Top 20 conversations saved to: /path/to/dialog2flow/data/top_K
✓ Dialog2Flow input saved to: /path/to/dialog2flow/data/example
✓ Trajectories and graph saved to: /path/to/dialog2flow/output
```

---

### 9. Verify Results

#### Check Output Files

```bash
# List all output files
ls -lh output/

# Expected files:
# - trajectories_with_metadata.json
# - graph_with_metadata.json
# - graph_with_metadata.graphml
# - graph_visualization.html
# - graph_dialog2flow/ (directory)
```

#### Check Graph Statistics

```bash
# Count nodes and edges
python3 << 'EOF'
import json
with open('output/graph_with_metadata.json') as f:
    data = json.load(f)
print(f"Nodes: {len(data['nodes'])}")
print(f"Edges: {len(data['edges'])}")

# Show first 5 node labels
print("\nFirst 5 node labels:")
for node in data['nodes'][:5]:
    print(f"  {node['id']}: {node.get('label', 'N/A')}")
EOF
```

**Expected output:**
```
Nodes: 40
Edges: 89

First 5 node labels:
  a6: Ca6
  c8: Cc8
  a16: Ca16
  c18: Cc18
  a13: Ca13
```

#### Check LLM Labels

```bash
# Show all cluster labels
python3 << 'EOF'
import json
with open('output/trajectories_with_metadata.json') as f:
    data = json.load(f)
labels = data.get('cluster_labels', {})
print(f"Total cluster labels: {len(labels)}\n")

# Group by speaker
agent_labels = {k: v for k, v in labels.items() if k.startswith('a')}
customer_labels = {k: v for k, v in labels.items() if k.startswith('c')}

print(f"Agent clusters ({len(agent_labels)}):")
for cid in sorted(agent_labels.keys(), key=lambda x: int(x[1:])):
    print(f"  {cid}: {agent_labels[cid]}")

print(f"\nCustomer clusters ({len(customer_labels)}):")
for cid in sorted(customer_labels.keys(), key=lambda x: int(x[1:])):
    print(f"  {cid}: {customer_labels[cid]}")
EOF
```

**Expected output:**
```
Total cluster labels: 40

Agent clusters (21):
  a0: Inform delivery time
  a1: Request clarification or information
  a2: Clarify information
  ...

Customer clusters (19):
  c0: Seek clarification
  c1: Report suspicious transaction
  c2: Seek clarification
  ...
```

---

### 10. View Interactive Visualization

```bash
# Linux
xdg-open output/graph_visualization.html

# macOS
open output/graph_visualization.html

# WSL (Windows Subsystem for Linux)
wslview output/graph_visualization.html

# Or copy path and open manually in browser
realpath output/graph_visualization.html
```

**What you should see:**
- Interactive force-directed graph
- Nodes labeled with cluster IDs (e.g., "Ca6", "Cc8")
- Directed arrows showing conversation flow
- Hover over nodes to see:
  - Full cluster label
  - Utterance examples
  - Metadata (escalation, churn risk, empathy)
  - Source transcripts and turn indices

---

### 11. Test Different Parameters

#### Different Thresholds

```bash
# More clusters (lower threshold)
python3 integrated_pipeline.py \
  --query "escalation issues" \
  --domain "Banking" \
  --distance-threshold 0.3 \
  -l -lm llama3:8b

# Fewer clusters (higher threshold)
python3 integrated_pipeline.py \
  --query "escalation issues" \
  --domain "Banking" \
  --distance-threshold 0.6 \
  -l -lm llama3:8b
```

#### Different Queries

```bash
# Flight delays
python3 integrated_pipeline.py \
  --query "flight delays and cancellations" \
  --domain "Flight" \
  --distance-threshold 0.4 \
  -l -lm llama3:8b

# Hotel complaints
python3 integrated_pipeline.py \
  --query "room service complaints" \
  --domain "Hotel" \
  --distance-threshold 0.4 \
  -l -lm llama3:8b
```

#### Different LLM Models

```bash
# Pull alternative model
ollama pull mistral

# Use mistral for labeling
python3 integrated_pipeline.py \
  --query "escalation issues" \
  --domain "Banking" \
  --distance-threshold 0.4 \
  -l -lm mistral
```

---

### 12. Clean Up (Optional)

```bash
# Remove all generated outputs
rm -rf output/*
rm -rf data/top_K/*
rm -rf data/example/*.txt

# Keep virtual environment and models
# (Next run will be faster)

# To completely reset:
deactivate  # Exit virtual environment
rm -rf .venv  # Remove virtual environment
rm -rf __pycache__  # Remove Python cache
rm -rf __cache__  # Remove LLM cache
```

---

## 🔄 Re-running After Setup

Once setup is complete, subsequent runs only need:

```bash
# Activate virtual environment
source .venv/bin/activate

# Ensure Ollama is running
pgrep -x ollama > /dev/null || (ollama serve &)

# Run pipeline
python3 integrated_pipeline.py \
  --query "escalation issues" \
  --domain "Banking" \
  --distance-threshold 0.4 \
  -l -lm llama3:8b
```

---

## 📊 Expected Results Summary

### For Query: "escalation issues", Domain: "Banking", Threshold: 0.4

| Metric | Expected Value |
|--------|----------------|
| Top conversations found | 3 |
| Total utterances | 115 |
| Agent clusters | 21 |
| Customer clusters | 19 |
| Total clusters | 40 |
| Graph nodes | 40 |
| Graph edges | 89 |
| LLM labels generated | 40 |
| Processing time | 30-60 seconds |

### Sample Output Files Sizes

| File | Approximate Size |
|------|------------------|
| trajectories_with_metadata.json | ~200-500 KB |
| graph_with_metadata.json | ~150-300 KB |
| graph_with_metadata.graphml | ~50-100 KB |
| graph_visualization.html | ~100-150 KB |

---

## 🐛 Common Issues and Fixes

### Issue: "Ollama connection refused"

```bash
# Check if Ollama is running
pgrep -x ollama

# If not running, start it:
ollama serve &

# Wait and retry
sleep 3
python3 integrated_pipeline.py ...
```

### Issue: "No module named 'sentence_transformers'"

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install sentence-transformers
```

### Issue: "CUDA out of memory"

```bash
# Use CPU version of FAISS
pip uninstall faiss-gpu
pip install faiss-cpu
```

### Issue: Models downloading during pipeline run

```bash
# Pre-download models (run once)
python3 << 'EOF'
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
SentenceTransformer('sergioburdisso/dialog2flow-joint-bert-base')
EOF
```

---

## ✅ Verification Checklist

Before reporting issues, verify:

- [ ] Python 3.10+ installed: `python3 --version`
- [ ] Virtual environment activated: `which python3` shows `.venv`
- [ ] Ollama running: `curl -s http://localhost:11434/api/tags`
- [ ] llama3:8b downloaded: `ollama list | grep llama3:8b`
- [ ] All pip packages installed: `pip list | grep -E "(torch|faiss|networkx|sentence)"`
- [ ] Data file exists: `ls data/final_json_for_d2f.json`
- [ ] No previous errors in logs

---

## 📦 Complete Dependency List

### System Dependencies
- python3 (3.10+)
- pip3
- git
- curl
- graphviz (optional, for PNG export)

### Python Packages
- torch
- sentence-transformers
- faiss-cpu (or faiss-gpu)
- networkx
- ollama
- numpy
- pandas
- scikit-learn
- matplotlib
- tqdm
- (see requirements.txt for complete list)

### External Services
- Ollama (local LLM server)
- llama3:8b model (~4.7GB)

---

## 🎯 Success Criteria

You've successfully reproduced the results if:

1. ✅ Pipeline runs without errors
2. ✅ Output directory contains all expected files
3. ✅ Graph has 40 nodes and 89 edges (for threshold 0.4)
4. ✅ All 40 clusters have LLM-generated labels
5. ✅ HTML visualization opens and shows directed arrows
6. ✅ Nodes display metadata on hover
7. ✅ Graph is genuinely directed (verified in code)

---

**Last Updated:** December 1, 2025
**Tested On:** Ubuntu 22.04, Python 3.13.9, Ollama 0.13.0
