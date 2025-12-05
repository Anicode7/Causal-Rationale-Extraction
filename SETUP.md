# Dialog2Flow Pipeline - Complete Setup Guide

This guide provides step-by-step instructions to reproduce the Dialog2Flow pipeline results from scratch.

---

## 📋 Prerequisites

- **OS**: Linux (Ubuntu/WSL recommended) or macOS
- **Python**: 3.10+ (tested with Python 3.13.9)
- **GPU**: NVIDIA GPU with CUDA 12.1+ (recommended for faster processing)
- **RAM**: 16GB+ recommended
- **Disk Space**: ~10GB for models and data

---

## 🚀 Quick Start (For Reproducing Results)

If you just cloned this repo and want to get the same results:

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd dialog2flow

# 2. Run the automated setup script
bash SETUP_ENVIRONMENT.sh

# 3. Run the pipeline
python3 integrated_pipeline.py \
  --query "escalation issues" \
  --domain "Banking" \
  --distance-threshold 0.4 \
  --formats json graphml html \
  -l -lm llama3:8b
```

**That's it!** The script handles everything automatically.

---

## 📦 Step-by-Step Manual Setup

### Step 1: System Dependencies

#### For Ubuntu/Debian/WSL:
```bash
sudo apt-get update
sudo apt-get install -y \
  python3 python3-pip python3-venv \
  git curl wget \
  graphviz graphviz-dev \
  build-essential
```

#### For macOS:
```bash
brew install python git graphviz
```

---

### Step 2: Install Ollama (for LLM-based cluster labeling)

#### Linux/WSL:
```bash
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve &

# Pull the required model
ollama pull llama3:8b
```

#### macOS:
```bash
# Download from https://ollama.com/download
# Or use brew:
brew install ollama

# Start Ollama
ollama serve &

# Pull the model
ollama pull llama3:8b
```

**Verify Ollama is running:**
```bash
ollama --version
ollama list  # Should show llama3:8b
```

---

### Step 3: Python Environment Setup

#### Option A: Using venv (Recommended)

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate  # Linux/macOS
# OR on Windows:
# .venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Option B: Using Conda

```bash
# Create conda environment
conda create -n dialog2flow python=3.10 -y
conda activate dialog2flow

# Install PyTorch with CUDA (if you have GPU)
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

# OR CPU-only version:
# conda install pytorch cpuonly -c pytorch -y
```

---

### Step 4: Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Install additional required packages
pip install faiss-cpu  # or faiss-gpu if you have CUDA
pip install ollama networkx
```

**Verify installation:**
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import sentence_transformers; print('SentenceTransformers: OK')"
python3 -c "import faiss; print('FAISS: OK')"
python3 -c "import networkx; print('NetworkX: OK')"
python3 -c "import ollama; print('Ollama: OK')"
```

---

### Step 5: Verify Data Files

Ensure you have the required data file:

```bash
# Check if data file exists
ls -lh data/final_json_for_d2f.json

# Should be ~1-10MB depending on your dataset
# If missing, you need to provide your conversation data in this format:
# [
#   {
#     "transcript_id": "...",
#     "domain": "Banking",
#     "intent": "...",
#     "reason_for_call": "...",
#     "conversation": [
#       {
#         "speaker": "Agent",
#         "text": "...",
#         "escalation_level": 0.1,
#         "churn_risk_score": 0.2,
#         ...
#       }
#     ]
#   }
# ]
```

---

## 🎯 Running the Pipeline

### Basic Usage

```bash
python3 integrated_pipeline.py \
  --query "escalation issues" \
  --domain "Banking" \
  --distance-threshold 0.4 \
  --formats json graphml html
```

### With LLM Cluster Labeling (Recommended)

```bash
python3 integrated_pipeline.py \
  --query "escalation issues" \
  --domain "Banking" \
  --distance-threshold 0.4 \
  --formats json graphml html \
  -l \
  -lm llama3:8b
```

### All Available Options

```bash
python3 integrated_pipeline.py \
  --query "your search query" \
  --domain "Banking" \                    # Optional: Banking, Flight, Hotel, etc.
  --distance-threshold 0.4 \              # Clustering threshold (0.3-0.6)
  --formats json graphml html \           # Output formats
  --model sergioburdisso/dialog2flow-joint-bert-base \  # Embedding model
  -l \                                    # Enable LLM labels
  -lm llama3:8b \                         # LLM model name
  --output-dir ./output                   # Output directory
```

---

## 📊 Expected Output

After running the pipeline, you should see:

```
dialog2flow/
├── data/
│   ├── top_K/                         # Top 20 conversations found
│   │   ├── top_K_banking_*.json       # Summary JSON
│   │   └── *.txt                       # Individual transcripts
│   └── example/
│       ├── *.txt                       # Dialog2Flow format
│       └── conversations_metadata.json # Metadata
├── output/
│   ├── trajectories_with_metadata.json # Extracted trajectories
│   ├── graph_with_metadata.json        # Graph (JSON format)
│   ├── graph_with_metadata.graphml     # Graph (GraphML format)
│   ├── graph_visualization.html        # Interactive visualization ⭐
│   └── graph_dialog2flow/
│       ├── graph.graphml               # Dialog2Flow graph
│       ├── graph.png                   # PNG visualization
│       └── visualization/graph.html    # Interactive HTML
```

---

## 🔍 Verification Steps

### 1. Check Pipeline Output

```bash
# List output files
ls -lh output/

# Check graph statistics
python3 -c "
import json
with open('output/graph_with_metadata.json') as f:
    data = json.load(f)
print(f\"Nodes: {len(data['nodes'])}\")
print(f\"Edges: {len(data['edges'])}\")
"

# Check if LLM labels were generated
python3 -c "
import json
with open('output/trajectories_with_metadata.json') as f:
    data = json.load(f)
labels = data.get('cluster_labels', {})
print(f\"Cluster labels generated: {len(labels)}\")
for cid, label in list(labels.items())[:5]:
    print(f\"  {cid}: {label}\")
"
```

### 2. Open Visualization

```bash
# On Linux with browser
xdg-open output/graph_visualization.html

# On macOS
open output/graph_visualization.html

# On WSL (Windows Subsystem for Linux)
wslview output/graph_visualization.html

# Or manually: Copy path and open in browser
realpath output/graph_visualization.html
```

### 3. Verify Results Match Expected Output

**Expected Results for "escalation issues" query on Banking domain:**

- **Top 20 Conversations**: 3-20 transcripts (depending on data)
- **Clusters**: ~40 clusters (with threshold 0.4)
  - ~21 Agent clusters (prefix 'a')
  - ~19 Customer clusters (prefix 'c')
- **LLM Labels**: All 40 clusters should have meaningful labels
  - Example: "Agent: Identify self and request assistance"
  - Example: "Customer: Compare financial services"
- **Graph**: 
  - Directed graph with ~40 nodes, ~89 edges
  - HTML visualization with arrows showing directionality
  - Nodes sized by number of utterances
  - Hover shows full metadata

---

## 🐛 Troubleshooting

### Issue: "Ollama connection failed"

```bash
# Check if Ollama is running
ps aux | grep ollama

# If not running, start it:
ollama serve &

# Wait a few seconds, then verify:
ollama list
```

### Issue: "CUDA out of memory"

```bash
# Use CPU-only version
pip uninstall faiss-gpu
pip install faiss-cpu

# Or reduce batch size in code (not recommended)
```

### Issue: "No module named 'sentence_transformers'"

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
pip install sentence-transformers
```

### Issue: "No transcripts found for domain 'Banking'"

```bash
# Check available domains in your data
python3 -c "
import json
with open('data/final_json_for_d2f.json') as f:
    data = json.load(f)
domains = set(t.get('domain', '') for t in data)
print('Available domains:', sorted(domains))
"

# Use one of the available domains
```

### Issue: Models downloading slowly

```bash
# Pre-download models (optional)
python3 -c "
from sentence_transformers import SentenceTransformer
print('Downloading all-mpnet-base-v2...')
SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
print('Downloading dialog2flow-joint-bert-base...')
SentenceTransformer('sergioburdisso/dialog2flow-joint-bert-base')
print('Done!')
"
```

---

## 📝 Different Queries and Domains

### Example: Flight Delays
```bash
python3 integrated_pipeline.py \
  --query "flight delays and cancellations" \
  --domain "Flight" \
  --distance-threshold 0.4 \
  -l -lm llama3:8b
```

### Example: Hotel Complaints
```bash
python3 integrated_pipeline.py \
  --query "room service complaints" \
  --domain "Hotel" \
  --distance-threshold 0.5 \
  -l -lm llama3:8b
```

### Example: Different Threshold
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

---

## 🔧 Advanced Configuration

### Using Different LLM Models

```bash
# List available Ollama models
ollama list

# Pull additional models
ollama pull mistral
ollama pull gemma:7b

# Use in pipeline
python3 integrated_pipeline.py \
  --query "escalation issues" \
  --domain "Banking" \
  -l -lm mistral
```

### Custom Output Directory

```bash
python3 integrated_pipeline.py \
  --query "escalation issues" \
  --domain "Banking" \
  --output-dir ./custom_output \
  -l -lm llama3:8b
```

### Skip Cleaning Previous Results

```bash
python3 integrated_pipeline.py \
  --query "escalation issues" \
  --domain "Banking" \
  --no-clean  # Keeps previous output
```

---

## 📚 Core Pipeline Files

The pipeline uses these files (all others can be ignored):

```
dialog2flow/
├── integrated_pipeline.py                    # Main orchestrator
├── find_top_K_conversations.py              # Step 1: Top 20 finder
├── prepare_top20_for_dialog2flow.py          # Step 2: Data prep
├── extract_trajectories_with_metadata.py     # Step 3: Clustering
├── build_graph_with_metadata.py              # Step 4a: Metadata graph
├── build_graph_dialog2flow_format.py         # Step 4b: D2F graph
├── build_graph.py                            # Original D2F builder
├── requirements.txt                          # Dependencies
└── data/
    └── final_json_for_d2f.json               # Your data
```

---

## 🎓 Understanding the Pipeline

### What Each Step Does

1. **Step 1: Find Top 20** (`find_top_K_conversations.py`)
   - Loads all conversations from JSON
   - Filters by domain (e.g., Banking)
   - Creates embeddings for all utterances
   - Builds FAISS index for fast search
   - Finds top 20 most relevant to your query
   - Saves to `data/top_K/`

2. **Step 2: Prepare** (`prepare_top20_for_dialog2flow.py`)
   - Converts top 20 to Dialog2Flow text format
   - Preserves all metadata
   - Saves to `data/example/`

3. **Step 3: Extract Trajectories** (`extract_trajectories_with_metadata.py`)
   - Loads conversations
   - Creates embeddings with Dialog2Flow model
   - Clusters Agent and Customer utterances separately
   - Generates LLM labels for each cluster (if enabled)
   - Aggregates metadata per cluster
   - Saves to `output/trajectories_with_metadata.json`

4. **Step 4: Build Graphs** (`build_graph_with_metadata.py` + `build_graph_dialog2flow_format.py`)
   - Builds directed graph from trajectories
   - Attaches metadata to nodes
   - Exports as JSON, GraphML, HTML
   - Creates Dialog2Flow action flow graph
   - Generates interactive visualizations

---

## 📖 Citation

If you use this pipeline, please cite:

```bibtex
@article{burdisso2022dialog2flow,
  title={Dialog2Flow: Pre-training Soft-Contrastive Action-Driven Sentence Embeddings for Automatic Dialog Flow Extraction},
  author={Burdisso, Sergio and Madikeri, Srikanth and Motlicek, Petr},
  journal={arXiv preprint arXiv:2206.07148},
  year={2022}
}
```

---

## 📞 Support

If you encounter issues:

1. Check the Troubleshooting section above
2. Verify all dependencies are installed: `pip list`
3. Check Ollama is running: `ollama list`
4. Review the log output for specific error messages
5. Open an issue on GitHub with:
   - Your Python version: `python3 --version`
   - Your OS: `uname -a` (Linux/macOS) or `ver` (Windows)
   - Full error message
   - Command you ran

---

## ✅ Success Checklist

- [ ] Python 3.10+ installed
- [ ] Ollama installed and running
- [ ] `llama3:8b` model pulled
- [ ] Virtual environment created and activated
- [ ] All pip dependencies installed
- [ ] Data file exists at `data/final_json_for_d2f.json`
- [ ] Pipeline runs without errors
- [ ] Output files created in `output/` directory
- [ ] HTML visualization opens in browser
- [ ] Graph shows directed arrows
- [ ] All clusters have LLM-generated labels

**If all boxes are checked, you're ready to reproduce the results!** 🎉
