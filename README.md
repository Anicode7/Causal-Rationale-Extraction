# Dialog2Flow: Metadata-Enhanced Conversation Flow Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Automated pipeline for extracting and visualizing conversation flows with metadata from customer service dialogues.**

This repository extends the original [Dialog2Flow](https://github.com/sergioburdisso/dialog2flow) methodology with:
- ✅ Semantic search to find relevant conversations
- ✅ Metadata preservation (escalation levels, churn risk, empathy scores, etc.)
- ✅ Speaker-separated clustering (Agent vs Customer)
- ✅ LLM-based cluster labeling using Ollama
- ✅ Interactive directed graph visualizations
- ✅ Comprehensive metadata tracking at utterance and cluster levels

---

## 🚀 Quick Start (3 Steps)

```bash
# 1. Clone and navigate
git clone <your-repo-url>
cd dialog2flow

# 2. Run automated setup
bash SETUP_ENVIRONMENT.sh

# 3. Run pipeline
python3 integrated_pipeline.py \
  --query "escalation issues" \
  --domain "Banking" \
  --distance-threshold 0.4 \
  --formats json graphml html \
  -l -lm llama3:8b
```

**Results:** Open `output/graph_visualization.html` in your browser to see the interactive graph!

---

## 📋 What This Pipeline Does

1. **Find Top Conversations** → Semantic search across your dataset
2. **Prepare Data** → Convert to Dialog2Flow format with metadata
3. **Extract Trajectories** → Cluster utterances (Agent & Customer separately)
4. **Generate Labels** → Use LLM to name each cluster
5. **Build Graphs** → Create directed flow graphs with metadata
6. **Visualize** → Interactive HTML graphs with hover tooltips

---

## 📊 Example Output

**Input:**
- Query: "escalation issues"  
- Domain: "Banking"
- 3 conversations, 115 utterances

**Output:**
- 40 clusters (21 Agent + 19 Customer)
- All clusters labeled by LLM:
  - "Agent: Identify self and request assistance"
  - "Customer: Compare financial services"
  - "Agent: Request fee waiver"
- Directed graph showing conversation flow
- Metadata: escalation levels, churn risk, empathy scores

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- Ollama (for LLM cluster labeling)
- NVIDIA GPU (optional, for faster processing)

### Automated Setup (Recommended)

```bash
bash SETUP_ENVIRONMENT.sh
```

This script handles everything: virtual environment, dependencies, Ollama, and model downloads.

### Manual Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install faiss-cpu ollama networkx

# Install and start Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3:8b
```

See [SETUP.md](SETUP.md) for detailed instructions.

---

## 📖 Usage

### Basic Command

```bash
python3 integrated_pipeline.py \
  --query "your search query" \
  --domain "Banking" \
  --distance-threshold 0.4
```

### With LLM Labels (Recommended)

```bash
python3 integrated_pipeline.py \
  --query "escalation issues" \
  --domain "Banking" \
  --distance-threshold 0.4 \
  --formats json graphml html \
  -l \
  -lm llama3:8b
```

### All Options

```bash
python3 integrated_pipeline.py \
  --query "escalation issues"                    # Search query
  --domain "Banking"                             # Filter by domain
  --distance-threshold 0.4                       # Clustering threshold (0.3-0.6)
  --formats json graphml html                    # Output formats
  --model sergioburdisso/dialog2flow-joint-bert-base  # Embedding model
  -l                                             # Enable LLM labels
  -lm llama3:8b                                  # LLM model
  --output-dir ./output                          # Output directory
```

### Available Domains

Your data may include: `Banking`, `Flight`, `Hotel`, `Retail`, `Telecom`, `Insurance`, etc.

Check available domains:
```bash
python3 -c "
import json
with open('data/final_json_for_d2f.json') as f:
    domains = sorted(set(t.get('domain', '') for t in json.load(f)))
print('\n'.join(domains))
"
```

---

## 📁 Output Files

```
output/
├── trajectories_with_metadata.json     # Extracted trajectories + clusters
├── graph_with_metadata.json            # Graph in JSON format
├── graph_with_metadata.graphml         # Graph in GraphML format
├── graph_visualization.html            # Interactive visualization ⭐
└── graph_dialog2flow/
    ├── graph.graphml                   # Dialog2Flow format
    ├── graph.png                       # PNG visualization
    └── visualization/graph.html        # Alternative HTML view
```

**Bonus:** Top 20 conversations saved in `data/top_K/` with full metadata.

---

## 🎯 Core Features

### 1. Semantic Search
- Uses `sentence-transformers/all-mpnet-base-v2` for embeddings
- FAISS index for fast similarity search
- Finds top K most relevant conversations for any query

### 2. Metadata Preservation
- **Utterance-level:** escalation_level, churn_risk_score, empathy_score, intents_emotions, dialogue_acts, action_type, escalation_reason_tags
- **Cluster-level:** Aggregated statistics (mean, std, min, max)
- **Full tracking:** transcript_id + turn_idx for every utterance

### 3. Speaker-Separated Clustering
- Agent and Customer utterances clustered separately
- Prevents mixing of Agent/Customer in same cluster
- Follows Dialog2Flow methodology

### 4. LLM Cluster Labeling
- Uses Ollama (llama3:8b, mistral, gemma, etc.)
- Generates canonical labels for each cluster
- Example: "Agent: Identify self and request assistance"

### 5. Directed Graph Visualization
- NetworkX DiGraph with visual arrow markers
- Interactive D3.js visualization
- Hover tooltips show full metadata
- Node size reflects number of utterances

---

## 🧪 Verification

After running the pipeline, verify outputs:

```bash
# Check graph statistics
python3 -c "
import json
with open('output/graph_with_metadata.json') as f:
    data = json.load(f)
print(f'Nodes: {len(data[\"nodes\"])}')
print(f'Edges: {len(data[\"edges\"])}')
"

# Check LLM labels
python3 -c "
import json
with open('output/trajectories_with_metadata.json') as f:
    data = json.load(f)
labels = data.get('cluster_labels', {})
print(f'Labels: {len(labels)}')
for cid, label in list(labels.items())[:5]:
    print(f'  {cid}: {label}')
"

# Open visualization
xdg-open output/graph_visualization.html  # Linux
# open output/graph_visualization.html    # macOS
```

---

## 📚 Documentation

- **[SETUP.md](SETUP.md)** - Detailed setup guide with troubleshooting
- **[PAPER.md](PAPER.md)** - Research paper reference
- **[FILES_ANALYSIS.txt](FILES_ANALYSIS.txt)** - File structure analysis

---

## 🔧 Pipeline Architecture

```
Step 1: Find Top 20 Conversations
  ├─ Load data from data/final_json_for_d2f.json
  ├─ Filter by domain
  ├─ Create utterance embeddings
  ├─ Build FAISS index
  ├─ Search with query
  └─ Save top 20 to data/top_K/

Step 2: Prepare for Dialog2Flow
  ├─ Convert to simplified text format
  └─ Save metadata to data/example/

Step 3: Extract Trajectories
  ├─ Load Dialog2Flow model
  ├─ Cluster Agent utterances separately
  ├─ Cluster Customer utterances separately
  ├─ Generate LLM labels (if enabled)
  ├─ Aggregate metadata per cluster
  └─ Save to output/trajectories_with_metadata.json

Step 4: Build Graphs
  ├─ Build metadata-enhanced graph
  ├─ Build Dialog2Flow action flow graph
  ├─ Export as JSON, GraphML, HTML
  └─ Generate interactive visualizations
```

---

## 🎓 Research Citation

This work extends the Dialog2Flow methodology:

```bibtex
@article{burdisso2022dialog2flow,
  title={Dialog2Flow: Pre-training Soft-Contrastive Action-Driven Sentence Embeddings for Automatic Dialog Flow Extraction},
  author={Burdisso, Sergio and Madikeri, Srikanth and Motlicek, Petr},
  journal={arXiv preprint arXiv:2206.07148},
  year={2022}
}
```

Original repository: https://github.com/sergioburdisso/dialog2flow

---

## 🐛 Troubleshooting

### Ollama Not Running
```bash
# Start Ollama
ollama serve &

# Verify
ollama list
```

### Missing Models
```bash
# Download models
python3 -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
SentenceTransformer('sergioburdisso/dialog2flow-joint-bert-base')
"
```

### No Domain Found
```bash
# Check available domains
python3 -c "
import json
with open('data/final_json_for_d2f.json') as f:
    domains = set(t.get('domain') for t in json.load(f))
print(sorted(domains))
"
```

See [SETUP.md](SETUP.md) for more troubleshooting.

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Check [SETUP.md](SETUP.md) for troubleshooting
- Review existing issues for solutions

---

## ⭐ Features in Progress

- [ ] Support for more LLM providers (OpenAI, Anthropic, etc.)
- [ ] Multi-domain comparison graphs
- [ ] Time-based flow analysis
- [ ] Export to Gephi/Cytoscape formats
- [ ] Real-time conversation analysis API

---

**Happy analyzing! 🎉**
