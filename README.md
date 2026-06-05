# Causal Analysis from Conversational Data

Backend system for retrieval, graph construction, causal reasoning, and graph-guided inference over large conversational datasets. The project retrieves semantically relevant transcripts, converts dialogues into structured flow graphs, scores candidate causal transitions with LLMs, and synthesizes evidence-grounded answers for analytical queries.

## Overview

- **Retrieval layer:** MPNet embeddings, SQLite-backed embedding cache, FAISS similarity search, domain-intent routing, and transcript-level ranking.
- **Graph layer:** Dialog2Flow-based trajectory extraction, speaker-separated clustering, metadata-aware directed graph construction, and export to JSON/GraphML/HTML.
- **Causal layer:** Memgraph-backed subgraph retrieval, LLM-based edge scoring, probabilistic traversal, and threshold-pruned causal chain discovery.
- **Inference layer:** Graph-guided RAG over retrieved causal chains with support for contextual follow-up queries.

## Demo and Report

- **YouTube Demo:** [https://youtu.be/9tKXU63vhBU]
- **Project Report:** [OBSERVE (3).pdf](<docs/OBSERVE (3).pdf>)
- **Architecture Notes:** [ARCHITECTURE.md](ARCHITECTURE.md)

## What the System Does

1. Reformulates and routes a user query into one or more domain-intent retrieval queries.
2. Searches a 19,621-transcript / 686,764-turn corpus using `sentence-transformers/all-mpnet-base-v2` embeddings and `FAISS`.
3. Aggregates top-matching turns into transcript-level results and materializes the Top-K working set.
4. Converts retrieved conversations into Dialog2Flow-compatible text while preserving turn metadata.
5. Extracts speaker-aware dialogue trajectories and cluster assignments.
6. Builds a metadata-enriched directed conversation graph with transition weights.
7. Embeds graph nodes, ingests them into Memgraph, and retrieves causal neighborhoods around semantic landing points.
8. Scores causal edges with an LLM, performs probabilistic traversal, and generates evidence-grounded answers.

## Repository Structure

```text
.
+-- main.py                             # End-to-end batch entry point
+-- graph_gen.py                        # Retrieval + graph generation orchestration
+-- graph_operator.py                   # Memgraph ingestion, subgraph extraction, causal traversal, answer synthesis
+-- query_splitter.py                   # Domain-intent routing and Top-K retrieval materialization
+-- create_embeddings_db.py             # Offline embedding + FAISS index builder
+-- query_embeddings_db.py              # Cached semantic retrieval over embeddings DB
+-- prepare_for_dialog2flow.py          # Converts Top-K transcripts into Dialog2Flow inputs
+-- extract_trajectories_with_metadata.py
|   # Speaker-separated clustering + metadata aggregation
+-- build_graph_with_metadata.py        # Metadata-aware directed graph construction and export
+-- embedder.py                         # Node/query embedding for graph retrieval
+-- llm_handler.py                      # LLM prompts for routing, edge scoring, and answer generation
+-- docker-compose.yml                  # Ollama + Memgraph + app stack
+-- ARCHITECTURE.md                     # System architecture notes
+-- STARTUP.md                          # Docker-first startup guide
+-- SETUP.md                            # Local environment setup guide
`-- data/final_annotated_dataset.json   # Main annotated transcript corpus
```

## Core Pipeline

### 1. Query Routing and Retrieval

- Uses `llama3.2` through `Ollama` to map user questions into domain-intent retrieval queries.
- Supports **36 domain-intent classes** across Banking, Flight, Hotel, Insurance, Retail, and Telecom.
- Loads cached turn-level embeddings from `SQLite` and performs domain-filtered `FAISS IndexFlatIP` search.
- Aggregates top-matching turns into transcript-level scores using max and average similarity statistics.

### 2. Top-K Corpus Materialization

- Writes retrieved transcripts and retrieval metadata into `data/top_K/`.
- Preserves transcript ID, intent, reason-for-call, and top matching turns for downstream processing.
- Supports follow-up queries by persisting transcript history across runs.

### 3. Conversation-to-Graph Preparation

- Converts retrieved JSON transcripts into Dialog2Flow-compatible `Speaker: text` files.
- Persists aligned turn metadata separately so cluster assignments can be mapped back to the source turns.
- Maintains provenance fields including transcript ID, turn index, speaker, and conversational metadata.

### 4. Trajectory Extraction and Clustering

- Uses `sergioburdisso/dialog2flow-joint-bert-base` for dialogue turn representations.
- Applies speaker-separated `AgglomerativeClustering` with cosine distance and average linkage.
- Prevents Agent and Customer turns from collapsing into the same latent dialogue state.
- Aggregates cluster-level metadata including means, standard deviations, minima, maxima, and deduplicated categorical fields.

### 5. Metadata-Aware Graph Construction

- Builds a `NetworkX DiGraph` where nodes correspond to clustered dialogue states and edges represent transitions.
- Stores utterance lists, source-turn mappings, cluster labels, and aggregated metadata directly on nodes.
- Tracks edge weights by transition frequency and by number of unique dialogues contributing to each edge.
- Exports graph artifacts in JSON, GraphML, GEXF, and interactive HTML forms.

### 6. Graph Retrieval and Causal Reasoning

- Embeds graph nodes using `all-mpnet-base-v2` and ingests them into `Memgraph`.
- Creates a Memgraph vector index to retrieve semantic landing points for the user query.
- Extracts bounded subgraphs around landing nodes and reconstructs local causal neighborhoods.
- Uses batched LLM inference to assign edge-level causal probabilities and short explanations.
- Runs threshold-pruned BFS-style traversal to recover high-probability causal chains.

### 7. Inference and Answer Generation

- Converts top-ranked causal chains into a compact evidence context.
- Answers the user query with LLM synthesis constrained to retrieved graph evidence.
- Supports follow-up reasoning by contextualizing ambiguous queries using stored conversation history.

## Technology Stack

- **Language:** Python 3.10+
- **Embedding models:** `sentence-transformers/all-mpnet-base-v2`, `sergioburdisso/dialog2flow-joint-bert-base`
- **Vector search:** `FAISS`
- **Storage:** `SQLite`, JSON artifacts
- **Graph processing:** `NetworkX`
- **Graph database:** `Memgraph` via `neo4j` driver
- **LLM serving:** `Ollama`
- **ML / NLP libraries:** `sentence-transformers`, `scikit-learn`, `transformers`, `torch`, `numpy`
- **Runtime / tooling:** Docker, Docker Compose

## Key Data Structures and Methods

- **Embeddings store:** turn-level embedding rows in `SQLite` with serialized vectors and metadata
- **Vector index:** normalized `FAISS IndexFlatIP` for cosine-similarity retrieval
- **Trajectory representation:** per-dialogue ordered lists of cluster assignments with aligned turn metadata
- **Graph representation:** `NetworkX DiGraph` with metadata-rich node and edge attributes
- **Subgraph extraction:** landing-point-centered neighborhood retrieval from Memgraph
- **Causal scoring:** batched LLM evaluation of candidate edges
- **Traversal:** BFS-style probability propagation with threshold pruning and best-score updates

## Running the Project

### Option 1: Docker

Start the full stack:

```bash
docker-compose up --build -d
```

Pull the LLM model inside the Ollama container:

```bash
docker exec -it ollama-debug ollama pull llama3.2
```

Run the application container:

```bash
docker exec -it dialog2flow-debug bash
python main.py
```

See [STARTUP.md](STARTUP.md) for the full Docker workflow.

### Option 2: Local Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Start Ollama and pull a model:

```bash
ollama serve
ollama pull llama3.2
```

Create the embedding database once:

```bash
python create_embeddings_db.py --data-path data/final_annotated_dataset.json --db-path data/embeddings.db
```

Run the pipeline:

```bash
python main.py
```

See [SETUP.md](SETUP.md) for detailed local setup instructions.

## Main Entry Points

### Build the embeddings database

```bash
python create_embeddings_db.py --data-path data/final_annotated_dataset.json --db-path data/embeddings.db
```

### Run the end-to-end backend pipeline

```bash
python main.py
```

## Inputs and Outputs

### Input

- `data/final_annotated_dataset.json`
- Annotated conversations with transcript-level and turn-level metadata

### Common output artifacts

- `ans.json` - query answers and serialized result bundles
- `causal_chains.json` - ranked causal chains
- `landing_points.json` - semantic landing nodes retrieved from the graph
- `subgraph.json` - extracted causal neighborhoods
- `edges_computed.json` - LLM-scored edges and explanations
- `output/graph_with_metadata.json` - metadata-aware directed graph
- `output/graph_with_metadata_embedded.json` - graph nodes enriched with embeddings
- `output/trajectories_with_metadata.json` - cluster assignments and aggregated metadata

## Example Execution Flow

```text
User query
  -> LLM routing into domain-intent searches
  -> MPNet + FAISS transcript retrieval
  -> Top-K transcript materialization
  -> Dialog2Flow-compatible conversation export
  -> Speaker-separated clustering
  -> Directed metadata-aware graph construction
  -> Memgraph vector landing-point retrieval
  -> LLM causal edge scoring
  -> BFS-style probabilistic chain discovery
  -> Evidence-grounded answer generation
```

## Verification Checklist

- `data/final_annotated_dataset.json` is present
- `Ollama` is running
- a local model such as `llama3.2` is pulled
- `data/embeddings.db` and the paired `.faiss` index exist
- `main.py` completes and writes `ans.json`
- graph artifacts are created under `output/`

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - system architecture and workflow
- [STARTUP.md](STARTUP.md) - Docker and startup instructions
- [SETUP.md](SETUP.md) - local setup and environment preparation
- [COMMANDS.md](COMMANDS.md) - reproducibility and command reference

## Acknowledgments

This repository builds on the ideas from [Dialog2Flow](https://github.com/sergioburdisso/dialog2flow) and extends them with retrieval infrastructure, metadata-aware graph construction, Memgraph-backed causal reasoning, and graph-guided answer synthesis.
