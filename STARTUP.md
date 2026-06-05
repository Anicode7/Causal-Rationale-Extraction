# Dialog2Flow Pipeline - Startup Guide

This guide provides comprehensive instructions on how to set up and run the Causal Analysis from Conversational Data (Dialog2Flow) project. You can choose to run the project using **Docker (Recommended)** or via a **Local Local Setup (Without Docker)**.

---

## 🐳 Option 1: Running with Docker (Recommended)

Using Docker ensures that all system dependencies, the graph database (Memgraph), and the LLM service (Ollama) are automatically set up without interfering with your host machine.

### 🪟 Windows vs. WSL: Which should you use?
If you are on Windows, **using WSL (Windows Subsystem for Linux) is highly recommended**. 
- **Docker Desktop**: Install Docker Desktop on Windows and ensure the **"Use the WSL 2 based engine"** setting is enabled (this is the default).
- **Execution**: While Docker commands can be run from normal Windows PowerShell, it is heavily advised to open your project and run your terminal commands from inside a **WSL environment** (e.g., Ubuntu). This prevents standard cross-platform headaches like file path issues, CRLF vs LF line endings in bash scripts, and volume-mounting permission errors.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac) or Docker Engine (Linux).
- [Docker Compose](https://docs.docker.com/compose/install/) (Comes automatically with Docker Desktop).
- If on Windows: WSL2 installed and configured with Docker Desktop.

### Step 1: Build and Start the Services

Open your terminal (WSL or Linux/macOS preferred), navigate to the project directory, and run:

```bash
docker-compose up --build -d
```

This command automatically builds the Python environment and runs the orchestration in detached mode (`-d`). It spins up three containers:
1. `ollama-debug`: The Ollama service running on port `11434`. Its data is persisted in a local Docker volume (`ollama_data`), so downloaded models are saved securely between restarts.
2. `memgraph-debug`: The Memgraph graph database accessible via port `7687` (Bolt Protocol) and `7444`.
3. `dialog2flow-debug`: The main Python application container. 
   - **Tip:** Your local project directory is mapped into this container via a volume (`.:/app`). This means **any changes you make to the `.py` files on your host machine will reflect inside the container instantly**, without needing a rebuild!

### Step 1.5: View Logs (Optional)
To verify everything started up correctly and watch for errors, view the live trace of logs:
```bash
docker-compose logs -f
```
(Press `Ctrl+C` to exit the log view at any time).

### Step 2: Download the LLM Model (One-time setup)

Exec into the Ollama container and pull the required model (e.g., `llama3:8b` or `llama3.2`):

```bash
docker exec -it ollama-debug ollama pull llama3:8b
# OR
docker exec -it ollama-debug ollama pull llama3.2
```

### Step 3: Run the Pipeline

Execute the pipeline from inside the application container:

```bash
docker exec -it dialog2flow-debug bash
```

Once inside the bash shell of the container, you can run the main script (e.g. `main.py` or pipeline equivalent):

```bash
python main.py
```
*(Or use `python integrated_pipeline.py` based on the entry point you are currently targeting with arguments.)*

### Step 4: Shutting Down

To stop the containers and clean up the environment, run:

```bash
docker-compose down
```

---

## 💻 Option 2: Running Locally (Without Docker)

Setting up without Docker requires manually installing the Python environment, the database (if needed for persistence), and Ollama.

### Prerequisites
- **OS**: Linux (Ubuntu/WSL recommended) or macOS.
- **Python**: 3.10 or higher.
- **GPU**: NVIDIA GPU with CUDA 12.1+ (Optional, but recommended for embeddings/LLM speed).

### Quick Automated Setup (macOS / Linux / WSL)

The repository provides a robust setup script that will check dependencies, install Ollama, create a virtual environment, install Python packages, and pull the necessary LLM model.

```bash
# 1. Provide execution permissions (if needed) and run setup
chmod +x SETUP_ENVIRONMENT.sh
bash SETUP_ENVIRONMENT.sh

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Run the application
python main.py
```

### Step-by-Step Manual Setup

If you prefer to install things manually, follow these steps:

#### 1. Install System Dependencies (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv git curl wget graphviz graphviz-dev build-essential
```

#### 2. Install and Setup Ollama
Install Ollama to run local LLMs for cluster labeling:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service in the background
ollama serve &

# Pull the required model
ollama pull llama3:8b
```

#### 3. Setup Python Virtual Environment
```bash
python3 -m venv .venv

# Activate it (Linux/macOS)
source .venv/bin/activate
# Activate it (Windows)
# .venv\Scripts\activate
```

#### 4. Install Python Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Additional critical dependencies to ensure are installed
pip install faiss-cpu ollama networkx
```

#### 5. Run the Memgraph Database (Optional but required for some modules)
If your architecture relies on Memgraph outside of its basic local graph generator, you will still need to run it via its standalone docker container:
```bash
docker run -p 7687:7687 -p 7444:7444 memgraph/memgraph-mage
```

#### 6. Run the Pipeline
```bash
python main.py
```

*(Note: Depending on your current architecture, replace `main.py` with `python integrated_pipeline.py --query "your query" ...` if invoking directly via CLI args).*
