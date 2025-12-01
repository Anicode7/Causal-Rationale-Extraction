#!/bin/bash

###############################################################################
# Dialog2Flow Pipeline - Automated Environment Setup Script
# 
# This script automatically sets up the complete environment for reproducing
# the Dialog2Flow pipeline results.
#
# Usage:
#   bash SETUP_ENVIRONMENT.sh
#
# What it does:
#   1. Checks system dependencies
#   2. Installs Ollama (if needed)
#   3. Creates Python virtual environment
#   4. Installs all Python dependencies
#   5. Downloads required models
#   6. Verifies the setup
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_command() {
    if command -v $1 &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_warning "$1 is not installed"
        return 1
    fi
}

###############################################################################
# Main Setup Process
###############################################################################

print_header "Dialog2Flow Pipeline - Environment Setup"

echo "This script will set up your environment for running the Dialog2Flow pipeline."
echo "It will install:"
echo "  - Python virtual environment"
echo "  - All required Python packages"
echo "  - Ollama (if not already installed)"
echo "  - Required AI models"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 0
fi

###############################################################################
# Step 1: Check System Dependencies
###############################################################################

print_header "Step 1: Checking System Dependencies"

# Check Python
if check_command python3; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_info "Python version: $PYTHON_VERSION"
    
    # Check if version is >= 3.10
    MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
        print_success "Python version is compatible (>= 3.10)"
    else
        print_error "Python 3.10+ is required. Please upgrade Python."
        exit 1
    fi
else
    print_error "Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

# Check pip
if ! check_command pip3 && ! check_command pip; then
    print_error "pip is not installed. Please install pip."
    exit 1
fi

# Check git
check_command git || print_warning "git not found (optional, but recommended)"

# Check graphviz
check_command dot || print_warning "graphviz not found (optional, for PNG graph export)"

###############################################################################
# Step 2: Install/Check Ollama
###############################################################################

print_header "Step 2: Setting up Ollama"

if check_command ollama; then
    OLLAMA_VERSION=$(ollama --version 2>&1 | grep -oP '\d+\.\d+\.\d+' | head -1)
    print_info "Ollama version: $OLLAMA_VERSION"
    
    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        print_success "Ollama is running"
    else
        print_info "Starting Ollama service..."
        ollama serve &> /dev/null &
        sleep 3
        
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            print_success "Ollama service started"
        else
            print_error "Failed to start Ollama service"
            exit 1
        fi
    fi
else
    print_info "Ollama not found. Installing..."
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_info "Installing Ollama for Linux..."
        curl -fsSL https://ollama.com/install.sh | sh
        
        # Start Ollama
        ollama serve &> /dev/null &
        sleep 3
        
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_info "Installing Ollama for macOS..."
        if check_command brew; then
            brew install ollama
            ollama serve &> /dev/null &
            sleep 3
        else
            print_error "Homebrew not found. Please install Ollama manually from https://ollama.com"
            exit 1
        fi
    else
        print_error "Unsupported OS. Please install Ollama manually from https://ollama.com"
        exit 1
    fi
    
    if check_command ollama; then
        print_success "Ollama installed successfully"
    else
        print_error "Failed to install Ollama"
        exit 1
    fi
fi

# Pull required model
print_info "Checking for llama3:8b model..."
if ollama list | grep -q "llama3:8b"; then
    print_success "llama3:8b model already downloaded"
else
    print_info "Downloading llama3:8b model (this may take a few minutes)..."
    ollama pull llama3:8b
    print_success "llama3:8b model downloaded"
fi

###############################################################################
# Step 3: Create Python Virtual Environment
###############################################################################

print_header "Step 3: Setting up Python Virtual Environment"

if [ -d ".venv" ]; then
    print_info "Virtual environment already exists"
    read -p "Recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .venv
        print_info "Removed old virtual environment"
    fi
fi

if [ ! -d ".venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv .venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
source .venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet
print_success "pip upgraded"

###############################################################################
# Step 4: Install Python Dependencies
###############################################################################

print_header "Step 4: Installing Python Dependencies"

if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found!"
    exit 1
fi

print_info "Installing dependencies from requirements.txt..."
pip install -r requirements.txt --quiet

print_info "Installing additional required packages..."
pip install faiss-cpu --quiet
pip install ollama --quiet
pip install networkx --quiet

print_success "All Python dependencies installed"

###############################################################################
# Step 5: Download AI Models
###############################################################################

print_header "Step 5: Downloading AI Models"

print_info "This will download SentenceTransformer models (~500MB each)"
print_info "Progress will be shown for each model"

python3 << 'PYTHON'
from sentence_transformers import SentenceTransformer
import sys

models = [
    'sentence-transformers/all-mpnet-base-v2',
    'sergioburdisso/dialog2flow-joint-bert-base'
]

for model_name in models:
    print(f"\n📥 Downloading {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        print(f"✓ {model_name} downloaded successfully")
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")
        sys.exit(1)

print("\n✓ All models downloaded successfully")
PYTHON

print_success "AI models downloaded"

###############################################################################
# Step 6: Verify Installation
###############################################################################

print_header "Step 6: Verifying Installation"

print_info "Running verification tests..."

python3 << 'PYTHON'
import sys

def check_import(module_name, display_name=None):
    if display_name is None:
        display_name = module_name
    try:
        if module_name == 'torch':
            import torch
            print(f"✓ PyTorch {torch.__version__}")
            if torch.cuda.is_available():
                print(f"  └─ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                print(f"  └─ Running on CPU")
        elif module_name == 'sentence_transformers':
            import sentence_transformers
            print(f"✓ SentenceTransformers {sentence_transformers.__version__}")
        elif module_name == 'faiss':
            import faiss
            print(f"✓ FAISS OK")
        elif module_name == 'networkx':
            import networkx as nx
            print(f"✓ NetworkX {nx.__version__}")
        elif module_name == 'ollama':
            import ollama
            print(f"✓ Ollama Python client OK")
        else:
            __import__(module_name)
            print(f"✓ {display_name}")
        return True
    except ImportError as e:
        print(f"✗ {display_name} not available: {e}")
        return False

modules = [
    ('torch', 'PyTorch'),
    ('sentence_transformers', 'SentenceTransformers'),
    ('faiss', 'FAISS'),
    ('networkx', 'NetworkX'),
    ('ollama', 'Ollama'),
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
    ('sklearn', 'scikit-learn')
]

all_ok = True
for module, name in modules:
    if not check_import(module, name):
        all_ok = False

if not all_ok:
    print("\n✗ Some modules failed to import")
    sys.exit(1)
else:
    print("\n✓ All modules verified successfully")
PYTHON

if [ $? -eq 0 ]; then
    print_success "All verifications passed"
else
    print_error "Verification failed"
    exit 1
fi

###############################################################################
# Step 7: Check Data File
###############################################################################

print_header "Step 7: Checking Data Files"

if [ -f "data/final_json_for_d2f.json" ]; then
    FILE_SIZE=$(du -h "data/final_json_for_d2f.json" | cut -f1)
    print_success "Data file found (${FILE_SIZE})"
    
    # Count transcripts
    TRANSCRIPT_COUNT=$(python3 -c "import json; print(len(json.load(open('data/final_json_for_d2f.json'))))")
    print_info "Transcripts in dataset: $TRANSCRIPT_COUNT"
    
    # Show available domains
    print_info "Available domains:"
    python3 -c "import json; domains = sorted(set(t.get('domain', 'Unknown') for t in json.load(open('data/final_json_for_d2f.json')))); print('  ' + '\n  '.join(domains))"
else
    print_warning "Data file not found: data/final_json_for_d2f.json"
    print_info "You need to provide your conversation data in this location"
    print_info "See SETUP.md for the required JSON format"
fi

###############################################################################
# Setup Complete
###############################################################################

print_header "Setup Complete! 🎉"

echo ""
echo "Your environment is ready to run the Dialog2Flow pipeline!"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   ${GREEN}source .venv/bin/activate${NC}"
echo ""
echo "2. Run the pipeline:"
echo "   ${GREEN}python3 integrated_pipeline.py \\${NC}"
echo "   ${GREEN}  --query \"escalation issues\" \\${NC}"
echo "   ${GREEN}  --domain \"Banking\" \\${NC}"
echo "   ${GREEN}  --distance-threshold 0.4 \\${NC}"
echo "   ${GREEN}  --formats json graphml html \\${NC}"
echo "   ${GREEN}  -l -lm llama3:8b${NC}"
echo ""
echo "3. View results:"
echo "   ${GREEN}open output/graph_visualization.html${NC}"
echo ""
echo "For more details, see SETUP.md"
echo ""
print_success "Setup completed successfully!"
echo ""

# Deactivate virtual environment
deactivate 2>/dev/null || true
