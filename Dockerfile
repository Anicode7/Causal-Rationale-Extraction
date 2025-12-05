FROM python:3.10-slim

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Install uv
RUN pip install --no-cache-dir uv

# 2. Copy requirements
COPY requirements.txt .

# 3. Install dependencies using uv
# --system: installs to global python (simpler for debug containers)
RUN uv pip install --system --no-cache -r requirements.txt

# Keep container running
CMD ["tail", "-f", "/dev/null"]