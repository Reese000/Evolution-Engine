# ══════════════════════════════════════════════════════════════════════
# Evolution Engine — Production Dockerfile
# Python 3.11 + CUDA support (optional GPU runtime)
# ══════════════════════════════════════════════════════════════════════

# ── Base: use nvidia/cuda image for GPU support, fallback to python:3.11 ──
# To build with GPU: docker build --build-arg CUDA=true -t evolution-engine .
# To build CPU only: docker build -t evolution-engine .

ARG CUDA=false

FROM python:3.11-slim AS base-cpu
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base-gpu

# Select final base based on CUDA arg
FROM base-${CUDA} AS runtime

# Set up Python in GPU base (which is Ubuntu-based)
ARG CUDA=false
RUN if [ "$CUDA" = "true" ]; then \
      apt-get update && \
      apt-get install -y python3 python3-pip python3-venv && \
      ln -sf /usr/bin/python3 /usr/bin/python && \
      rm -rf /var/lib/apt/lists/*; \
    fi

# ── System deps ─────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      gcc g++ libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# ── Non-root user ───────────────────────────────────────────────────
RUN groupadd -r evo && useradd -r -g evo -d /app -s /sbin/nologin evo

# ── Working directory ───────────────────────────────────────────────
WORKDIR /app

# ── Copy requirements first (layer cache) ──────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project ───────────────────────────────────────────────────
COPY . .

# Ensure all scripts are executable
RUN chmod +x deploy.sh stop.sh status.sh 2>/dev/null || true

# Create log/data directories
RUN mkdir -p logs metrics cycle_data archive && \
    chown -R evo:evo /app

# ── Health check ───────────────────────────────────────────────────
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "\
import json, sys; \
f=open('health.json') if __import__('os').path.isfile('health.json') else None; \
d=json.load(f) if f else {}; \
f and f.close(); \
s=d.get('status','unknown'); \
sys.exit(0 if s in ('running','idle') else 1)" || \
  python -c "import sys; sys.exit(0 if __import__('os').path.isfile('health.json') else 1)"

# ── User ───────────────────────────────────────────────────────────
USER evo

# ── Entrypoint ─────────────────────────────────────────────────────
ENTRYPOINT ["python", "autonomous_runner.py"]
CMD ["--resume"]
