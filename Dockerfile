FROM docker.io/library/python:3.11-slim

WORKDIR /app

# Install minimal build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .

# Install CPU-only PyTorch first to prevent downloading CUDA wheels, then remaining requirements
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch -r requirements.txt

# Copy application source code and documents
COPY app/ ./app/
COPY data/ ./data/
COPY eval/ ./eval/
COPY static/ ./static/

EXPOSE 8000

ENV HOST=0.0.0.0
ENV PORT=8000
ENV QDRANT_URL=http://qdrant:6333

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
