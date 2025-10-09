# Dockerfile
# Railway deployment with full ML library support

FROM python:3.12-slim

# Install system dependencies required for ML libraries
# This includes OpenMP (libgomp) needed by LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libopenblas-dev \
    liblapack-dev \
    build-essential \
    gcc \
    g++ \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p model data streamlit src

# Expose port (Railway sets $PORT at runtime)
EXPOSE $PORT

# Start Streamlit app (matches your railway.json)
CMD ["streamlit", "run", "streamlit/app.py", "--server.address=0.0.0.0", "--server.port=$PORT"]

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/src
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
