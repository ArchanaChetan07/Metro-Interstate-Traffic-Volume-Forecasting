# Fast-path image: ADF + LinearRegression + GradientBoosting (no TensorFlow).
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-fast.txt .
RUN pip install --no-cache-dir -r requirements-fast.txt

COPY metro_traffic/ metro_traffic/
COPY scripts/ scripts/
COPY data/ data/
COPY artifacts/benchmark_report.json artifacts/benchmark_report.json
COPY tests/ tests/

ENV PYTHONPATH=/app

# Default: train/eval fast models and print metrics JSON
CMD ["python", "scripts/run_benchmark.py", "--out", "/tmp/benchmark_report.json"]
