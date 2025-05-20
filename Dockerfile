FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .
COPY setup.py .
COPY README.md .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy source code
COPY src/ src/
COPY config/ config/
COPY run.py .

# Create directories
RUN mkdir -p data/raw data/processed models outputs

# Set environment variables
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 12000
EXPOSE 12001

# Set entrypoint
ENTRYPOINT ["python", "run.py"]

# Default command
CMD ["--help"]