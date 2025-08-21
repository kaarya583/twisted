# TwistEd ML Application Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt requirements_ml.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_ml.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p ml_outputs/models ml_outputs/evaluations ml_outputs/explanations ml_outputs/features

# Download NOAA data (optional - can be done at runtime)
RUN python -c "from rag.downloader import download_noaa_data; download_noaa_data(years=[2023, 2024], output_dir='noaa_data')" || echo "Data download optional"

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "twisted_ml_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
