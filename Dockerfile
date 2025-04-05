FROM python:3.11-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Install system dependencies and tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \  # Required for XGBoost
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir uv \
    && uv pip install -e .

# Copy application code
COPY . .

# Create models directory if it doesn't exist
RUN mkdir -p models

# Create necessary streamlit configuration directory
RUN mkdir -p .streamlit

# Create streamlit config file
RUN echo '[server]\nheadless = true\naddress = "0.0.0.0"\nport = 5000\nenableCORS = false' > .streamlit/config.toml

# Add healthcheck for the application
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/ || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port for Streamlit and Flask API
EXPOSE 5000
EXPOSE 8000

# Start both API server and Streamlit application
CMD ["sh", "-c", "python -m api & streamlit run app.py --server.port 5000"]