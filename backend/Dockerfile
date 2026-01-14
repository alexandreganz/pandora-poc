FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for generated images
RUN mkdir -p generated_images

# Expose port (Railway sets PORT dynamically)
EXPOSE 8000

# Run the application (use PORT env var if set, default to 8000)
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
