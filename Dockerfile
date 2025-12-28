# Start with a very small base (approx 125MB)
FROM python:3.9-slim

# Set environment variables to keep the container clean
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install ONLY the essential libraries for OpenCV to work
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip to handle the small CPU-only wheels correctly
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements first to keep the build fast
COPY requirements.txt .

# Install only what is in your requirements
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy your model and API script
COPY best_brain_tumor_model.pth .
COPY api.py .

EXPOSE 8000

# Start the API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]