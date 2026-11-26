# Dockerfile for deploying the YOLO Flask app on Render (Docker runtime)
FROM python:3.11-slim

# System dependencies required by ultralytics (opencv, libgl1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (app.py, model, optional index.html, etc.)
COPY . .

EXPOSE 8787

CMD ["python", "app.py"]
