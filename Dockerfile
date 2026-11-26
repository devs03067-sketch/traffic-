# Dockerfile – Render (Docker runtime)
FROM python:3.11-slim-bookworm

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# System packages needed by Ultralytics and OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        libsm6 \
        libxext6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app source, including the YOLO model file
COPY . .

# Render sets the PORT environment variable
EXPOSE 8787

# Start the Flask server
CMD ["python", "app.py"]
