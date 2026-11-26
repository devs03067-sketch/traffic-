# ------------------------------------------------------------
# Dockerfile – Render (Docker runtime) – free tier
# ------------------------------------------------------------
FROM python:3.11-slim-bullseye

# Prevent interactive debconf dialogs during apt install
ENV DEBIAN_FRONTEND=noninteractive

# System packages needed by ultralytics / OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libgl1-mesa-glx \
        libglib2.0-0 \
        ffmpeg \
        libsm6 \
        libxext6 && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------
# Python environment
# -----------------------------------------------------------------
WORKDIR /app

# Install Python dependencies first (cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app source (including the YOLO model file)
COPY . .

# Expose the port Render will map (default is $PORT, we expose 8787 just in case)
EXPOSE 8787

# Run the Flask server – it will read $PORT supplied by Render
CMD ["python", "app.py"]
