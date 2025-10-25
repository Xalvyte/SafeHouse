# ==========================
# SafeHouse Flask Dockerfile
# ==========================

# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libopenblas-dev \
    build-essential \
    python3-dev \
    gcc \
    libcap-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*



# Copy all project files to /app
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Flask port
EXPOSE 8000

# Environment variables (Flask)
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the Flask app
CMD ["python", "app.py"]
