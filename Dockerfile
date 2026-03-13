FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    # Install CPU-only torch first (~200MB vs ~800MB for default)
    pip install --no-cache-dir torch==2.2.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --timeout=120 -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-u", "server.py"]