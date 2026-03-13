FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install root requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout=120 -r requirements.txt

# Install the basic-pitch local package
COPY basic-pitch/ ./basic-pitch/
RUN pip install --no-cache-dir ./basic-pitch/

# Pre-download ByteDance model at build time
RUN python -c "\
import urllib.request, pathlib; \
model_dir = pathlib.Path('/root/piano_transcription_inference_data'); \
model_dir.mkdir(parents=True, exist_ok=True); \
model_path = model_dir / 'note_F1=0.9677_pedal_F1=0.9186.pth'; \
url = 'https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1'; \
print('Downloading ByteDance model...'); \
urllib.request.urlretrieve(url, model_path); \
print('Done.') \
" || echo "Model download failed — will retry at runtime"

# Copy rest of app
COPY . .

EXPOSE 8000

CMD ["python", "-u", "server.py"]