FROM python:3.11-slim

# System deps for librosa / soundfile / matplotlib
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ByteDance model (~165MB) at build time
# so first user request isn't slow
RUN python -c "\
import urllib.request, pathlib, os; \
model_dir = pathlib.Path('/root/piano_transcription_inference_data'); \
model_dir.mkdir(parents=True, exist_ok=True); \
model_path = model_dir / 'note_F1=0.9677_pedal_F1=0.9186.pth'; \
url = 'https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1'; \
print('Downloading ByteDance model...'); \
urllib.request.urlretrieve(url, model_path); \
print('Done.') \
" || echo "Model download failed — will retry at runtime"

# Copy app code
COPY . .

EXPOSE 8000

CMD ["python", "server.py"]