# Dockerfile — AUTO Eraser (RunPod Serverless)
# Detecție automată (EasyOCR + Florence-2) + Inpainting (ProPainter + DiffuEraser)
#
# Build & push:
#   docker build -t <dockerhub_user>/auto-eraser:latest .
#   docker push <dockerhub_user>/auto-eraser:latest
#
# RunPod: New Endpoint → Container Image → <dockerhub_user>/auto-eraser:latest
#   GPU recomandat: 24GB+ (L4 / RTX 4090 / A5000). Pentru 1080p full: L40S 48GB.
#   Container Disk: minim 60 GB (greutățile au ~15 GB + imagine + temp)
#
# ⚠️ Imaginea finală e MARE (~30-40GB) pentru că greutățile sunt copiate în ea
#    → cold start-ul e mai lent la primul run pe un host nou, apoi e cache-uit.
#    Alternativă: Network Volume — setează WEIGHTS_DIR=/runpod-volume/weights
#    și rulează download_weights.py o singură dată pe volum (vezi README).

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Dependințe sistem: FFmpeg + OpenCV runtime + git
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 1. DiffuEraser (cod + dependințele lui)
RUN git clone --depth 1 https://github.com/lixiaowen-xw/DiffuEraser.git /app/DiffuEraser && \
    pip install --no-cache-dir -r /app/DiffuEraser/requirements.txt

# 2. Dependințele handler-ului (detecție + runpod)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Greutățile — DOUĂ variante:
#    A) RECOMANDAT — Network Volume: imagine slim, push rapid de acasă.
#       NU face nimic aici. Pe endpoint setezi WEIGHTS_DIR=/runpod-volume/weights
#       și atașezi un Network Volume; handler.py descarcă automat greutățile
#       pe volum la primul start (o singură dată, ~15GB pe netul RunPod).
#    B) Baked în imagine: decomentează RUN-ul de mai jos → cold start mai rapid,
#       dar imaginea crește la ~35GB și push-ul de acasă durează ore.
ENV WEIGHTS_DIR=/app/weights
ENV DIFFUERASER_DIR=/app/DiffuEraser
COPY download_weights.py .
# RUN python download_weights.py   # ← varianta B (bake în imagine)

# 4. Handler
COPY handler.py .

CMD ["python", "-u", "handler.py"]
