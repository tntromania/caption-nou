# Dockerfile — AUTO Eraser (RunPod Serverless)
# Detecție automată (EasyOCR + Florence-2) + Inpainting (ProPainter + DiffuEraser)
#
# Imaginea conține TOT: cod + greutăți (~15GB, descărcate la BUILD) → zero
# download la cold start, pe orice mașină. Nu mai e nevoie de Network Volume;
# dacă endpointul mai are WEIGHTS_DIR setat pe volum, handler-ul îl ignoră
# când găsește greutățile complete în imagine.
#
# PyTorch 2.7 + CUDA 12.8 → kernele pentru toate GPU-urile, inclusiv
# Blackwell/RTX 50xx (sm_120) — fix pentru "no kernel image is available".
#
# GPU recomandat: 24GB+ (RTX 4090 / 5090 / L4 / A5000; pt 1080p full: L40S).
# Container Disk: minim 60 GB (imagine ~35GB + temp la procesare).

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Sistem: Python 3.10 + FFmpeg + OpenCV runtime + git
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip git ffmpeg \
        libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/local/bin/python && \
    python -m pip install --no-cache-dir --upgrade pip

# PyTorch 2.7 cu CUDA 12.8 (suport nativ Blackwell sm_120)
RUN pip install --no-cache-dir \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 1. DiffuEraser (cod + dependințele lui) — FĂRĂ torch-ul pinuit de el
#    (torch==2.3.1/torchvision/torchaudio ne-ar downgrade-ui build-ul cu128)
RUN git clone --depth 1 https://github.com/lixiaowen-xw/DiffuEraser.git /app/DiffuEraser && \
    sed -i '/^torch/d' /app/DiffuEraser/requirements.txt && \
    pip install --no-cache-dir -r /app/DiffuEraser/requirements.txt

# 2. Dependințele handler-ului (detecție + runpod)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Greutățile — BAKED în imagine la build: SD1.5 + VAE + DiffuEraser + PCM
#    + ProPainter + Florence-2 + EasyOCR (~15GB). Build-ul durează mai mult,
#    dar workerii pornesc fără niciun download, exact ca la captionremover.
ENV WEIGHTS_DIR=/app/weights
ENV DIFFUERASER_DIR=/app/DiffuEraser
COPY download_weights.py .
RUN python download_weights.py

# 4. Handler
COPY handler.py .

CMD ["python", "-u", "handler.py"]
