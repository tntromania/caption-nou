# Dockerfile — AUTO Eraser (RunPod Serverless)
# Detecție automată (EasyOCR + Florence-2) + Inpainting (ProPainter + DiffuEraser)
#
# Greutățile NU mai sunt în imagine (v2 le avea baked → layer de ~27GB pe care
# fiecare mașină nouă îl trăgea la fiecare release/scalare, 5-8 min de
# "initializing"). Acum stau pe Network Volume (WEIGHTS_DIR=/runpod-volume/weights):
# handler-ul le descarcă O SINGURĂ DATĂ pe volum la primul start, apoi toți
# workerii din datacenter le montează instant. Imaginea rămâne mică → pull
# rapid pe mașini noi, release-uri de handler aproape instant.
# ⚠️ Endpointul TREBUIE să aibă Network Volume atașat înainte de deploy-ul
# acestei imagini, altfel greutățile se re-descarcă la fiecare cold start.
#
# PyTorch 2.7 + CUDA 12.8 → kernele pentru toate GPU-urile, inclusiv
# Blackwell/RTX 50xx (sm_120) — fix pentru "no kernel image is available".
#
# GPU recomandat: 24GB+ (RTX 4090 / 5090 / L4 / A5000; pt 1080p full: L40S).
# Container Disk: 30 GB ajunge (imagine ~12GB + temp la procesare).

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

# 3. Greutățile — pe Network Volume: handler.py rulează download_weights.py
#    la primul start dacă nu găsește markerul .easyocr_zh.done în WEIGHTS_DIR
#    (~15GB, o singură dată per volum, apoi persistă).
ENV WEIGHTS_DIR=/runpod-volume/weights
ENV DIFFUERASER_DIR=/app/DiffuEraser
COPY download_weights.py .

# 4. Handler
COPY handler.py .

CMD ["python", "-u", "handler.py"]
