#!/usr/bin/env python3
"""
download_weights.py — descarcă toate greutățile pentru AUTO Eraser.

Rulează la build (Dockerfile) SAU la primul start dacă folosești Network Volume
(setează WEIGHTS_DIR pe volum, ex: /runpod-volume/weights).

Total: ~12-15 GB (SD1.5 diffusers + VAE + DiffuEraser + PCM + ProPainter + Florence-2)
"""
import os
import urllib.request

WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/app/weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Cache-urile HF + EasyOCR lângă greutăți (persistă pe Network Volume)
os.environ.setdefault("HF_HOME", os.path.join(WEIGHTS_DIR, "hf-cache"))
os.environ.setdefault("EASYOCR_MODULE_PATH", os.path.join(WEIGHTS_DIR, "easyocr"))

from huggingface_hub import snapshot_download  # după setarea HF_HOME

def done_marker(name):
    return os.path.join(WEIGHTS_DIR, f".{name}.done")

def mark_done(name):
    open(done_marker(name), "w").close()

def hf(repo_id, subdir, name, ignore=None):
    if os.path.exists(done_marker(name)):
        print(f"[SKIP] {name}")
        return
    print(f"[DL] {repo_id} → {subdir}")
    snapshot_download(
        repo_id,
        local_dir=os.path.join(WEIGHTS_DIR, subdir),
        ignore_patterns=ignore or [],
    )
    mark_done(name)

# 1. Stable Diffusion 1.5 (mirror oficial — runwayml a fost retras de pe HF)
#    Ignorăm checkpoint-urile monolitice; diffusers folosește doar subfolderele.
hf("sd-legacy/stable-diffusion-v1-5", "stable-diffusion-v1-5", "sd15",
   ignore=["*.ckpt", "v1-5-pruned*", "*.non_ema*"])

# 2. VAE îmbunătățit
hf("stabilityai/sd-vae-ft-mse", "sd-vae-ft-mse", "vae")

# 3. DiffuEraser (greutățile modelului de video inpainting)
hf("lixiaowen/diffuEraser", "diffuEraser", "diffueraser")

# 4. PCM LoRA (few-step sampling — ckpt "2-Step")
hf("wangfuyun/PCM_Weights", "PCM_Weights", "pcm", ignore=["sdxl/*"])

# 5. ProPainter (priori) — de pe GitHub releases
PROPAINTER_DIR = os.path.join(WEIGHTS_DIR, "propainter")
os.makedirs(PROPAINTER_DIR, exist_ok=True)
PP_BASE = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0"
for fname in ["ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth"]:
    dest = os.path.join(PROPAINTER_DIR, fname)
    if os.path.exists(dest):
        print(f"[SKIP] propainter/{fname}")
        continue
    print(f"[DL] {fname}")
    urllib.request.urlretrieve(f"{PP_BASE}/{fname}", dest)

# 6. Florence-2 (detecție logo/watermark) — se descarcă în cache-ul HF standard
if not os.path.exists(done_marker("florence")):
    print("[DL] microsoft/Florence-2-large")
    snapshot_download(os.environ.get("FLORENCE_MODEL", "microsoft/Florence-2-large"))
    mark_done("florence")

# 7. EasyOCR (en + ro) — descarcă modelele de detecție/recunoaștere
if not os.path.exists(done_marker("easyocr")):
    print("[DL] EasyOCR en+ro")
    import easyocr
    easyocr.Reader(["en", "ro"], gpu=False, verbose=False)
    mark_done("easyocr")

print("✅ Toate greutățile sunt descărcate.")
