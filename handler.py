#!/usr/bin/env python3
"""
handler.py — AUTO Eraser (RunPod Serverless)

Pipeline COMPLET AUTOMAT — userul nu selectează nimic:
  1. DETECȚIE  — EasyOCR pe keyframes (text/captions, inclusiv text care se schimbă)
               + Florence-2 open-vocabulary grounding (logo, watermark — ce OCR nu vede)
  2. MĂȘTI TEMPORALE — box-urile statice (logo/watermark) acoperă tot videoul;
               box-urile dinamice (captions) doar intervalul în care apar
  3. INPAINTING — ProPainter (priori) + DiffuEraser (rafinare diffusion)
               → calitate mult peste LaMa per-frame, consistent temporal
  4. AUDIO     — remux audio original + scale înapoi la rezoluția originală

Input JSON:
  {
    "video_url":    "https://...",     # preferat
    "video_base64": "...",             # alternativ (<50MB)
    "targets":      ["captions","logos","watermarks"],   # default: toate
    "extra_prompts": ["nume canal"],   # opțional: alte lucruri de șters (Florence-2)
    "max_img_size": 960,               # rezoluția max de procesare (512-1920)
    "callback_url": "https://.../api/receive-ai-result", # upload direct la server
    "job_id":       "123"
  }

Output JSON:
  { "result_uploaded": true, "size_mb": 12.3, "detections": {...} }
  sau { "video_base64": "...", "detections": {...} }
  sau { "nothing_detected": true }   # nu s-a găsit nimic de șters
"""

import os
import sys
import base64
import shutil
import tempfile
import subprocess
import time
import traceback

import gc

_T_BOOT = time.time()

# alocatorul CUDA cu segmente expandabile reduce fragmentarea — OOM-urile
# ProPainter arătau 1-2GB "reserved but unallocated"; trebuie setat ÎNAINTE
# de importul torch ca să fie citit de alocator
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import runpod
import requests
import numpy as np
import cv2
import torch
from PIL import Image

# ── Config din env ───────────────────────────────────────────────────────────
WEIGHTS_DIR      = os.environ.get("WEIGHTS_DIR", "/app/weights")
DIFFUERASER_DIR  = os.environ.get("DIFFUERASER_DIR", "/app/DiffuEraser")

# Greutățile BAKED în imagine au prioritate: dacă build-ul le conține deja
# complete (markerul .easyocr_zh.done e ultimul scris de download_weights.py),
# ignorăm WEIGHTS_DIR extern (ex. Network Volume rămas setat pe endpoint)
# → zero download la cold start, indiferent de mașină/volum.
_BAKED_WEIGHTS = "/app/weights"
if WEIGHTS_DIR != _BAKED_WEIGHTS and os.path.exists(os.path.join(_BAKED_WEIGHTS, ".easyocr_zh.done")):
    print(f"[INIT] Greutăți baked în imagine → folosesc {_BAKED_WEIGHTS} (ignor WEIGHTS_DIR={WEIGHTS_DIR})", flush=True)
    WEIGHTS_DIR = _BAKED_WEIGHTS
MAX_SECONDS      = float(os.environ.get("MAX_SECONDS", "90"))
MAX_FPS          = int(os.environ.get("MAX_FPS", "30"))              # 60fps → 30fps: jumătate din cadre = jumătate din VRAM/timp

# ── Rezoluția de inpainting ──────────────────────────────────────────────────
# Rezoluția NU mai depinde de lungimea clipului. Înainte, `_proc_size` o scădea
# cu sqrt(600/cadre) ca să țină VRAM-ul în frâu → un clip de 78s la 720x1280
# ajungea să fie reconstruit la 182x324 și ridicat înapoi cu lanczos de 3.95x
# (= blur) iar `mask_dilation=8`, aplicat la acea rezoluție, ștergea efectiv
# 32px la rezoluția reală (= "scoate mai mult decât trebuie").
# Acum clipurile lungi se taie în BUCĂȚI temporale procesate la rezoluție plină
# (vezi run_inpainting), iar bugetul de VRAM se calculează din GPU-ul real.
PROC_MAX_SIDE    = int(os.environ.get("PROC_MAX_SIDE", "832"))       # latura lungă la care rulează inpainting-ul
MIN_PROC_SIDE    = int(os.environ.get("MIN_PROC_SIDE", "320"))       # nu coborâm sub atât nici la OOM (240p scalat la 172p e absurd)
# Calibrare VRAM: 600 cadre × 640×360px (~138M px·cadre) era zona dovedită sigură
# pe 24GB → ~5.75M px·cadre/GB. Luăm 5.0M ca marjă. Pe un GPU de 80GB bugetul
# devine ~400M px·cadre în loc de cei 138M hardcodați pentru 24GB.
PXFRAMES_PER_GB  = float(os.environ.get("PXFRAMES_PER_GB", "5.0e6"))
CHUNK_OVERLAP    = int(os.environ.get("CHUNK_OVERLAP", "12"))        # cadre de suprapunere între bucăți (continuitate temporală)
# Cap explicit pe cadre/bucată. Gol = derivat din VRAM (recomandat).
PROC_FRAME_BUDGET = int(os.environ.get("PROC_FRAME_BUDGET", "0"))
# Cât să dilate masca inpainting-ul, exprimat în pixeli la rezoluția ORIGINALĂ.
# ProPainter primește valoarea convertită în pixeli de procesare, deci efectul
# rămâne constant indiferent de scalare.
MASK_DILATE_PX   = float(os.environ.get("MASK_DILATE_PX", "6"))

DETECT_INTERVAL  = float(os.environ.get("DETECT_INTERVAL", "0.5"))   # secunde între keyframes OCR
FLORENCE_INTERVAL = float(os.environ.get("FLORENCE_INTERVAL", "2.0")) # secunde între keyframes Florence
FLORENCE_BEAMS   = int(os.environ.get("FLORENCE_BEAMS", "1"))        # 1 = greedy (3 = beam search, ~3x mai lent)
OCR_CONF         = float(os.environ.get("OCR_CONF", "0.25"))
STATIC_RATIO     = float(os.environ.get("STATIC_RATIO", "0.60"))     # % din keyframes ca un box să fie "static"
MAX_BOX_AREA_PCT = float(os.environ.get("MAX_BOX_AREA_PCT", "0.25")) # ignoră box-uri > 25% din frame
BOX_PAD          = int(os.environ.get("BOX_PAD", "6"))
DRIFT_MAX_PCT    = float(os.environ.get("DRIFT_MAX_PCT", "0.04"))    # drift max al unui cluster (fracție din diagonală) ca să fie overlay, nu text pe obiect
MASK_MAX_COVERAGE = float(os.environ.get("MASK_MAX_COVERAGE", "0.25")) # plafonul măștii pe un frame — peste, scoatem box-urile cele mai mari
                                                                       # (0.40 lăsa inpainting-ul fără sursă: 40% din cadru șters = terci)

sys.path.insert(0, DIFFUERASER_DIR)

# Cache-urile HuggingFace + EasyOCR merg lângă greutăți — esențial pe Network
# Volume, altfel Florence-2/EasyOCR s-ar re-descărca la fiecare cold start.
os.environ.setdefault("HF_HOME", os.path.join(WEIGHTS_DIR, "hf-cache"))
os.environ.setdefault("EASYOCR_MODULE_PATH", os.path.join(WEIGHTS_DIR, "easyocr"))

# Mod Network Volume: dacă greutățile lipsesc, le descărcăm O SINGURĂ DATĂ aici.
# (.easyocr_zh.done e ultimul marker scris de download_weights.py = descărcare completă;
#  markerul vechi .easyocr.done = imagine fără modelul chinezesc → re-rulăm downloadul)
if not os.path.exists(os.path.join(WEIGHTS_DIR, ".easyocr_zh.done")):
    print(f"[INIT] Greutăți lipsă în {WEIGHTS_DIR} — le descarc acum (~15GB, o singură dată)...", flush=True)
    subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_weights.py")],
        check=True,
    )

# DiffuEraser are hardcodată calea RELATIVĂ "weights/PCM_Weights" pentru LoRA-ul
# PCM (diffueraser.py, load_lora_weights). Cu greutățile pe Network Volume,
# ./weights nu există în CWD → symlink către WEIGHTS_DIR ca să se rezolve.
_local_weights = os.path.join(os.getcwd(), "weights")
if os.path.realpath(_local_weights) != os.path.realpath(WEIGHTS_DIR):
    if os.path.islink(_local_weights):
        os.unlink(_local_weights)
    if not os.path.exists(_local_weights):
        os.symlink(WEIGHTS_DIR, _local_weights)
        print(f"[INIT] Symlink {_local_weights} → {WEIGHTS_DIR}", flush=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

# VRAM-ul real al plăcii — bugetul de inpainting se derivă din el, nu dintr-o
# constantă calibrată pentru 24GB. Fără printul ăsta nu se putea vedea din
# loguri pe ce GPU rulează endpointul (workerul LaMa îl scrie, ăsta nu-l scria).
if DEVICE == "cuda":
    _PROPS = torch.cuda.get_device_properties(0)
    VRAM_GB = _PROPS.total_memory / (1024 ** 3)
    print(f"[INIT] GPU: {_PROPS.name} — {VRAM_GB:.0f}GB VRAM, sm_{_PROPS.major}{_PROPS.minor}", flush=True)
else:
    VRAM_GB = 8.0
    print("[INIT] GPU indisponibil — rulez pe CPU (foarte lent)", flush=True)


def _probe_nvenc():
    """h264_nvenc e de câteva ori mai rapid ca libx264 la re-encode full-res și,
    pe serverless, timpul de encode e timp GPU facturat. Îl PROBĂM efectiv (nu
    doar `-encoders`) fiindcă driverul poate lipsi chiar dacă ffmpeg îl listează."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-nostats", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=c=black:s=128x128:d=0.1",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, timeout=60,
        )
        return r.returncode == 0
    except Exception:
        return False


HAS_NVENC = DEVICE == "cuda" and _probe_nvenc()
print(f"[INIT] Encoder: {'h264_nvenc (GPU)' if HAS_NVENC else 'libx264 (CPU)'}", flush=True)


def _venc(crf="18", preset="medium"):
    """Argumentele de encoder video pentru ffmpeg — nvenc dacă există, altfel x264."""
    if HAS_NVENC:
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", crf, "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", preset, "-crf", crf]

# ═════════════════════════════════════════════════════════════════════════════
# INIT — toate modelele se încarcă O SINGURĂ DATĂ la pornirea workerului
# ═════════════════════════════════════════════════════════════════════════════
# ── Încărcare LENEȘĂ ─────────────────────────────────────────────────────────
# Cold start-ul era ~84s, din care ~62s doar blocul DiffuEraser + ProPainter.
# DiffuEraser e un pipeline Stable Diffusion 1.5 complet (UNet + VAE + motion
# module) și se folosea DOAR la quality="max" — pe care serverul nu-l trimite
# niciodată; toate joburile din loguri erau quality=fast, cu
# "[INPAINT] quality=fast → sar peste DiffuEraser". Plăteam deci ~1 minut de GPU
# la FIECARE cold start pentru un model neatins.
# Acum fiecare model se încarcă la prima folosire reală. ProPainter rămâne
# eager: el se folosește la orice job.
_t0 = time.time()
print("[INIT] Încărcare ProPainter...", flush=True)
from propainter.inference import Propainter
PROPAINTER = Propainter(os.path.join(WEIGHTS_DIR, "propainter"), device=DEVICE)
print(f"[INIT] ProPainter OK ({time.time() - _t0:.1f}s)", flush=True)

_DIFFU = None
def get_diffu():
    """DiffuEraser — doar la quality="max"."""
    global _DIFFU
    if _DIFFU is None:
        t = time.time()
        print("[LAZY] Încărcare DiffuEraser...", flush=True)
        from diffueraser.diffueraser import DiffuEraser
        _DIFFU = DiffuEraser(
            DEVICE,
            os.path.join(WEIGHTS_DIR, "stable-diffusion-v1-5"),
            os.path.join(WEIGHTS_DIR, "sd-vae-ft-mse"),
            os.path.join(WEIGHTS_DIR, "diffuEraser"),
            ckpt="2-Step",
        )
        print(f"[LAZY] DiffuEraser OK ({time.time() - t:.1f}s)", flush=True)
    return _DIFFU


# DOUĂ cititoare: latin (en+ro) + chinez (ch_sim+en).
# Sursele sunt Douyin/RedNote/TikTok: caption-urile chinezești citite de modelul
# latin ieșeau gunoi cu conf<0.25 → nu se ștergeau NICIODATĂ. Modelul ch_sim le
# citește, dar dă conf ~0 chiar când citește corect → pragul de conf nu se aplică
# ideogramelor (vezi detect_text_ocr); latinul păstrează pragul normal.
_OCR, _OCR_ZH = None, None
def get_ocr():
    global _OCR
    if _OCR is None:
        t = time.time()
        import easyocr
        _OCR = easyocr.Reader(["en", "ro"], gpu=(DEVICE == "cuda"), verbose=False)
        print(f"[LAZY] EasyOCR en+ro OK ({time.time() - t:.1f}s)", flush=True)
    return _OCR


def get_ocr_zh():
    global _OCR_ZH
    if _OCR_ZH is None:
        t = time.time()
        import easyocr
        _OCR_ZH = easyocr.Reader(["ch_sim", "en"], gpu=(DEVICE == "cuda"), verbose=False)
        print(f"[LAZY] EasyOCR ch_sim OK ({time.time() - t:.1f}s)", flush=True)
    return _OCR_ZH


FLORENCE_ID = os.environ.get("FLORENCE_MODEL", "microsoft/Florence-2-large")
_FLORENCE, _FLORENCE_PROC = None, None
def get_florence():
    """Florence-2 — doar dacă jobul cere logos/watermarks/prompturi custom."""
    global _FLORENCE, _FLORENCE_PROC
    if _FLORENCE is None:
        t = time.time()
        print("[LAZY] Încărcare Florence-2...", flush=True)
        from unittest.mock import patch
        from transformers import AutoModelForCausalLM, AutoProcessor
        from transformers.dynamic_module_utils import get_imports

        def _fixed_get_imports(filename):
            # Florence-2 declară flash_attn ca import obligatoriu; nu e necesar cu SDPA.
            imports = get_imports(filename)
            if "flash_attn" in imports:
                imports.remove("flash_attn")
            return imports

        with patch("transformers.dynamic_module_utils.get_imports", _fixed_get_imports):
            _FLORENCE = AutoModelForCausalLM.from_pretrained(
                FLORENCE_ID, trust_remote_code=True, torch_dtype=DTYPE,
                attn_implementation="sdpa",
            ).to(DEVICE).eval()
            _FLORENCE_PROC = AutoProcessor.from_pretrained(FLORENCE_ID, trust_remote_code=True)
        print(f"[LAZY] Florence-2 OK ({time.time() - t:.1f}s)", flush=True)
    return _FLORENCE, _FLORENCE_PROC


print(f"[INIT] Worker gata în {time.time() - _T_BOOT:.1f}s (restul modelelor se încarcă la cerere)", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# DETECȚIE
# ═════════════════════════════════════════════════════════════════════════════
def _clamp_box(x1, y1, x2, y2, w, h, pad=BOX_PAD, pad_x=None, pad_y=None):
    px = pad if pad_x is None else pad_x
    py = pad if pad_y is None else pad_y
    x1 = max(0, int(x1) - px); y1 = max(0, int(y1) - py)
    x2 = min(w, int(x2) + px); y2 = min(h, int(y2) + py)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return (x1, y1, x2, y2)


_CJK_RE = None
def _has_cjk(text):
    global _CJK_RE
    if _CJK_RE is None:
        import re
        _CJK_RE = re.compile(r"[㐀-䶿一-鿿]")
    return bool(_CJK_RE.search(text))


def detect_text_ocr(frame_bgr, w, h):
    """EasyOCR pe un frame → listă de box-uri (x1,y1,x2,y2). Downscale pt viteză.
    Rulează AMBELE cititoare: latin cu prag de conf normal; din cel chinezesc se
    păstrează doar box-urile cu ideograme, FĂRĂ prag de conf — ch_sim raportează
    conf ~0 chiar la citiri corecte, iar pt ștergere contează regiunea, nu textul."""
    scale = 1.0
    img = frame_bgr
    if w > 1280:
        scale = 1280.0 / w
        img = cv2.resize(frame_bgr, (1280, int(h * scale)))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hits = []
    for (bbox, text, conf) in get_ocr().readtext(rgb, detail=1):
        if conf >= OCR_CONF and str(text).strip():
            hits.append(bbox)
    for (bbox, text, conf) in get_ocr_zh().readtext(rgb, detail=1):
        if _has_cjk(str(text)):
            hits.append(bbox)

    boxes = []
    for bbox in hits:
        pts = np.array(bbox, dtype=np.float32) / scale
        x, y, bw, bh = cv2.boundingRect(pts.astype(np.int32))
        # un „text" mai mare de 25% din frame = fals pozitiv OCR (aceeași regulă ca la Florence)
        if bw * bh > w * h * MAX_BOX_AREA_PCT:
            continue
        # Padding PROPORȚIONAL cu înălțimea textului (nu 6px fix): box-urile EasyOCR
        # sunt strânse fix pe glife, iar prima/ultima literă ies adesea în afara lor
        # (fonturi mari, litere cu diacritice/descendente, pop-in animat între
        # keyframes) → rămâneau arse în video. Orizontal ~o lățime de literă.
        pad_x = max(BOX_PAD, int(round(bh * 0.55)))
        pad_y = max(BOX_PAD, int(round(bh * 0.30)))
        b = _clamp_box(x, y, x + bw, y + bh, w, h, pad_x=pad_x, pad_y=pad_y)
        if b:
            boxes.append(b)
    return boxes


@torch.inference_mode()
def detect_florence(frame_bgr, w, h, prompts):
    """Florence-2 phrase grounding → box-uri pt logo/watermark/prompturi custom."""
    model, proc = get_florence()
    task = "<CAPTION_TO_PHRASE_GROUNDING>"
    text = task + " ".join(p.rstrip(".") + "." for p in prompts)
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = proc(text=text, images=pil, return_tensors="pt").to(DEVICE)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(DTYPE)
    ids = model.generate(
        input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
        # num_beams=1 (greedy) în loc de 3: beam search triplează costul generării
        # pentru un task de grounding unde ieșirea e o listă de box-uri, nu proză.
        max_new_tokens=256, num_beams=FLORENCE_BEAMS, do_sample=False,
    )
    out = proc.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = proc.post_process_generation(out, task=task, image_size=(w, h))
    result = parsed.get(task, {})
    boxes = []
    frame_area = float(w * h)
    for bbox in result.get("bboxes", []):
        x1, y1, x2, y2 = bbox
        # box-urile care acoperă aproape tot frame-ul = halucinație de grounding
        if (x2 - x1) * (y2 - y1) > frame_area * MAX_BOX_AREA_PCT:
            continue
        b = _clamp_box(x1, y1, x2, y2, w, h)
        if b:
            boxes.append(b)
    return boxes


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def _anchor_drift(members):
    """Cât de mult „călătorește" un cluster prin cadru (px).
    Textul ARS pe ecran stă pe loc — dar când fraza se schimbă, box-ul își schimbă
    lățimea; în funcție de aliniere rămâne fixă marginea stângă / centrul / dreapta.
    Luăm deci pe fiecare axă MINIMUL intervalului de variație dintre cele 3 ancore
    (min / centru / max) — dacă și cea mai stabilă ancoră se plimbă mult, textul e
    lipit de un obiect din scenă (tricou, produs, mașină), nu de ecran."""
    def spread(vals):
        return max(vals) - min(vals)
    xs1 = [b[0] for _, b in members]; xs2 = [b[2] for _, b in members]
    ys1 = [b[1] for _, b in members]; ys2 = [b[3] for _, b in members]
    cxs = [(a + b) / 2 for a, b in zip(xs1, xs2)]
    cys = [(a + b) / 2 for a, b in zip(ys1, ys2)]
    dx = min(spread(xs1), spread(cxs), spread(xs2))
    dy = min(spread(ys1), spread(cys), spread(ys2))
    return (dx * dx + dy * dy) ** 0.5


def group_static_boxes(per_frame_boxes, min_ratio, n_frames_detected, frame_diag=None):
    """
    Grupează box-urile care apar (IoU>0.5) în ≥min_ratio din keyframes → statice.
    Returnează (static_boxes, per_frame_dynamic).

    Anti-distrugere (fix „video terci pe RedNote"):
      • clusterele care DERIVEAZĂ prin cadru (text pe haine/obiecte filmate) se
        ARUNCĂ — nu-s overlay ars, iar inpainting-ul lor tocă subiectul video;
      • box-urile dinamice folosesc box-ul DETECTAT la fiecare keyframe, nu
        union-ul clusterului — union-ul creștea în lanț (IoU cu el însuși) până
        acoperea jumătate de frame și se ștanța pe toate cadrele din interval.
    """
    clusters = []  # fiecare: {"box": union (doar pt matching/static), "hits": set, "members": [(fi, box)]}
    for fi, boxes in per_frame_boxes.items():
        for b in boxes:
            placed = False
            for c in clusters:
                if _iou(c["box"], b) > 0.5:
                    x1 = min(c["box"][0], b[0]); y1 = min(c["box"][1], b[1])
                    x2 = max(c["box"][2], b[2]); y2 = max(c["box"][3], b[3])
                    c["box"] = (x1, y1, x2, y2)
                    c["hits"].add(fi)
                    c["members"].append((fi, b))
                    placed = True
                    break
            if not placed:
                clusters.append({"box": b, "hits": {fi}, "members": [(fi, b)]})

    static, dynamic = [], {fi: [] for fi in per_frame_boxes}
    n_drifting = 0
    for c in clusters:
        if frame_diag and len(c["members"]) >= 3:
            drift = _anchor_drift(c["members"])
            if drift > DRIFT_MAX_PCT * frame_diag:
                n_drifting += 1
                continue
        if len(c["hits"]) >= max(2, min_ratio * n_frames_detected):
            static.append(c["box"])
        else:
            for fi, b in c["members"]:
                dynamic[fi].append(b)
    if n_drifting:
        print(f"[DETECT] {n_drifting} cluster(e) în mișcare ignorate (text pe obiecte, nu overlay)", flush=True)
    return static, dynamic


def run_detection(video_path, w, h, fps, n_frames, targets, extra_prompts):
    """Detecție pe keyframes → (static_boxes, dynamic_by_kf, kf_indices)."""
    cap = cv2.VideoCapture(video_path)
    step_ocr      = max(1, int(round(fps * DETECT_INTERVAL)))
    step_florence = max(1, int(round(fps * FLORENCE_INTERVAL)))

    want_text  = "captions" in targets or "watermarks" in targets
    want_logos = "logos" in targets or "watermarks" in targets or extra_prompts

    florence_prompts = []
    if "logos" in targets:
        florence_prompts += ["logo", "channel logo"]
    if "watermarks" in targets:
        florence_prompts += ["watermark", "semi-transparent watermark"]
    florence_prompts += list(extra_prompts or [])

    ocr_hits, flo_hits = {}, {}
    kf_indices = []
    # Citire SECVENȚIALĂ, sărind cadrele nedorite cu grab() (decodare sărită).
    # Înainte era `cap.set(CAP_PROP_POS_FRAMES, idx)` per keyframe: fiecare seek
    # golește decodorul, sare la keyframe-ul H.264 anterior și re-decodează
    # înainte — de zeci de ori pe job. grab() nu decodează cadrul deloc.
    idx = 0
    next_kf = 0
    t_det = time.time()
    while idx < n_frames:
        if idx < next_kf:
            if not cap.grab():
                break
            idx += 1
            continue
        ret, frame = cap.read()
        if not ret:
            break
        kf_indices.append(idx)
        if want_text:
            ocr_hits[idx] = detect_text_ocr(frame, w, h)
        if want_logos and florence_prompts and idx % step_florence < step_ocr:
            flo_hits[idx] = detect_florence(frame, w, h, florence_prompts)
        next_kf = idx + step_ocr
        idx += 1
    cap.release()

    # zero cadre citite = video nedecodabil, NU "nimic detectat" — altfel jobul
    # raportează succes fals și clientul retrimite la nesfârșit
    if not kf_indices:
        raise ValueError("Nu am putut citi niciun cadru din video (decodare eșuată)")

    n_kf = max(1, len(kf_indices))
    frame_diag = (w * w + h * h) ** 0.5
    static_boxes, dynamic_by_kf = [], {fi: [] for fi in kf_indices}

    if ocr_hits:
        ocr_static, ocr_dyn = group_static_boxes(ocr_hits, STATIC_RATIO, len(ocr_hits), frame_diag)
        # dacă userul NU vrea captions, păstrăm din OCR doar textul STATIC (watermark text)
        if "captions" in targets:
            for fi, bs in ocr_dyn.items():
                dynamic_by_kf.setdefault(fi, []).extend(bs)
        static_boxes += ocr_static

    if flo_hits:
        # Florence: logo/watermark = static prin definiție → cerem persistență în
        # ≥50% din frame-urile Florence ca să eliminăm halucinațiile pe obiecte
        flo_static, _flo_dyn = group_static_boxes(flo_hits, 0.5, len(flo_hits), frame_diag)
        static_boxes += flo_static

    # union-ul unui cluster static nu are voie să depășească plafonul de arie —
    # un „watermark" de un sfert de ecran e o grupare scăpată de sub control
    frame_area = float(w * h)
    static_boxes = [b for b in static_boxes
                    if (b[2] - b[0]) * (b[3] - b[1]) <= frame_area * MAX_BOX_AREA_PCT]

    n_static = len(static_boxes)
    n_dynamic = sum(len(v) for v in dynamic_by_kf.values())
    print(f"[DETECT] keyframes={n_kf} static={n_static} dynamic_hits={n_dynamic} "
          f"({time.time() - t_det:.1f}s)", flush=True)
    return static_boxes, dynamic_by_kf, kf_indices


# ═════════════════════════════════════════════════════════════════════════════
# MĂȘTI TEMPORALE
# ═════════════════════════════════════════════════════════════════════════════
def build_mask_video(mask_path, w, h, fps, n_frames, static_boxes, dynamic_by_kf, kf_indices, workdir):
    """
    Scrie mask.mp4 (alb = de șters). Pentru fiecare frame:
      static  → mereu activ
      dinamic → union(box-urile de la keyframe-ul anterior și următor)
                (dilatare temporală ± un interval — sigur pt captions care se schimbă)
    """
    total_active = 0
    kf_sorted = sorted(kf_indices)

    def boxes_for_frame(fidx):
        boxes = list(static_boxes)
        prev_kf = next_kf = None
        for k in kf_sorted:
            if k <= fidx:
                prev_kf = k
            if k >= fidx and next_kf is None:
                next_kf = k
        for k in (prev_kf, next_kf):
            if k is not None:
                boxes.extend(dynamic_by_kf.get(k, []))
        return boxes

    base_static = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in static_boxes:
        base_static[y1:y2, x1:x2] = 255

    # Cadrele merg direct în stdin-ul lui ffmpeg ca raw gray. Înainte se scria
    # câte un PNG full-res per cadru pe disc (2341 fișiere la un clip de 78s,
    # 608 fișiere de 8Mpx la 4K) și abia apoi se encoda — I/O pur, plătit ca
    # timp GPU. `-qp 0` = lossless: masca e binară, marginile ei nu au voie să
    # fie mâncate de compresie (crf 12 le înmuia, de-aia era nevoie de lut).
    proc = subprocess.Popen([
        "ffmpeg", "-y", "-nostats", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-framerate", f"{fps}",
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0",
        "-pix_fmt", "yuv420p",
        mask_path,
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    n_capped = 0
    try:
        for fidx in range(n_frames):
            boxes = boxes_for_frame(fidx)
            mask = base_static.copy()
            for (x1, y1, x2, y2) in boxes:
                mask[y1:y2, x1:x2] = 255
            # plasă de siguranță: dacă masca ar acoperi >MASK_MAX_COVERAGE din frame,
            # inpainting-ul nu mai are din ce reconstrui → scoatem box-urile cele mai
            # mari până coborâm sub plafon (mai bine rămâne puțin text decât video terci)
            if mask.mean() / 255.0 > MASK_MAX_COVERAGE:
                n_capped += 1
                boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))  # crescător după arie
                while boxes:
                    mask = base_static.copy()
                    for (x1, y1, x2, y2) in boxes:
                        mask[y1:y2, x1:x2] = 255
                    if mask.mean() / 255.0 <= MASK_MAX_COVERAGE:
                        break
                    boxes.pop()  # scoate box-ul cel mai mare (ultimul)
            if mask.any():
                total_active += 1
            proc.stdin.write(mask.tobytes())
    finally:
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Encodarea măștii a eșuat: {err[-300:]}")

    if n_capped:
        print(f"[MASK] {n_capped} frame-uri plafonate la {MASK_MAX_COVERAGE:.0%} (box-urile cele mai mari scoase)", flush=True)
    print(f"[MASK] {n_frames} frames, {total_active} cu mască activă → {mask_path}", flush=True)
    return total_active


# ═════════════════════════════════════════════════════════════════════════════
# INPAINTING + FINISARE
# ═════════════════════════════════════════════════════════════════════════════
def _proc_size(w, h, max_side=None):
    """Rezoluția la care rulează ProPainter — depinde DOAR de rezoluția sursă,
    nu și de lungimea clipului. Lungimea se rezolvă prin chunking temporal
    (vezi run_inpainting), nu prin scăderea rezoluției pe tot videoul."""
    side = float(max_side or PROC_MAX_SIDE)
    ratio = min(1.0, side / float(max(w, h)))
    pw = max(64, int(w * ratio)) // 2 * 2
    ph = max(64, int(h * ratio)) // 2 * 2
    return pw, ph


def _chunk_frames(pw, ph):
    """Câte cadre încap într-o bucată la rezoluția dată, derivat din VRAM-ul REAL.
    Constanta veche (600 cadre) era calibrată pentru 24GB și se aplica și pe
    plăci de 80GB — de unde rezoluțiile absurde pe clipurile lungi.

    PROC_FRAME_BUDGET, dacă e setat, e doar un PLAFON suplimentar — nu poate
    ridica limita peste ce încape în VRAM. Altfel o valoare pusă din env ar
    garanta un OOM la primul clip lung."""
    n = int(VRAM_GB * PXFRAMES_PER_GB / float(pw * ph))
    if PROC_FRAME_BUDGET > 0:
        n = min(n, PROC_FRAME_BUDGET)
    return max(60, n)


def _extract_segment(src, dst, start_frame, end_frame, lossless=False):
    """Taie [start_frame, end_frame) din src. `trim` pe numere de cadre e exact
    pe CFR (inputul e normalizat la CFR întreg la intrare)."""
    subprocess.run([
        "ffmpeg", "-y", "-nostats", "-loglevel", "error",
        "-i", src,
        "-vf", f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS",
        "-an",
        *(["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0"] if lossless
          else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "14"]),
        "-pix_fmt", "yuv420p", dst,
    ], check=True)


def _concat_segments(seg_paths, drops, out_path, fps):
    """Lipește bucățile aruncând primele `drops[i]` cadre din fiecare (cadrele de
    suprapunere, deja acoperite de bucata anterioară). Un singur re-encode."""
    if len(seg_paths) == 1 and drops[0] == 0:
        shutil.move(seg_paths[0], out_path)
        return
    cmd = ["ffmpeg", "-y", "-nostats", "-loglevel", "error"]
    for p in seg_paths:
        cmd += ["-i", p]
    parts, labels = [], []
    for i, d in enumerate(drops):
        parts.append(f"[{i}:v]trim=start_frame={d},setpts=PTS-STARTPTS[v{i}]")
        labels.append(f"[v{i}]")
    parts.append("".join(labels) + f"concat=n={len(seg_paths)}:v=1:a=0[out]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[out]",
            "-r", f"{fps}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "14",
            "-pix_fmt", "yuv420p", out_path]
    subprocess.run(cmd, check=True)


def run_inpainting(video_path, mask_path, workdir, duration_s, max_img_size, quality, w, h, n_frames, fps):
    """quality="fast" → doar ProPainter (~2 min pt 20s video, foarte bun pe captions).
    quality="max"  → + rafinare DiffuEraser (calitate maximă, dar de 3-5x mai lent).

    Inpainting-ul rulează la _proc_size (independent de lungime); clipurile care
    nu încap în VRAM se taie în bucăți temporale cu CHUNK_OVERLAP cadre de
    suprapunere, procesate una câte una la rezoluție PLINĂ, apoi lipite.
    finalize() pune rezultatul înapoi peste originalul full-res doar sub mască."""
    priori_path = os.path.join(workdir, "priori.mp4")
    result_path = os.path.join(workdir, "diffueraser_out.mp4")

    def _attempt(max_side):
        pw, ph = _proc_size(w, h, max_side)
        chunk = _chunk_frames(pw, ph)
        # dilatarea se dă lui ProPainter în pixeli DE PROCESARE, dar o exprimăm
        # în pixeli la rezoluția originală → efectul rămâne același indiferent
        # de scalare (înainte, 8px la 182x324 însemnau 32px pe video-ul real)
        ratio = pw / float(w)
        dil = max(1, int(round(MASK_DILATE_PX * ratio)))
        seg_dir = os.path.join(workdir, "segments")
        shutil.rmtree(seg_dir, ignore_errors=True)
        os.makedirs(seg_dir, exist_ok=True)

        # resize_ratio=1.0 + width/height explicite → dezactivăm downscale-ul
        # intern nedeterminist al DiffuEraser (default 0.6, ×0.5 peste 960px)
        def _priori(vid, msk, dst, seg_frames):
            PROPAINTER.forward(
                vid, msk, dst,
                resize_ratio=1.0, width=pw, height=ph,
                video_length=int(seg_frames / float(fps)) + 1,
                ref_stride=10, neighbor_length=10, subvideo_length=50,
                mask_dilation=dil,
            )

        if n_frames <= chunk:
            print(f"[INPAINT] ProPainter @ {pw}x{ph} ({n_frames} cadre, dilate={dil}px)...", flush=True)
            _priori(video_path, mask_path, priori_path, n_frames)
            return

        # Bucata i PRODUCE cadrele [out_start, e) și le mai PROCESEAZĂ pe cele
        # `lead` dinaintea lor doar ca context temporal (se aruncă la lipire).
        # Intervalele produse se cap-coadă exact, fără suprapunere în output.
        overlap = min(CHUNK_OVERLAP, max(0, chunk // 4))
        plan = []
        out_start = 0
        while out_start < n_frames:
            lead = min(overlap, out_start)
            s = out_start - lead
            e = min(n_frames, s + chunk)
            # coada scurtă se lipește de bucata curentă în loc să devină o bucată
            # separată: altfel ultima rulare ProPainter procesa `overlap`+2 cadre
            # ca să producă 1-2 utile — o trecere întreagă de GPU degeaba
            if 0 < n_frames - e <= overlap:
                e = n_frames
            plan.append((s, e, lead))
            out_start = e
        print(f"[INPAINT] ProPainter @ {pw}x{ph} ({n_frames} cadre, dilate={dil}px) "
              f"→ {len(plan)} bucăți × max {chunk} cadre (overlap {overlap})", flush=True)

        seg_outs, drops = [], []
        for i, (s, e, lead) in enumerate(plan):
            seg_v = os.path.join(seg_dir, f"v{i:03d}.mp4")
            seg_m = os.path.join(seg_dir, f"m{i:03d}.mp4")
            seg_o = os.path.join(seg_dir, f"o{i:03d}.mp4")
            _extract_segment(video_path, seg_v, s, e)
            _extract_segment(mask_path, seg_m, s, e, lossless=True)
            print(f"[INPAINT]   bucata {i+1}/{len(plan)}: cadre {s}-{e}", flush=True)
            _priori(seg_v, seg_m, seg_o, e - s)
            seg_outs.append(seg_o)
            drops.append(lead)
            os.remove(seg_v)
            os.remove(seg_m)
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        _concat_segments(seg_outs, drops, priori_path, fps)
        shutil.rmtree(seg_dir, ignore_errors=True)

    oom = False
    try:
        _attempt(PROC_MAX_SIDE)
    except torch.cuda.OutOfMemoryError:
        # NU reîncercăm aici: cât timp suntem în except, traceback-ul activ ține
        # referințe la tensorii din ProPainter → empty_cache() nu poate elibera
        # VRAM-ul și retry-ul murea tot cu OOM ("22.5 GiB in use" la reîncercare)
        oom = True
    if oom:
        gc.collect()
        torch.cuda.empty_cache()
        retry_side = max(MIN_PROC_SIDE, int(PROC_MAX_SIDE * 0.7))
        print(f"[INPAINT] CUDA OOM → reîncerc cu latura lungă {retry_side}", flush=True)
        _attempt(retry_side)

    if quality != "max":
        print("[INPAINT] quality=fast → sar peste DiffuEraser", flush=True)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return priori_path

    # DiffuEraser rulează la max_img_size, nu la _proc_size → dilatarea lui se
    # convertește la ACEA scară, tot din MASK_DILATE_PX (pixeli la full res)
    diffu_ratio = min(1.0, max_img_size / float(max(w, h)))
    diffu_dil = max(1, int(round(MASK_DILATE_PX * diffu_ratio)))
    print(f"[INPAINT] DiffuEraser refine (dilate={diffu_dil}px)...", flush=True)
    get_diffu().forward(
        video_path, mask_path, priori_path, result_path,
        max_img_size=max_img_size,
        video_length=int(duration_s) + 1,
        mask_dilation_iter=diffu_dil,
        guidance_scale=None,
    )
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return result_path


def finalize(result_path, original_path, mask_path, out_path, w, h):
    """Compune rezultatul inpaint (procesat la rezoluție redusă) înapoi peste
    originalul full-res DOAR în zonele mascate + remux audio original.
    Înainte, TOT videoul era upscalat din rezoluția de procesare (~576p) —
    acum doar pixelii de sub mască vin din inpaint, restul rămân 1:1 originali.
    Masca e binarizată explicit (lut) fiindcă mp4-ul ei e limited-range
    (alb = Y 235, nu 255 → ar lăsa 8% din textul original să transpară).
    Masca se dilată aici DOAR cât să acopere marginea de anti-aliasing a
    literelor. Înainte era gblur sigma=6 + prag 16 ≈ 9px, care se ADUNAU peste
    dilatarea ProPainter (32px efectivi pe clipurile lungi) → total ~40-70px
    dincolo de glife. Acum ProPainter dilatează controlat (MASK_DILATE_PX la
    rezoluția reală), deci aici e nevoie doar de ~2px + un feather scurt.
    Pragul de la începutul lanțului rămâne: mp4-ul măștii e limited-range
    (alb = Y 235, nu 255 → altfel 8% din textul original ar transpărea)."""
    # gblur sigma=σ urmat de prag t dilată cu ≈ σ·Φ⁻¹(1−t/255) pixeli.
    # sigma=2 + prag 40 → ~2px. Feather-ul final rămâne subțire.
    subprocess.run([
        "ffmpeg", "-y", "-nostats", "-loglevel", "error",
        "-i", original_path,
        "-i", result_path,
        "-i", mask_path,
        "-filter_complex",
        f"[1:v]scale={w}:{h}:flags=lanczos,setsar=1,format=yuva420p[res];"
        f"[2:v]scale={w}:{h},format=gray,lut=c0='if(gt(val,40),255,0)',"
        f"gblur=sigma=2,lut=c0='if(gt(val,40),255,0)',gblur=sigma=1.5[m];"
        f"[res][m]alphamerge[ov];"
        f"[0:v][ov]overlay=shortest=1,format=yuv420p[out]",
        "-map", "[out]", "-map", "0:a:0?",
        *_venc(crf="18"),
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-shortest",
        out_path,
    ], check=True)


# ═════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═════════════════════════════════════════════════════════════════════════════
def fetch_video(job_input, workdir):
    video_path = os.path.join(workdir, "input.mp4")
    if job_input.get("video_url"):
        r = requests.get(job_input["video_url"], timeout=120, stream=True)
        r.raise_for_status()
        with open(video_path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
    elif job_input.get("video_base64"):
        with open(video_path, "wb") as f:
            f.write(base64.b64decode(job_input["video_base64"]))
    else:
        raise ValueError("Lipsește video_url sau video_base64")
    return video_path


# codecuri pe care lanțul cv2/ffmpeg/imageio le decodează sigur; AV1 (TikTok/
# Douyin) dădea "Get current frame error" în OpenCV → zero cadre citite →
# jobul raporta fals "nothing_detected" și clientul reîncerca la nesfârșit,
# plătind un cold start GPU pentru fiecare încercare
SAFE_CODECS = {"h264", "hevc", "mpeg4", "mjpeg", "vp8", "vp9"}


def _probe_codec(video_path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=30,
        )
        return (out.stdout or "").strip().splitlines()[0].lower() if (out.stdout or "").strip() else ""
    except Exception:
        return ""


def _can_read_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ok, _ = cap.read()
    cap.release()
    return ok


def normalize_input(video_path, workdir):
    """Normalizează inputul o singură dată, la intrare — două motive:

    FPS: DiffuEraser cere fps IDENTIC între video, mască și priori (read_priori
    compară strict). FPS-urile fracționare din filmări de telefon (ex. 30.05,
    29.97) se cuantizează diferit prin lanțul cv2/ffmpeg/imageio → re-eșantionăm
    la fps ÎNTREG constant (CFR), plafonat la MAX_FPS (60fps = dublu VRAM → OOM).

    CODEC: OpenCV nu decodează AV1 (metadatele merg, cadrele nu) → transcodăm
    în H.264 cu ffmpeg-ul de sistem (are dav1d). Dacă nici acesta nu poate
    decoda, aruncăm eroare EXPLICITĂ — niciodată succes fals."""
    codec = _probe_codec(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    target = max(10, min(MAX_FPS, int(round(fps)) or 30))

    bad_codec = codec not in SAFE_CODECS or not _can_read_frame(video_path)
    if not bad_codec and abs(fps - target) < 0.01:
        return video_path

    norm_path = os.path.join(workdir, "input_cfr.mp4")
    print(f"[NORM] codec={codec or '?'} {fps:.3f}fps → h264 {target}fps CFR", flush=True)
    proc = subprocess.run([
        "ffmpeg", "-y", "-nostats", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"fps={target}",
        *_venc(crf="16", preset="veryfast"),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        norm_path,
    ], capture_output=True, text=True)
    if proc.returncode != 0 or not _can_read_frame(norm_path):
        if proc.stderr:
            print(f"[NORM] ffmpeg stderr: {proc.stderr[-400:]}", flush=True)
        raise ValueError(
            f"Nu pot decoda videoul (codec: {codec or 'necunoscut'}). "
            "Re-exportă-l ca MP4 (H.264) și încearcă din nou."
        )
    return norm_path


def probe(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Nu pot deschide videoul")
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0 or w <= 0 or h <= 0:
        raise ValueError("Metadate video invalide")
    return w, h, fps, n, n / fps


def deliver(out_path, job_input):
    size_mb = round(os.path.getsize(out_path) / 1024 / 1024, 2)
    cb = job_input.get("callback_url")
    if cb and job_input.get("job_id"):
        with open(out_path, "rb") as f:
            r = requests.post(
                cb,
                files={"video": ("result.mp4", f, "video/mp4")},
                data={"job_id": str(job_input["job_id"])},
                timeout=300,
            )
        r.raise_for_status()
        return {"result_uploaded": True, "size_mb": size_mb}
    with open(out_path, "rb") as f:
        return {"video_base64": base64.b64encode(f.read()).decode()}


# ═════════════════════════════════════════════════════════════════════════════
# HANDLER
# ═════════════════════════════════════════════════════════════════════════════
def handler(job):
    job_input = job.get("input", {}) or {}
    workdir = tempfile.mkdtemp(prefix="autoeraser_")
    try:
        targets = job_input.get("targets") or ["captions", "logos", "watermarks"]
        extra_prompts = job_input.get("extra_prompts") or []
        max_img_size = int(job_input.get("max_img_size") or 960)
        max_img_size = max(512, min(1920, max_img_size))
        quality = str(job_input.get("quality") or "fast").lower()

        video_path = fetch_video(job_input, workdir)

        # durata se verifică pe originalul brut, ÎNAINTE de transcodare — nu
        # plătim normalize pentru un video pe care oricum îl respingem
        # (metadatele cv2 merg și pe codecuri pe care nu le putem decoda)
        _, _, _, _, duration = probe(video_path)
        if duration > MAX_SECONDS:
            return {"error": f"Video prea lung ({duration:.0f}s). Maxim: {MAX_SECONDS:.0f}s."}

        video_path = normalize_input(video_path, workdir)
        w, h, fps, n_frames, duration = probe(video_path)
        print(f"[JOB] {w}x{h} @ {fps:.2f}fps, {n_frames} frames, {duration:.1f}s, targets={targets}, quality={quality}", flush=True)

        static_boxes, dynamic_by_kf, kf_indices = run_detection(
            video_path, w, h, fps, n_frames, targets, extra_prompts
        )

        n_dynamic = sum(len(v) for v in dynamic_by_kf.values())
        if not static_boxes and n_dynamic == 0:
            print("[JOB] Nimic detectat — returnez fără procesare", flush=True)
            return {"nothing_detected": True}

        # eliberăm VRAM-ul rămas de la detecție (Florence/EasyOCR) înainte de inpainting
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        mask_path = os.path.join(workdir, "mask.mp4")
        build_mask_video(mask_path, w, h, fps, n_frames,
                         static_boxes, dynamic_by_kf, kf_indices, workdir)

        result_path = run_inpainting(video_path, mask_path, workdir, duration,
                                     max_img_size, quality, w, h, n_frames, fps)

        out_path = os.path.join(workdir, "final.mp4")
        finalize(result_path, video_path, mask_path, out_path, w, h)

        out = deliver(out_path, job_input)
        out["detections"] = {
            "static_boxes": len(static_boxes),
            "dynamic_hits": n_dynamic,
            "keyframes": len(kf_indices),
        }
        return out

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
