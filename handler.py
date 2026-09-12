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
import math

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
# ── Decupaj pe banda de text (ROI) ───────────────────────────────────────────
# Caption-urile stau într-o bandă, dar ProPainter primea cadrul întreg. La un
# 720x1280 cu text jos, ~75% din pixelii măcinați n-aveau nicio mască pe ei.
ROI_PAD_PCT      = float(os.environ.get("ROI_PAD_PCT", "0.18"))   # context în jurul benzii (din latura ei)
ROI_MIN_PAD      = int(os.environ.get("ROI_MIN_PAD", "40"))       # dar niciodată mai puțin de atât, în px
ROI_MAX_AREA_PCT = float(os.environ.get("ROI_MAX_AREA_PCT", "0.75"))  # peste atât, decupajul n-aduce nimic
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
DRIFT_MAX_PCT    = float(os.environ.get("DRIFT_MAX_PCT", "0.04"))    # cât poate varia poziția unui cluster (fracție din diagonală) ca să fie ștanțat STATIC pe tot clipul
DRIFT_TRIM       = float(os.environ.get("DRIFT_TRIM", "0.10"))       # cozile ignorate la măsurarea variației (0 = max-min, ca înainte)
DRIFT_STEP_PCT   = float(os.environ.get("DRIFT_STEP_PCT", "0.012"))  # cât se poate mișca ACELAȘI text între două keyframe-uri (fracție din diagonală) ca să nu fie text pe obiect
SAME_TEXT_TOL    = float(os.environ.get("SAME_TEXT_TOL", "0.12"))    # două box-uri cu lățime/înălțime la ±12% = același text (pt măsurarea mișcării)
MASK_MAX_COVERAGE = float(os.environ.get("MASK_MAX_COVERAGE", "0.25")) # plafonul măștii pe un frame — peste, se sacrifică întâi ce NU e caption
                                                                       # (0.40 lăsa inpainting-ul fără sursă: 40% din cadru șters = terci)
# ── Ce acceptăm din OCR ca text de șters ─────────────────────────────────────
# EasyOCR taie o linie de caption în bucăți și dă fiecăreia încrederea de
# RECUNOAȘTERE. Pentru ștergere contează doar UNDE e textul, nu dacă l-a citit
# corect: pe fonturile de caption (bold + contur + umbră) cuvinte întregi ies cu
# conf 0.05-0.20 („just" din „Lee, just one inch", „…up, driving") și erau
# aruncate → rămâneau arse în mijlocul liniei. O bucată slabă se acceptă acum
# dacă stă pe ACEEAȘI LINIE cu text sigur (sau e rândul de deasupra/dedesubt al
# aceluiași caption), ori dacă în ±WEAK_TEMPORAL_KF keyframe-uri s-a citit text
# sigur exact pe linia aceea.
HORIZ_DEG        = float(os.environ.get("HORIZ_DEG", "10"))         # sub atâtea grade = linie orizontală (caption); peste = text rotit
LINE_GAP         = float(os.environ.get("LINE_GAP", "1.2"))         # distanța max între bucățile aceleiași linii (× înălțimea textului)
LINE_H_RATIO     = float(os.environ.get("LINE_H_RATIO", "2.2"))     # raportul max de înălțime între bucățile aceleiași linii
WEAK_TEMPORAL_KF = int(os.environ.get("WEAK_TEMPORAL_KF", "2"))     # câte keyframe-uri înainte/după caută sprijin temporal o bucată slabă

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

def _cuda_usable():
    """CUDA chiar merge? `is_available()` singur nu ajunge: pe unele mașini din
    fleet driverul e mai vechi decât runtime-ul și abia PRIMA alocare crapă cu
    „Error code: 804 — forward compatibility was attempted on non supported HW".
    Verificarea de fitness a RunPod-ului o prinde, dar doar o raportează ca
    warning și lasă workerul să pornească. Deci o facem noi, explicit."""
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() = False"
    try:
        torch.zeros(1024, 1024, device="cuda").sum().item()
        torch.cuda.synchronize()
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# Fără GPU acest worker NU are voie să accepte joburi. Pe CPU, detecția
# (EasyOCR + Florence-2) și ProPainter merg de 10-20× mai lent: în loguri, un
# clip de 19s a stat 353s doar în DETECT și n-a terminat niciodată — jobul
# expira sau workerul se reciclă, clientul retrimite, iar GPU-time-ul se
# facturează integral pentru zero rezultat. Mai bine murim la pornire: RunPod
# marchează workerul nesănătos și mută jobul pe altă mașină, în câteva secunde.
_CUDA_OK, _CUDA_WHY = _cuda_usable()
ALLOW_CPU = os.environ.get("ALLOW_CPU", "0") == "1"

DEVICE = "cuda" if _CUDA_OK else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

# VRAM-ul real al plăcii — bugetul de inpainting se derivă din el, nu dintr-o
# constantă calibrată pentru 24GB. Fără printul ăsta nu se putea vedea din
# loguri pe ce GPU rulează endpointul (workerul LaMa îl scrie, ăsta nu-l scria).
if _CUDA_OK:
    _PROPS = torch.cuda.get_device_properties(0)
    VRAM_GB = _PROPS.total_memory / (1024 ** 3)
    print(f"[INIT] GPU: {_PROPS.name} — {VRAM_GB:.0f}GB VRAM, sm_{_PROPS.major}{_PROPS.minor}", flush=True)
else:
    VRAM_GB = 8.0
    print(f"[INIT] ❌ GPU INDISPONIBIL pe mașina asta — {_CUDA_WHY}", flush=True)
    if not ALLOW_CPU:
        print("[INIT] ❌ Refuz să pornesc pe CPU (ar arde minute de GPU-time "
              "facturat pentru joburi care oricum nu se termină). "
              "Setează ALLOW_CPU=1 dacă chiar vrei asta.", flush=True)
        sys.exit(1)
    print("[INIT] ⚠ ALLOW_CPU=1 — rulez pe CPU (foarte lent)", flush=True)


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


class TextBox(tuple):
    """(x1, y1, x2, y2) — pe dreptunghi merge toată logica (IoU, clustere, ROI) —
    plus ce trebuie ca masca să fie desenată corect:
      poly  — patrulaterul REAL al textului, cu padding (None = chiar dreptunghiul)
      core  — același contur FĂRĂ padding: glifele propriu-zise, sub care
              plafonul de acoperire nu are voie să strângă
      horiz — linie orizontală de text (caption), nu text rotit
    Textul rotit (watermark diagonal „BAYOAR FILM" care se plimbă prin cadru) are
    dreptunghiul încadrator de 3-4× mai mare decât textul: ștampilat ca dreptunghi
    rodea jumătate de personaj și umplea singur plafonul de 25%."""
    poly = None
    core = None
    horiz = False


def _textbox(rect, poly=None, core=None, horiz=False):
    tb = TextBox(tuple(int(v) for v in rect))
    tb.poly, tb.core, tb.horiz = poly, core, horiz
    return tb


def _draw_box(mask, b):
    """Desenează un box în mască — poligonul real dacă există, altfel dreptunghiul."""
    poly = getattr(b, "poly", None)
    if poly is not None:
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)
    else:
        x1, y1, x2, y2 = b[:4]
        mask[y1:y2, x1:x2] = 255


def _is_caption_line(b):
    return bool(getattr(b, "horiz", False)) and getattr(b, "core", None) is not None


def _box_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _quad_info(quad, strong):
    """Patrulaterul EasyOCR (TL, TR, BR, BL) → geometria de care avem nevoie."""
    p = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    u = p[1] - p[0]
    L = float(np.hypot(u[0], u[1])) or 1.0
    v = p[3] - p[0]
    H = float(np.hypot(v[0], v[1])) or 1.0
    ang = math.degrees(math.atan2(float(u[1]), float(u[0])))
    x1, y1 = p.min(axis=0)
    x2, y2 = p.max(axis=0)
    return {"p": p, "u": u / L, "v": v / H, "L": L, "H": H,
            "rect": (float(x1), float(y1), float(x2), float(y2)),
            "horiz": abs(ang) <= HORIZ_DEG, "strong": strong}


def detect_text_ocr(frame_bgr, w, h):
    """EasyOCR pe un frame → bucăți BRUTE de text (patrulater în px originali +
    dacă e „sigură"). Downscale pt viteză. Rulează AMBELE cititoare: latin cu
    prag de conf; din cel chinezesc doar box-urile cu ideograme, FĂRĂ prag —
    ch_sim raportează conf ~0 chiar la citiri corecte.
    Bucățile latine slabe NU se mai aruncă aici: _text_boxes_from_hits decide,
    cu tot clipul în față, care dintre ele sunt tot text de caption."""
    scale = 1.0
    img = frame_bgr
    if w > 1280:
        scale = 1280.0 / w
        img = cv2.resize(frame_bgr, (1280, int(h * scale)))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hits = []
    for (bbox, text, conf) in get_ocr().readtext(rgb, detail=1):
        strong = conf >= OCR_CONF and bool(str(text).strip())
        hits.append(_quad_info(np.array(bbox, dtype=np.float32) / scale, strong))
    for (bbox, text, conf) in get_ocr_zh().readtext(rgb, detail=1):
        if _has_cjk(str(text)):
            hits.append(_quad_info(np.array(bbox, dtype=np.float32) / scale, True))
    return hits


def _same_line(a, b):
    """Două dreptunghiuri de text pe ACEEAȘI linie (bucăți ale aceleiași fraze)."""
    ha, hb = a[3] - a[1], b[3] - b[1]
    if min(ha, hb) <= 0 or max(ha, hb) > LINE_H_RATIO * min(ha, hb):
        return False
    if min(a[3], b[3]) - max(a[1], b[1]) < 0.5 * min(ha, hb):
        return False
    gap = max(0.0, b[0] - a[2], a[0] - b[2])
    return gap <= LINE_GAP * max(ha, hb)


def _stacked(a, b):
    """b e rândul de deasupra/dedesubt al aceluiași bloc de caption ca a."""
    ha, hb = a[3] - a[1], b[3] - b[1]
    wa, wb = a[2] - a[0], b[2] - b[0]
    if min(ha, hb) <= 0 or max(ha, hb) > 1.8 * min(ha, hb):
        return False
    if min(a[2], b[2]) - max(a[0], b[0]) < 0.3 * min(wa, wb):
        return False
    return max(0.0, b[1] - a[3], a[1] - b[3]) <= 0.6 * max(ha, hb)


def _same_place(a, b):
    """Aceeași linie, în același loc (pt sprijin temporal: ±1s înainte/după)."""
    ha, hb = a[3] - a[1], b[3] - b[1]
    if min(ha, hb) <= 0 or max(ha, hb) > LINE_H_RATIO * min(ha, hb):
        return False
    return (min(a[3], b[3]) - max(a[1], b[1]) >= 0.5 * min(ha, hb)
            and min(a[2], b[2]) - max(a[0], b[0]) > 0)


def _boxes_from_accepted(hits, w, h):
    """Bucățile acceptate ale unui keyframe → TextBox-uri cu padding.
    Bucățile de pe aceeași linie se LIPESC într-un singur box: fără goluri între
    cuvinte, iar clusterele văd linia întreagă, nu jumătăți care „se plimbă"."""
    lines = [list(hh["rect"]) + [hh["H"]] for hh in hits if hh["horiz"]]
    merged = True
    while merged:
        merged = False
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, b = lines[i], lines[j]
                if _same_line(a, b):
                    lines[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), max(a[4], b[4])]
                    del lines[j]
                    merged = True
                    break
            if merged:
                break

    boxes = []
    frame_area = float(w * h)
    for x1, y1, x2, y2, H in lines:
        # un „text" mai mare de 25% din frame = fals pozitiv OCR (aceeași regulă ca la Florence)
        if (x2 - x1) * (y2 - y1) > frame_area * MAX_BOX_AREA_PCT:
            continue
        # Padding PROPORȚIONAL cu înălțimea textului: box-urile EasyOCR sunt strânse
        # pe glife, iar prima/ultima literă, conturul și umbra ies în afara lor.
        pad_x = max(BOX_PAD, int(round(H * 0.55)))
        pad_y = max(BOX_PAD, int(round(H * 0.30)))
        b = _clamp_box(x1, y1, x2, y2, w, h, pad_x=pad_x, pad_y=pad_y)
        core = _clamp_box(x1, y1, x2, y2, w, h, pad=0)
        if b and core:
            boxes.append(_textbox(b, core=core, horiz=True))

    for hh in hits:
        if hh["horiz"]:
            continue
        # text rotit: padding pe axele LUI (lungime/înălțime), desenat ca poligon.
        # Paddingul vine din înălțimea reală a textului, nu din dreptunghiul
        # încadrator (la -20° acela e de 3-4× mai înalt → padding de sute de px).
        if hh["L"] * hh["H"] > frame_area * MAX_BOX_AREA_PCT:
            continue
        p, u, v = hh["p"], hh["u"], hh["v"]
        px = max(BOX_PAD, hh["H"] * 0.55)
        py = max(BOX_PAD, hh["H"] * 0.30)
        poly = np.array([p[0] - u * px - v * py, p[1] + u * px - v * py,
                         p[2] + u * px + v * py, p[3] - u * px + v * py], dtype=np.float32)
        b = _clamp_box(poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max(), w, h, pad=0)
        if b:
            boxes.append(_textbox(b, poly=poly, core=p.copy(), horiz=False))
    return boxes


def _text_boxes_from_hits(raw_by_kf, w, h):
    """Bucățile brute de pe toate keyframe-urile → {kf: [TextBox]}.
    Aici se decide ce bucăți slabe sunt tot caption (vezi WEAK_TEMPORAL_KF)."""
    kfs = sorted(raw_by_kf)
    strong_lines = {k: [hh["rect"] for hh in raw_by_kf[k] if hh["strong"] and hh["horiz"]] for k in kfs}
    out, n_weak = {}, 0
    for i, k in enumerate(kfs):
        acc = [hh for hh in raw_by_kf[k] if hh["strong"]]
        weak = [hh for hh in raw_by_kf[k] if not hh["strong"] and hh["horiz"]]
        near = []
        for j in range(max(0, i - WEAK_TEMPORAL_KF), min(len(kfs), i + WEAK_TEMPORAL_KF + 1)):
            if j != i:
                near += strong_lines[kfs[j]]
        # în lanț: o bucată acceptată poate sprijini la rândul ei una vecină
        changed = True
        while changed and weak:
            changed = False
            still = []
            for hh in weak:
                r = hh["rect"]
                ok = any(a["horiz"] and (_same_line(a["rect"], r) or _stacked(a["rect"], r)) for a in acc)
                if not ok:
                    ok = any(_same_place(s, r) for s in near)
                if ok:
                    acc.append(hh)
                    n_weak += 1
                    changed = True
                else:
                    still.append(hh)
            weak = still
        out[k] = _boxes_from_accepted(acc, w, h)
    if n_weak:
        print(f"[DETECT] +{n_weak} bucăți de text citite nesigur, păstrate (pe linie cu text sigur / în același loc ±{WEAK_TEMPORAL_KF} kf)", flush=True)
    return out


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
    lipit de un obiect din scenă (tricou, produs, mașină), nu de ecran.

    Intervalul se măsoară TĂIND cozile (DRIFT_TRIM), nu cu max-min: EasyOCR mai
    întoarce din când în când doar o bucată din frază („envelope" în loc de „on
    an envelope"), iar centrul bucății sare cu sute de px. Cu max-min, 3 citiri
    trunchiate din 53 făceau un caption fix ca un ceas (mediană 542px, MAD 1px)
    să pară că traversează 217px din cadru — clusterul era aruncat întreg și
    subtitrarea rămânea arsă în video. Un obiect care chiar se mișcă prin scenă
    derivă pe MAJORITATEA cadrelor, deci trece în continuare de prag."""
    def spread(vals):
        if DRIFT_TRIM <= 0 or len(vals) < 5:
            return max(vals) - min(vals)
        s = sorted(vals)
        lo = s[int(round((len(s) - 1) * DRIFT_TRIM))]
        hi = s[int(round((len(s) - 1) * (1.0 - DRIFT_TRIM)))]
        return hi - lo
    dx, dy = _anchor_spread(members, spread)
    return (dx * dx + dy * dy) ** 0.5


def _anchor_spread(members, spread):
    """(dx, dy) — variația celei mai stabile ancore pe fiecare axă."""
    xs1 = [b[0] for _, b in members]; xs2 = [b[2] for _, b in members]
    ys1 = [b[1] for _, b in members]; ys2 = [b[3] for _, b in members]
    cxs = [(a + b) / 2 for a, b in zip(xs1, xs2)]
    cys = [(a + b) / 2 for a, b in zip(ys1, ys2)]
    dx = min(spread(xs1), spread(cxs), spread(xs2))
    dy = min(spread(ys1), spread(cys), spread(ys2))
    return dx, dy


def _vertical_drift(members):
    """Variația pe VERTICALĂ a clusterului (px), cozi tăiate ca la _anchor_drift.
    O citire OCR parțială taie din LĂȚIMEA liniei, niciodată din înălțime — deci
    pe verticală un caption ars stă pe loc oricât de ciuntit ar fi citit, pe când
    un tricou/obiect filmat apare la înălțimi diferite de la o scenă la alta."""
    def spread(vals):
        if DRIFT_TRIM <= 0 or len(vals) < 5:
            return max(vals) - min(vals)
        s = sorted(vals)
        return s[int(round((len(s) - 1) * (1.0 - DRIFT_TRIM)))] - s[int(round((len(s) - 1) * DRIFT_TRIM))]
    return _anchor_spread(members, spread)[1]


def _motion_drift(members, max_gap):
    """Cât se MIȘCĂ textul cât timp rămâne ACELAȘI text (px între keyframe-uri
    vecine). Se compară doar box-uri de mărime aproape egală (±SAME_TEXT_TOL) de
    pe keyframe-uri consecutive: un caption ars stă nemișcat cât e afișat, apoi
    SARE (altă frază, altă lățime) — săriturile nu contează; textul de pe un
    tricou/produs filmat se deplasează continuu păstrându-și mărimea.
    Se ia mediana de jos: câteva perechi strâmbe (citiri parțiale de mărime
    apropiată) nu pot face singure un caption să pară în mișcare."""
    by_fi = {}
    for fi, b in members:
        by_fi.setdefault(fi, []).append(b)
    fis = sorted(by_fi)
    moves = []
    for a, c in zip(fis, fis[1:]):
        if c - a > max_gap:
            continue
        for ba in by_fi[a]:
            wa, ha = ba[2] - ba[0], ba[3] - ba[1]
            best = None
            for bc in by_fi[c]:
                wc, hc = bc[2] - bc[0], bc[3] - bc[1]
                if abs(wa - wc) > SAME_TEXT_TOL * max(wa, wc) or abs(ha - hc) > SAME_TEXT_TOL * max(ha, hc):
                    continue
                d = math.hypot((ba[0] + ba[2] - bc[0] - bc[2]) / 2.0, (ba[1] + ba[3] - bc[1] - bc[3]) / 2.0)
                best = d if best is None else min(best, d)
            if best is not None:
                moves.append(best)
    if len(moves) < 2:
        return 0.0
    moves.sort()
    return moves[(len(moves) - 1) // 2]


def group_static_boxes(per_frame_boxes, min_ratio, n_frames_detected, frame_diag=None):
    """
    Grupează box-urile care apar (IoU>0.5) în ≥min_ratio din keyframes → statice.
    Returnează (static_boxes, per_frame_dynamic).

    Anti-distrugere (fix „video terci pe RedNote"):
      • clusterele cu text care SE MIȘCĂ (tricouri/obiecte filmate) se ARUNCĂ —
        nu-s overlay ars, iar inpainting-ul lor tocă subiectul video;
      • union-ul unui cluster se ștanțează pe TOT clipul (static) doar dacă box-ul
        chiar stă pe loc; altfel clusterul merge pe box-urile detectate la fiecare
        keyframe — union-ul creștea în lanț (IoU cu el însuși) până acoperea
        jumătate de frame.

    „În mișcare" = se deplasează între keyframe-uri VECINE cu același text
    (_motion_drift) SAU apare la înălțimi diferite pe parcursul clipului
    (_vertical_drift). NU se mai judecă după cât variază poziția ORIZONTALĂ pe tot
    clipul: clusterul benzii de caption adună zeci de fraze de lățimi diferite,
    iar pe liniile late EasyOCR citește des doar o bucată („but Brucedidnftdo"
    fără „any") — centrul bucății sare cu sute de px. Cu varianta veche un cluster
    de 36 de caption-uri reale ieșea „în mișcare" (133px > 121px, din care 119px
    orizontal) și era aruncat ÎNTREG: toate frazele late rămâneau arse.
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

    kf_sorted = sorted(per_frame_boxes)
    gaps = sorted(b - a for a, b in zip(kf_sorted, kf_sorted[1:]) if b > a)
    kf_step = gaps[len(gaps) // 2] if gaps else 1

    static, dynamic = [], {fi: [] for fi in per_frame_boxes}
    n_moving = 0
    for c in clusters:
        many = frame_diag and len(c["members"]) >= 3
        if many and (_motion_drift(c["members"], 2 * kf_step) > DRIFT_STEP_PCT * frame_diag
                     or _vertical_drift(c["members"]) > DRIFT_MAX_PCT * frame_diag):
            n_moving += 1
            continue
        fixed = not many or _anchor_drift(c["members"]) <= DRIFT_MAX_PCT * frame_diag
        if fixed and len(c["hits"]) >= max(2, min_ratio * n_frames_detected):
            static.append(c["box"])
        else:
            for fi, b in c["members"]:
                dynamic[fi].append(b)
    if n_moving:
        print(f"[DETECT] {n_moving} cluster(e) în mișcare ignorate (text pe obiecte, nu overlay)", flush=True)
    return static, dynamic


def _fill_kf_gaps(dynamic_by_kf, kf_sorted):
    """OCR-ul ratează uneori o linie pe UN SINGUR keyframe (încrederea cade sub
    prag exact atunci). Cadrul acela rămânea fără mască → textul apărea o clipă
    întreg, iar ProPainter îl propaga apoi în cadrele vecine (fantome de litere).
    Dacă aceeași linie orizontală e găsită pe keyframe-ul dinainte ȘI pe cel de
    după, în același loc, o punem și pe cel din mijloc."""
    add = {}
    for i in range(1, len(kf_sorted) - 1):
        a, k, c = kf_sorted[i - 1], kf_sorted[i], kf_sorted[i + 1]
        have = dynamic_by_kf.get(k, [])
        for ba in dynamic_by_kf.get(a, []):
            if not _is_caption_line(ba):
                continue
            for bc in dynamic_by_kf.get(c, []):
                if not _is_caption_line(bc) or _iou(ba, bc) < 0.5:
                    continue
                u = (min(ba[0], bc[0]), min(ba[1], bc[1]), max(ba[2], bc[2]), max(ba[3], bc[3]))
                # linia chiar lipsește? (dacă acolo e ALT text — caption cuvânt-cu-cuvânt —
                # nu e ratare, iar box-ul lui se desenează oricum)
                present = any(
                    _box_area((max(u[0], b[0]), max(u[1], b[1]), min(u[2], b[2]), min(u[3], b[3])))
                    >= 0.3 * min(_box_area(b), _box_area(u))
                    for b in have)
                if not present:
                    core = (min(ba.core[0], bc.core[0]), min(ba.core[1], bc.core[1]),
                            max(ba.core[2], bc.core[2]), max(ba.core[3], bc.core[3]))
                    add.setdefault(k, []).append(_textbox(u, core=core, horiz=True))
                break
    for k, bs in add.items():
        dynamic_by_kf.setdefault(k, []).extend(bs)
    return sum(len(bs) for bs in add.values())


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

    ocr_raw, flo_hits = {}, {}
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
            ocr_raw[idx] = detect_text_ocr(frame, w, h)
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

    if ocr_raw:
        ocr_hits = _text_boxes_from_hits(ocr_raw, w, h)
        ocr_static, ocr_dyn = group_static_boxes(ocr_hits, STATIC_RATIO, len(ocr_hits), frame_diag)
        # dacă userul NU vrea captions, păstrăm din OCR doar textul STATIC (watermark text)
        if "captions" in targets:
            for fi, bs in ocr_dyn.items():
                dynamic_by_kf.setdefault(fi, []).extend(bs)
            n_fill = _fill_kf_gaps(dynamic_by_kf, sorted(kf_indices))
            if n_fill:
                print(f"[DETECT] {n_fill} linii ratate pe câte un keyframe, completate din vecini", flush=True)
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
# ── Plafonul de acoperire: caption-urile nu se ating ─────────────────────────
# Versiunile vechi, peste plafon, fie scoteau box-urile cele mai mari, fie le
# strângeau pe TOATE spre centru cu aceeași fracție. În ambele cazuri plătea
# caption-ul lat: e cel mai mare box, deci fie dispărea întreg, fie își pierdea
# capetele („to se……ack.", „momentu……hen swing,"). Iar plafonul îl umpleau
# altele — watermark-ul diagonal care se plimbă, box-urile statice de la Florence.
# Acum se sacrifică în ordinea răului făcut:
#   1) paddingul a tot ce NU e linie de caption (până la conturul textului)
#   2) box-urile care nu-s linii de caption, cele mai mari întâi
#   3) jumătate din paddingul caption-urilor
# Glifele unui caption nu se taie niciodată: un caption lăsat ars pe ecran e
# exact defectul pe care îl reparăm, deci dacă doar ele depășesc plafonul,
# rămân întregi (se loghează).

def _toward_core(b, t):
    """Mută box-ul spre conturul textului (fără padding) cu fracția t ∈ [0, 1]."""
    core = getattr(b, "core", None)
    if core is None or t <= 0:
        return b
    if b.poly is not None:
        poly = b.poly + (np.asarray(core, dtype=np.float32) - b.poly) * t
        rect = (max(b[0], math.floor(poly[:, 0].min())), max(b[1], math.floor(poly[:, 1].min())),
                min(b[2], math.ceil(poly[:, 0].max())), min(b[3], math.ceil(poly[:, 1].max())))
        return _textbox(rect, poly=poly, core=core, horiz=b.horiz)
    rect = tuple(int(round(b[i] + (core[i] - b[i]) * t)) for i in range(4))
    return _textbox(rect, core=core, horiz=b.horiz)


def _fit_coverage(boxes, base_static, cap):
    """Aduce masca sub plafon fără să taie din caption-uri.
    Întoarce (casete, mască, ce s-a făcut: "ok"/"padding"/"dropped"/"over")."""
    def cover(bs):
        m = base_static.copy()
        for b in bs:
            _draw_box(m, b)
        return m.mean() / 255.0, m

    c, m = cover(boxes)
    if c <= cap:
        return boxes, m, "ok"
    lines = [b for b in boxes if _is_caption_line(b)]
    rest = [_toward_core(b, 1.0) for b in boxes if not _is_caption_line(b)]
    c, m = cover(lines + rest)
    if c <= cap:
        return lines + rest, m, "padding"
    rest.sort(key=_box_area, reverse=True)
    dropped = False
    while rest:
        rest.pop(0)
        dropped = True
        c, m = cover(lines + rest)
        if c <= cap:
            return lines + rest, m, "dropped"
    lines = [_toward_core(b, 0.5) for b in lines]
    c, m = cover(lines)
    if c <= cap:
        return lines, m, "dropped" if dropped else "padding"
    return lines, m, "over"



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
        # doar cele dinamice: staticele sunt deja în base_static (și în fb, mai jos)
        boxes = []
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
    for b in static_boxes:
        _draw_box(base_static, b)

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

    cap_stats = {"padding": 0, "dropped": 0, "over": 0}
    # bbox-ul măștii PE FIECARE CADRU. Uniunea pe tot clipul aproape mereu iese
    # cât tot cadrul (text sus într-o scenă, jos în alta) și decupajul nu se mai
    # activa. Per bucată temporală banda e mult mai strânsă — vezi run_inpainting.
    frame_boxes = []
    # bounding box-ul UNIUNII tuturor măștilor, pe tot clipul: ProPainter n-are
    # de ce să macine cadrul întreg când textul stă într-o bandă. Se calculează
    # aici, după plafonare, ca să reflecte exact ce s-a desenat în mask.mp4.
    ux1, uy1, ux2, uy2 = w, h, 0, 0
    try:
        for fidx in range(n_frames):
            boxes = boxes_for_frame(fidx)
            mask = base_static.copy()
            for b in boxes:
                _draw_box(mask, b)
            # plasă de siguranță: dacă masca ar acoperi >MASK_MAX_COVERAGE din frame,
            # inpainting-ul nu mai are din ce reconstrui. Se sacrifică întâi ce NU
            # e caption — vezi _fit_coverage.
            if mask.mean() / 255.0 > MASK_MAX_COVERAGE:
                boxes, mask, how = _fit_coverage(boxes, base_static, MASK_MAX_COVERAGE)
                if how in cap_stats:
                    cap_stats[how] += 1
            fb = None
            if mask.any():
                total_active += 1
                bx1, by1, bx2, by2 = w, h, 0, 0
                for (x1, y1, x2, y2) in (list(boxes) + list(static_boxes)):
                    ux1, uy1 = min(ux1, x1), min(uy1, y1)
                    ux2, uy2 = max(ux2, x2), max(uy2, y2)
                    bx1, by1 = min(bx1, x1), min(by1, y1)
                    bx2, by2 = max(bx2, x2), max(by2, y2)
                if bx2 > bx1 and by2 > by1:
                    fb = (bx1, by1, bx2, by2)
            frame_boxes.append(fb)
            proc.stdin.write(mask.tobytes())
    finally:
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Encodarea măștii a eșuat: {err[-300:]}")

    if any(cap_stats.values()):
        print(f"[MASK] peste plafonul de {MASK_MAX_COVERAGE:.0%}: {cap_stats['padding']} cadre doar fără padding, "
              f"{cap_stats['dropped']} fără box-uri non-caption, {cap_stats['over']} lăsate peste plafon "
              f"(doar caption-uri — nu se taie)", flush=True)
    print(f"[MASK] {n_frames} frames, {total_active} cu mască activă → {mask_path}", flush=True)
    roi = (ux1, uy1, ux2, uy2) if ux2 > ux1 and uy2 > uy1 else None
    return total_active, roi, frame_boxes


def compute_roi(roi, w, h, label=""):
    """Banda de inpainting: bbox-ul măștii + contur de context, aliniat la par.

    ProPainter are nevoie de pixeli SĂNĂTOȘI în jurul găurii ca să aibă din ce
    reconstrui, deci nu tăiem fix pe bbox. Dacă banda oricum acoperă aproape tot
    cadrul, nu are rost decupajul — returnăm None și rulăm ca înainte."""
    if not roi:
        return None
    x1, y1, x2, y2 = roi
    pad_x = max(ROI_MIN_PAD, int((x2 - x1) * ROI_PAD_PCT))
    pad_y = max(ROI_MIN_PAD, int((y2 - y1) * ROI_PAD_PCT))
    x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x); y2 = min(h, y2 + pad_y)
    # ffmpeg crop + encoderele vor dimensiuni pare
    x1 -= x1 % 2; y1 -= y1 % 2
    x2 -= (x2 - x1) % 2; y2 -= (y2 - y1) % 2
    rw, rh = x2 - x1, y2 - y1
    if rw < 64 or rh < 64:
        return None
    if (rw * rh) / float(w * h) > ROI_MAX_AREA_PCT:
        print(f"[ROI]{label} banda acoperă {(rw*rh)/float(w*h):.0%} din cadru — nu decupez", flush=True)
        return None
    print(f"[ROI]{label} inpaint doar pe {rw}x{rh} @ ({x1},{y1}) = {(rw*rh)/float(w*h):.0%} din cadru "
          f"(în loc de {w}x{h})", flush=True)
    return (x1, y1, rw, rh)


def crop_to_roi(src, dst, roi, lossless=False):
    """Decupează un video la ROI. Masca trebuie lossless (margini binare)."""
    x, y, rw, rh = roi
    venc = ["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0"] if lossless else _venc(crf="16", preset="veryfast")
    subprocess.run([
        "ffmpeg", "-y", "-nostats", "-loglevel", "error", "-i", src,
        "-vf", f"crop={rw}:{rh}:{x}:{y}", *venc, "-an", dst,
    ], check=True)
    return dst


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


def _extract_segment(src, dst, start_frame, end_frame, lossless=False, roi=None):
    """Taie [start_frame, end_frame) din src, opțional direct pe banda de text.
    `trim` pe numere de cadre e exact pe CFR (inputul e normalizat la intrare).
    Decupajul intră în ACELAȘI lanț de filtre: altfel am fi trecut de două ori
    prin toate cadrele la rezoluție plină, doar ca să tăiem apoi marginile."""
    vf = f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS"
    if roi:
        x, y, rw, rh = roi
        vf += f",crop={rw}:{rh}:{x}:{y}"
    subprocess.run([
        "ffmpeg", "-y", "-nostats", "-loglevel", "error",
        "-i", src,
        "-vf", vf,
        "-an",
        *(["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0"] if lossless
          else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "14"]),
        "-pix_fmt", "yuv420p", dst,
    ], check=True)


def _concat_segments(seg_paths, drops, out_path, fps, places=None, w=None, h=None):
    """Lipește bucățile aruncând primele `drops[i]` cadre din fiecare (cadrele de
    suprapunere, deja acoperite de bucata anterioară). Un singur re-encode.

    `places[i]` = banda pe care a fost inpaint-ată bucata i (sau None dacă a mers
    pe cadru întreg). Aducerea la dimensiune comună se face AICI, în același lanț
    de filtre: `concat` refuză intrări de mărimi diferite, iar o trecere separată
    per bucată însemna încă o codare full-res de fiecare dată — pe un clip de 33s
    se simțea în timpul total."""
    same_size = not places or all(p is None for p in places)
    if len(seg_paths) == 1 and drops[0] == 0 and same_size:
        shutil.move(seg_paths[0], out_path)
        return
    cmd = ["ffmpeg", "-y", "-nostats", "-loglevel", "error"]
    for p in seg_paths:
        cmd += ["-i", p]
    parts, labels = [], []
    for i, d in enumerate(drops):
        chain = f"[{i}:v]trim=start_frame={d},setpts=PTS-STARTPTS"
        if not same_size:
            roi = places[i] if places else None
            if roi:
                rx, ry, rw, rh = roi
                chain += f",scale={rw}:{rh}:flags=lanczos,pad={w}:{h}:{rx}:{ry}"
            else:
                chain += f",scale={w}:{h}:flags=lanczos"
            chain += ",setsar=1"
        parts.append(chain + f"[v{i}]")
        labels.append(f"[v{i}]")
    parts.append("".join(labels) + f"concat=n={len(seg_paths)}:v=1:a=0[out]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[out]",
            "-r", f"{fps}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "14",
            "-pix_fmt", "yuv420p", out_path]
    subprocess.run(cmd, check=True)


def _band_of(frame_boxes, s, e, w, h):
    """Uniunea bbox-urilor măștii pe cadrele [s, e)."""
    if not frame_boxes:
        return None
    x1, y1, x2, y2 = w, h, 0, 0
    for fb in frame_boxes[s:e]:
        if not fb:
            continue
        x1, y1 = min(x1, fb[0]), min(y1, fb[1])
        x2, y2 = max(x2, fb[2]), max(y2, fb[3])
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def run_inpainting(video_path, mask_path, workdir, duration_s, max_img_size, quality,
                   w, h, n_frames, fps, frame_boxes=None):
    """quality="fast" -> doar ProPainter (~2 min pt 20s video, foarte bun pe captions).
    quality="max"  -> + rafinare DiffuEraser (calitate maximă, dar de 3-5x mai lent).

    Clipul se taie în bucăți temporale cu CHUNK_OVERLAP cadre de suprapunere.
    FIECARE bucată își calculează BANDA ei de text și se inpaint-ează doar pe ea:
    uniunea pe tot clipul ieșea 82-100% din cadru (text sus într-o scenă, jos în
    alta) și decupajul nu se activa aproape niciodată. Per bucată banda e strânsă,
    deci la același PROC_MAX_SIDE textul intră la o rezoluție efectivă mult mai
    mare — de acolo veneau dreptunghiurile fantomă rămase peste text.
    Bucata se așază înapoi în cadrul întreg înainte de lipire, ca toate să aibă
    aceeași dimensiune. finalize() compune peste originalul full-res, doar sub mască."""
    priori_path = os.path.join(workdir, "priori.mp4")
    result_path = os.path.join(workdir, "diffueraser_out.mp4")

    def _attempt(max_side):
        # planul de bucăți se face pe cadrul ÎNTREG (conservator): banda fiecărei
        # bucăți e mai mică sau egală, deci nu poate ieși din VRAM față de plan
        fw, fh = _proc_size(w, h, max_side)
        chunk = _chunk_frames(fw, fh)
        seg_dir = os.path.join(workdir, "segments")
        shutil.rmtree(seg_dir, ignore_errors=True)
        os.makedirs(seg_dir, exist_ok=True)

        def _priori(vid, msk, dst, seg_frames, pw, ph, dil):
            # resize_ratio=1.0 + width/height explicite -> dezactivăm downscale-ul
            # intern nedeterminist al DiffuEraser (default 0.6, x0.5 peste 960px)
            PROPAINTER.forward(
                vid, msk, dst,
                resize_ratio=1.0, width=pw, height=ph,
                video_length=int(seg_frames / float(fps)) + 1,
                ref_stride=10, neighbor_length=10, subvideo_length=50,
                mask_dilation=dil,
            )

        # Bucata i PRODUCE cadrele [out_start, e) și le mai PROCESEAZĂ pe cele
        # `lead` dinaintea lor doar ca context temporal (se aruncă la lipire).
        # Intervalele produse se cap-coadă exact, fără suprapunere în output.
        overlap = min(CHUNK_OVERLAP, max(0, chunk // 4))
        plan = []
        out_start = 0
        while out_start < n_frames:
            lead = min(overlap, out_start)
            st = out_start - lead
            e = min(n_frames, st + chunk)
            # coada scurtă se lipește de bucata curentă în loc să devină o bucată
            # separată: altfel ultima rulare ProPainter procesa `overlap`+2 cadre
            # ca să producă 1-2 utile — o trecere întreagă de GPU degeaba
            if 0 < n_frames - e <= overlap:
                e = n_frames
            plan.append((st, e, lead))
            out_start = e
        print(f"[INPAINT] {n_frames} cadre -> {len(plan)} bucăți x max {chunk} "
              f"(overlap {overlap}), bandă proprie per bucată", flush=True)

        seg_outs, drops, places = [], [], []
        for i, (st, e, lead) in enumerate(plan):
            tag = f" bucata {i+1}/{len(plan)}"
            roi = compute_roi(_band_of(frame_boxes, st, e, w, h), w, h, label=tag)
            bw, bh = (roi[2], roi[3]) if roi else (w, h)
            pw, ph = _proc_size(bw, bh, max_side)
            # dilatarea se dă lui ProPainter în pixeli DE PROCESARE, dar o exprimăm
            # în pixeli la rezoluția originală -> efectul rămâne același indiferent
            # de scalare (înainte, 8px la 182x324 însemnau 32px pe video-ul real)
            dil = max(1, int(round(MASK_DILATE_PX * (pw / float(bw)))))
            seg_v = os.path.join(seg_dir, f"v{i:03d}.mp4")
            seg_m = os.path.join(seg_dir, f"m{i:03d}.mp4")
            seg_o = os.path.join(seg_dir, f"o{i:03d}.mp4")
            _extract_segment(video_path, seg_v, st, e, roi=roi)
            _extract_segment(mask_path, seg_m, st, e, lossless=True, roi=roi)
            print(f"[INPAINT] {tag}: cadre {st}-{e} @ {pw}x{ph} (dilate={dil}px)", flush=True)
            _priori(seg_v, seg_m, seg_o, e - st, pw, ph, dil)
            seg_outs.append(seg_o)
            drops.append(lead)
            places.append(roi)
            for tmp in (seg_v, seg_m):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        _concat_segments(seg_outs, drops, priori_path, fps, places=places, w=w, h=h)
        shutil.rmtree(seg_dir, ignore_errors=True)

    oom = False
    try:
        _attempt(PROC_MAX_SIDE)
    except torch.cuda.OutOfMemoryError:
        # NU reîncercăm aici: cât timp suntem în except, traceback-ul activ ține
        # referințe la tensorii din ProPainter -> empty_cache() nu poate elibera
        # VRAM-ul și retry-ul murea tot cu OOM ("22.5 GiB in use" la reîncercare)
        oom = True
    if oom:
        gc.collect()
        torch.cuda.empty_cache()
        retry_side = max(MIN_PROC_SIDE, int(PROC_MAX_SIDE * 0.7))
        print(f"[INPAINT] CUDA OOM -> reîncerc cu latura lungă {retry_side}", flush=True)
        _attempt(retry_side)

    if quality != "max":
        print("[INPAINT] quality=fast -> sar peste DiffuEraser", flush=True)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return priori_path

    # DiffuEraser rulează la max_img_size, nu la _proc_size -> dilatarea lui se
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


def finalize(result_path, original_path, mask_path, out_path, w, h, roi=None):
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
    # Cu ROI, rezultatul acoperă doar banda de text: îl scalăm la dimensiunea ei
    # și îl așezăm la offsetul din cadru. Restul cadrului rămâne negru, dar nu se
    # vede niciodată — masca e 0 acolo prin construcție (ROI conține tot ce e mascat).
    if roi:
        rx, ry, rw, rh = roi
        place = f"scale={rw}:{rh}:flags=lanczos,pad={w}:{h}:{rx}:{ry}"
    else:
        place = f"scale={w}:{h}:flags=lanczos"
    # gblur sigma=σ urmat de prag t dilată cu ≈ σ·Φ⁻¹(1−t/255) pixeli.
    # sigma=2 + prag 40 → ~2px. Feather-ul final rămâne subțire.
    subprocess.run([
        "ffmpeg", "-y", "-nostats", "-loglevel", "error",
        "-i", original_path,
        "-i", result_path,
        "-i", mask_path,
        "-filter_complex",
        f"[1:v]{place},setsar=1,format=yuva420p[res];"
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
    # Plasa de siguranță pentru cazul în care CUDA moare DUPĂ pornire (driver
    # reset, GPU scos de sub container). Fără ea, jobul ar continua pe CPU și ar
    # ține workerul ocupat zeci de minute; așa pică în 2s și clientul îl poate
    # relua imediat pe altă mașină.
    if DEVICE != "cuda":
        return {"error": "Worker fără GPU disponibil — reia jobul (se va aloca altă mașină)."}
    t_job = time.time()
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
        _total_active, _roi_all, frame_boxes = build_mask_video(mask_path, w, h, fps, n_frames,
                                                                 static_boxes, dynamic_by_kf, kf_indices, workdir)

        # Inpainting DOAR pe banda de text, calculata PER BUCATA temporala.
        # Uniunea pe tot clipul iesea 82-100% din cadru (text sus intr-o scena, jos
        # in alta) si decupajul nu se activa aproape niciodata. Per bucata banda e
        # stransa, deci la acelasi PROC_MAX_SIDE textul intra la rezolutie efectiva
        # mult mai mare. Compunerea finala se face oricum peste originalul full-res,
        # doar sub masca, deci restul cadrului ramane neatins.
        result_path = run_inpainting(video_path, mask_path, workdir, duration,
                                     max_img_size, quality, w, h, n_frames, fps,
                                     frame_boxes=frame_boxes)

        out_path = os.path.join(workdir, "final.mp4")
        # rezultatul vine deja la cadru intreg (fiecare bucata s-a asezat inapoi
        # la offsetul benzii ei), deci aici nu mai e nimic de repozitionat
        finalize(result_path, video_path, mask_path, out_path, w, h, roi=None)

        out = deliver(out_path, job_input)
        out["detections"] = {
            "static_boxes": len(static_boxes),
            "dynamic_hits": n_dynamic,
            "keyframes": len(kf_indices),
        }
        # timpul total pe job = ce se facturează; fără el nu se vede din loguri
        # dacă un endpoint a devenit brusc de 5× mai scump
        print(f"[JOB] gata în {time.time() - t_job:.1f}s ({duration:.1f}s video)", flush=True)
        return out

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
