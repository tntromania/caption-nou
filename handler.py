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
import json
import base64
import shutil
import tempfile
import subprocess
import traceback

import gc

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
# complete (markerul .easyocr.done e ultimul scris de download_weights.py),
# ignorăm WEIGHTS_DIR extern (ex. Network Volume rămas setat pe endpoint)
# → zero download la cold start, indiferent de mașină/volum.
_BAKED_WEIGHTS = "/app/weights"
if WEIGHTS_DIR != _BAKED_WEIGHTS and os.path.exists(os.path.join(_BAKED_WEIGHTS, ".easyocr.done")):
    print(f"[INIT] Greutăți baked în imagine → folosesc {_BAKED_WEIGHTS} (ignor WEIGHTS_DIR={WEIGHTS_DIR})", flush=True)
    WEIGHTS_DIR = _BAKED_WEIGHTS
MAX_SECONDS      = float(os.environ.get("MAX_SECONDS", "90"))
MAX_FPS          = int(os.environ.get("MAX_FPS", "30"))              # 60fps → 30fps: jumătate din cadre = jumătate din VRAM/timp
PROC_MAX_SIDE    = int(os.environ.get("PROC_MAX_SIDE", "640"))       # latura lungă la care rulează inpainting-ul
PROC_FRAME_BUDGET = int(os.environ.get("PROC_FRAME_BUDGET", "1200")) # peste atât, rezoluția scade proporțional (VRAM ~ cadre × pixeli)
DETECT_INTERVAL  = float(os.environ.get("DETECT_INTERVAL", "0.5"))   # secunde între keyframes OCR
FLORENCE_INTERVAL = float(os.environ.get("FLORENCE_INTERVAL", "2.0")) # secunde între keyframes Florence
OCR_CONF         = float(os.environ.get("OCR_CONF", "0.25"))
STATIC_RATIO     = float(os.environ.get("STATIC_RATIO", "0.60"))     # % din keyframes ca un box să fie "static"
MAX_BOX_AREA_PCT = float(os.environ.get("MAX_BOX_AREA_PCT", "0.25")) # ignoră box-uri > 25% din frame
BOX_PAD          = int(os.environ.get("BOX_PAD", "6"))

sys.path.insert(0, DIFFUERASER_DIR)

# Cache-urile HuggingFace + EasyOCR merg lângă greutăți — esențial pe Network
# Volume, altfel Florence-2/EasyOCR s-ar re-descărca la fiecare cold start.
os.environ.setdefault("HF_HOME", os.path.join(WEIGHTS_DIR, "hf-cache"))
os.environ.setdefault("EASYOCR_MODULE_PATH", os.path.join(WEIGHTS_DIR, "easyocr"))

# Mod Network Volume: dacă greutățile lipsesc, le descărcăm O SINGURĂ DATĂ aici.
# (.easyocr.done e ultimul marker scris de download_weights.py = descărcare completă)
if not os.path.exists(os.path.join(WEIGHTS_DIR, ".easyocr.done")):
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

# ═════════════════════════════════════════════════════════════════════════════
# INIT — toate modelele se încarcă O SINGURĂ DATĂ la pornirea workerului
# ═════════════════════════════════════════════════════════════════════════════
print("[INIT] Încărcare DiffuEraser + ProPainter...", flush=True)
from diffueraser.diffueraser import DiffuEraser
from propainter.inference import Propainter

DIFFU = DiffuEraser(
    DEVICE,
    os.path.join(WEIGHTS_DIR, "stable-diffusion-v1-5"),
    os.path.join(WEIGHTS_DIR, "sd-vae-ft-mse"),
    os.path.join(WEIGHTS_DIR, "diffuEraser"),
    ckpt="2-Step",
)
PROPAINTER = Propainter(os.path.join(WEIGHTS_DIR, "propainter"), device=DEVICE)
print("[INIT] DiffuEraser + ProPainter OK", flush=True)

print("[INIT] Încărcare EasyOCR...", flush=True)
import easyocr
OCR = easyocr.Reader(["en", "ro"], gpu=(DEVICE == "cuda"), verbose=False)
print("[INIT] EasyOCR OK", flush=True)

print("[INIT] Încărcare Florence-2...", flush=True)
from unittest.mock import patch
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports

def _fixed_get_imports(filename):
    # Florence-2 declară flash_attn ca import obligatoriu; nu e necesar cu SDPA.
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

FLORENCE_ID = os.environ.get("FLORENCE_MODEL", "microsoft/Florence-2-large")
with patch("transformers.dynamic_module_utils.get_imports", _fixed_get_imports):
    FLORENCE = AutoModelForCausalLM.from_pretrained(
        FLORENCE_ID, trust_remote_code=True, torch_dtype=DTYPE,
        attn_implementation="sdpa",
    ).to(DEVICE).eval()
    FLORENCE_PROC = AutoProcessor.from_pretrained(FLORENCE_ID, trust_remote_code=True)
print("[INIT] Florence-2 OK — worker gata", flush=True)


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


def detect_text_ocr(frame_bgr, w, h):
    """EasyOCR pe un frame → listă de box-uri (x1,y1,x2,y2). Downscale pt viteză."""
    scale = 1.0
    img = frame_bgr
    if w > 1280:
        scale = 1280.0 / w
        img = cv2.resize(frame_bgr, (1280, int(h * scale)))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = []
    for (bbox, text, conf) in OCR.readtext(rgb, detail=1):
        if conf < OCR_CONF or not str(text).strip():
            continue
        pts = np.array(bbox, dtype=np.float32) / scale
        x, y, bw, bh = cv2.boundingRect(pts.astype(np.int32))
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
    task = "<CAPTION_TO_PHRASE_GROUNDING>"
    text = task + " ".join(p.rstrip(".") + "." for p in prompts)
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = FLORENCE_PROC(text=text, images=pil, return_tensors="pt").to(DEVICE)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(DTYPE)
    ids = FLORENCE.generate(
        input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
        max_new_tokens=256, num_beams=3, do_sample=False,
    )
    out = FLORENCE_PROC.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = FLORENCE_PROC.post_process_generation(out, task=task, image_size=(w, h))
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


def group_static_boxes(per_frame_boxes, min_ratio, n_frames_detected):
    """
    Grupează box-urile care apar (IoU>0.5) în ≥min_ratio din keyframes → statice.
    Returnează (static_boxes, per_frame_dynamic).
    """
    clusters = []  # fiecare: {"box": union, "hits": set(frame_idx)}
    for fi, boxes in per_frame_boxes.items():
        for b in boxes:
            placed = False
            for c in clusters:
                if _iou(c["box"], b) > 0.5:
                    x1 = min(c["box"][0], b[0]); y1 = min(c["box"][1], b[1])
                    x2 = max(c["box"][2], b[2]); y2 = max(c["box"][3], b[3])
                    c["box"] = (x1, y1, x2, y2)
                    c["hits"].add(fi)
                    placed = True
                    break
            if not placed:
                clusters.append({"box": b, "hits": {fi}})

    static, dynamic = [], {fi: [] for fi in per_frame_boxes}
    for c in clusters:
        if len(c["hits"]) >= max(2, min_ratio * n_frames_detected):
            static.append(c["box"])
        else:
            for fi in c["hits"]:
                dynamic[fi].append(c["box"])
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
    idx = 0
    while idx < n_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        kf_indices.append(idx)
        if want_text:
            ocr_hits[idx] = detect_text_ocr(frame, w, h)
        if want_logos and florence_prompts and idx % step_florence < step_ocr:
            flo_hits[idx] = detect_florence(frame, w, h, florence_prompts)
        idx += step_ocr
    cap.release()

    n_kf = max(1, len(kf_indices))
    static_boxes, dynamic_by_kf = [], {fi: [] for fi in kf_indices}

    if ocr_hits:
        ocr_static, ocr_dyn = group_static_boxes(ocr_hits, STATIC_RATIO, len(ocr_hits))
        # dacă userul NU vrea captions, păstrăm din OCR doar textul STATIC (watermark text)
        if "captions" in targets:
            for fi, bs in ocr_dyn.items():
                dynamic_by_kf.setdefault(fi, []).extend(bs)
        static_boxes += ocr_static

    if flo_hits:
        # Florence: logo/watermark = static prin definiție → cerem persistență în
        # ≥50% din frame-urile Florence ca să eliminăm halucinațiile pe obiecte
        flo_static, _flo_dyn = group_static_boxes(flo_hits, 0.5, len(flo_hits))
        static_boxes += flo_static

    n_static = len(static_boxes)
    n_dynamic = sum(len(v) for v in dynamic_by_kf.values())
    print(f"[DETECT] keyframes={n_kf} static={n_static} dynamic_hits={n_dynamic}", flush=True)
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
    png_dir = os.path.join(workdir, "mask_frames")
    os.makedirs(png_dir, exist_ok=True)

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

    for fidx in range(n_frames):
        mask = base_static.copy()
        for (x1, y1, x2, y2) in boxes_for_frame(fidx):
            mask[y1:y2, x1:x2] = 255
        if mask.any():
            total_active += 1
        cv2.imwrite(os.path.join(png_dir, f"{fidx:06d}.png"), mask,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])

    subprocess.run([
        "ffmpeg", "-y", "-nostats", "-loglevel", "error",
        "-framerate", f"{fps}",
        "-i", os.path.join(png_dir, "%06d.png"),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "12",
        "-pix_fmt", "yuv420p",
        mask_path,
    ], check=True)
    shutil.rmtree(png_dir, ignore_errors=True)
    print(f"[MASK] {n_frames} frames, {total_active} cu mască activă → {mask_path}", flush=True)
    return total_active


# ═════════════════════════════════════════════════════════════════════════════
# INPAINTING + FINISARE
# ═════════════════════════════════════════════════════════════════════════════
def _proc_size(w, h, n_frames):
    """Rezoluția la care rulează ProPainter. VRAM-ul lui crește liniar cu
    cadre × pixeli (ține TOT videoul + flow-urile pe GPU) → țintim latura lungă
    PROC_MAX_SIDE, iar peste PROC_FRAME_BUDGET cadre reducem suplimentar cu
    sqrt(budget/cadre) ca produsul cadre×pixeli să rămână constant."""
    ratio = min(1.0, PROC_MAX_SIDE / float(max(w, h)))
    if n_frames > PROC_FRAME_BUDGET:
        ratio *= (PROC_FRAME_BUDGET / float(n_frames)) ** 0.5
    pw = max(64, int(w * ratio)) // 2 * 2
    ph = max(64, int(h * ratio)) // 2 * 2
    return pw, ph


def run_inpainting(video_path, mask_path, workdir, duration_s, max_img_size, quality, w, h, n_frames):
    """quality="fast" → doar ProPainter (~2 min pt 20s video, foarte bun pe captions).
    quality="max"  → + rafinare DiffuEraser (calitate maximă, dar de 3-5x mai lent).
    Inpainting-ul rulează la rezoluție redusă (_proc_size); finalize() pune
    rezultatul înapoi peste originalul full-res doar în zonele mascate."""
    priori_path = os.path.join(workdir, "priori.mp4")
    result_path = os.path.join(workdir, "diffueraser_out.mp4")
    video_length = int(duration_s) + 1

    pw, ph = _proc_size(w, h, n_frames)
    print(f"[INPAINT] ProPainter priori @ {pw}x{ph} ({n_frames} cadre)...", flush=True)
    try:
        # resize_ratio=1.0 + width/height explicite → dezactivăm downscale-ul
        # intern nedeterminist al DiffuEraser (default 0.6, ×0.5 peste 960px)
        PROPAINTER.forward(
            video_path, mask_path, priori_path,
            resize_ratio=1.0, width=pw, height=ph,
            video_length=video_length,
            ref_stride=10, neighbor_length=10, subvideo_length=50,
            mask_dilation=8,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); gc.collect()
        pw, ph = max(64, int(pw * 0.6)) // 2 * 2, max(64, int(ph * 0.6)) // 2 * 2
        print(f"[INPAINT] CUDA OOM → reîncerc la {pw}x{ph}", flush=True)
        PROPAINTER.forward(
            video_path, mask_path, priori_path,
            resize_ratio=1.0, width=pw, height=ph,
            video_length=video_length,
            ref_stride=10, neighbor_length=10, subvideo_length=50,
            mask_dilation=8,
        )
    if quality != "max":
        print("[INPAINT] quality=fast → sar peste DiffuEraser", flush=True)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return priori_path

    print("[INPAINT] DiffuEraser refine...", flush=True)
    DIFFU.forward(
        video_path, mask_path, priori_path, result_path,
        max_img_size=max_img_size,
        video_length=video_length,
        mask_dilation_iter=8,
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
    Masca e apoi DILATATĂ ~9px (gblur mare + prag jos) înainte de feather:
    compozitarea folosea masca brută, strânsă pe box-urile OCR, deși inpainting-ul
    curăța mai lat (mask_dilation=8) → marginile literelor (prima/ultima din cuvânt)
    rămâneau vizibile din originalul full-res."""
    subprocess.run([
        "ffmpeg", "-y", "-nostats", "-loglevel", "error",
        "-i", original_path,
        "-i", result_path,
        "-i", mask_path,
        "-filter_complex",
        f"[1:v]scale={w}:{h}:flags=lanczos,setsar=1,format=yuva420p[res];"
        f"[2:v]scale={w}:{h},format=gray,lut=c0='if(gt(val,40),255,0)',"
        f"gblur=sigma=6,lut=c0='if(gt(val,16),255,0)',gblur=sigma=2[m];"
        f"[res][m]alphamerge[ov];"
        f"[0:v][ov]overlay=shortest=1,format=yuv420p[out]",
        "-map", "[out]", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
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


def normalize_fps(video_path, workdir):
    """DiffuEraser cere fps IDENTIC între video, mască și priori (read_priori
    compară strict și aruncă "The frame rate of all input videos needs to be
    consistent."). FPS-urile fracționare din filmări de telefon (ex. 30.05,
    29.97) se cuantizează diferit prin lanțul cv2/ffmpeg/imageio → re-eșantionăm
    inputul la fps ÎNTREG constant (CFR), o singură dată, la intrare.
    Plafonăm la MAX_FPS (default 30): la 60fps un clip de 27s = 1600+ cadre →
    ProPainter le ține pe toate în VRAM și dă CUDA OOM pe 24GB."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    target = max(10, min(MAX_FPS, int(round(fps)) or 30))
    if abs(fps - target) < 0.01:
        return video_path
    norm_path = os.path.join(workdir, "input_cfr.mp4")
    print(f"[NORM] {fps:.3f}fps → {target}fps CFR", flush=True)
    subprocess.run([
        "ffmpeg", "-y", "-nostats", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"fps={target}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        norm_path,
    ], check=True)
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
        video_path = normalize_fps(video_path, workdir)
        w, h, fps, n_frames, duration = probe(video_path)
        print(f"[JOB] {w}x{h} @ {fps:.2f}fps, {n_frames} frames, {duration:.1f}s, targets={targets}, quality={quality}", flush=True)

        if duration > MAX_SECONDS:
            return {"error": f"Video prea lung ({duration:.0f}s). Maxim: {MAX_SECONDS:.0f}s."}

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
                                     max_img_size, quality, w, h, n_frames)

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
