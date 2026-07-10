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
PROC_MAX_SIDE    = int(os.environ.get("PROC_MAX_SIDE", "640"))       # latura lungă la care rulează inpainting-ul
PROC_FRAME_BUDGET = int(os.environ.get("PROC_FRAME_BUDGET", "600"))  # peste atât, rezoluția scade proporțional (VRAM ~ cadre × pixeli);
                                                                     # 1200 dădea OOM pe 24GB la 1800+ cadre (~274M px·cadre) — 600 ține
                                                                     # produsul în zona dovedită sigură (~138M, cât joburile care au mers)
DETECT_INTERVAL  = float(os.environ.get("DETECT_INTERVAL", "0.5"))   # secunde între keyframes OCR
FLORENCE_INTERVAL = float(os.environ.get("FLORENCE_INTERVAL", "2.0")) # secunde între keyframes Florence
OCR_CONF         = float(os.environ.get("OCR_CONF", "0.25"))
STATIC_RATIO     = float(os.environ.get("STATIC_RATIO", "0.60"))     # % din keyframes ca un box să fie "static"
MAX_BOX_AREA_PCT = float(os.environ.get("MAX_BOX_AREA_PCT", "0.25")) # ignoră box-uri > 25% din frame
BOX_PAD          = int(os.environ.get("BOX_PAD", "6"))
DRIFT_MAX_PCT    = float(os.environ.get("DRIFT_MAX_PCT", "0.04"))    # drift max al unui cluster (fracție din diagonală) ca să fie overlay, nu text pe obiect
MASK_MAX_COVERAGE = float(os.environ.get("MASK_MAX_COVERAGE", "0.40")) # plafonul măștii pe un frame — peste, scoatem box-urile cele mai mari

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
# DOUĂ cititoare: latin (en+ro, ca până acum) + chinez (ch_sim+en).
# Sursele sunt Douyin/RedNote/TikTok: caption-urile chinezești citite de modelul
# latin ieșeau gunoi cu conf<0.25 → nu se ștergeau NICIODATĂ. Modelul ch_sim le
# citește, dar dă conf ~0 chiar când citește corect → pragul de conf nu se aplică
# ideogramelor (vezi detect_text_ocr); latinul păstrează pragul normal.
OCR     = easyocr.Reader(["en", "ro"], gpu=(DEVICE == "cuda"), verbose=False)
OCR_ZH  = easyocr.Reader(["ch_sim", "en"], gpu=(DEVICE == "cuda"), verbose=False)
print("[INIT] EasyOCR OK (en+ro, ch_sim+en)", flush=True)

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
    for (bbox, text, conf) in OCR.readtext(rgb, detail=1):
        if conf >= OCR_CONF and str(text).strip():
            hits.append(bbox)
    for (bbox, text, conf) in OCR_ZH.readtext(rgb, detail=1):
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

    n_capped = 0
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
        cv2.imwrite(os.path.join(png_dir, f"{fidx:06d}.png"), mask,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if n_capped:
        print(f"[MASK] {n_capped} frame-uri plafonate la {MASK_MAX_COVERAGE:.0%} (box-urile cele mai mari scoase)", flush=True)

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

    def _priori(width, height):
        # resize_ratio=1.0 + width/height explicite → dezactivăm downscale-ul
        # intern nedeterminist al DiffuEraser (default 0.6, ×0.5 peste 960px)
        PROPAINTER.forward(
            video_path, mask_path, priori_path,
            resize_ratio=1.0, width=width, height=height,
            video_length=video_length,
            ref_stride=10, neighbor_length=10, subvideo_length=50,
            mask_dilation=8,
        )

    oom = False
    try:
        _priori(pw, ph)
    except torch.cuda.OutOfMemoryError:
        # NU reîncercăm aici: cât timp suntem în except, traceback-ul activ ține
        # referințe la tensorii din ProPainter → empty_cache() nu poate elibera
        # VRAM-ul și retry-ul murea tot cu OOM ("22.5 GiB in use" la reîncercare)
        oom = True
    if oom:
        gc.collect()
        torch.cuda.empty_cache()
        pw, ph = max(64, int(pw * 0.6)) // 2 * 2, max(64, int(ph * 0.6)) // 2 * 2
        print(f"[INPAINT] CUDA OOM → reîncerc la {pw}x{ph}", flush=True)
        _priori(pw, ph)
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
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
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
