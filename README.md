# auto-eraser-worker

<!-- build: v3.0 — greutăți pe Network Volume (imagine mică, release-uri rapide) + torch 2.7/cu128 (Blackwell) -->

Worker RunPod Serverless: ștderge **automat** captions, logo-uri și watermark-uri din video.
Detecție: EasyOCR (text, pe keyframes) + Florence-2 (logo/watermark, open-vocabulary).
Inpainting: ProPainter (priori) + DiffuEraser (rafinare diffusion, consistent temporal).

**Un singur endpoint → oricâte aplicații.** Orice app îi trimite un job cu propriul
`callback_url`; workerul nu ține minte nimic între joburi. Nu clona/duplica endpointul
per aplicație.

## Deploy pe RunPod (build direct din GitHub)

1. RunPod → **Settings → Connections → GitHub** → autorizează acest repo.
2. **Serverless → New Endpoint → GitHub Repo** → alege repo-ul, branch `main`,
   Dockerfile în root.
3. Config endpoint:
   - GPU: 24 GB (RTX 4090 / 5090 / L4 / A5000); pt 1080p full: L40S 48GB.
     torch 2.7 + cu128 → merge inclusiv pe Blackwell/RTX 50xx.
   - Container Disk: 30 GB (imaginea ~12GB + temp la procesare)
   - Network Volume: **OBLIGATORIU** (~30GB, în același datacenter cu endpointul).
     Greutățile (~15GB) se descarcă pe el o singură dată la primul start
     (`WEIGHTS_DIR=/runpod-volume/weights` e setat din imagine); fără volum,
     fiecare cold start le-ar re-descărca.
   - Idle Timeout: 120s · Execution Timeout: 1800s

## API

### Input
```json
{
  "input": {
    "video_url":     "https://.../clip.mp4",
    "video_base64":  "(alternativ, <50MB)",
    "targets":       ["captions", "logos", "watermarks"],
    "extra_prompts": ["numele canalului"],
    "max_img_size":  960,
    "callback_url":  "https://app-ta.ro/api/receive-ai-result",
    "job_id":        "123"
  }
}
```
Toate câmpurile în afară de video sunt opționale. Fără `targets` → șterge tot.

### Output
```json
{ "result_uploaded": true, "size_mb": 12.3, "detections": { "static_boxes": 2, "dynamic_hits": 41, "keyframes": 60 } }
```
sau `{ "video_base64": "..." }` (fără callback) · sau `{ "nothing_detected": true }` · sau `{ "error": "..." }`

Cu `callback_url`: workerul face POST multipart (`video` + `job_id`) la URL-ul dat —
aplicația ta trebuie să aibă endpointul de recepție (vezi `/api/receive-ai-result` din
serverul AUTO Eraser).

### Env tuning (opțional, pe endpoint)

**Detecție:** `MAX_SECONDS` (90) · `DETECT_INTERVAL` (0.5) · `FLORENCE_INTERVAL` (2.0) ·
`OCR_CONF` (0.25) · `STATIC_RATIO` (0.60) · `MAX_BOX_AREA_PCT` (0.25) ·
`BOX_PAD` (6) · `MASK_MAX_COVERAGE` (0.25 — peste plafon se sacrifică întâi
ce nu e caption; glifele caption-urilor nu se taie niciodată)

**Text pe obiecte vs caption ars:** `DRIFT_STEP_PCT` (0.012 — cât se poate mișca
același text între keyframe-uri vecine) · `DRIFT_MAX_PCT` (0.04 — variația pe
verticală pe tot clipul; și pragul ca un cluster să fie ștanțat static) ·
`DRIFT_TRIM` (0.10) · `SAME_TEXT_TOL` (0.12)

**Bucăți OCR citite nesigur (păstrate dacă stau pe linie cu text sigur):**
`HORIZ_DEG` (10) · `LINE_GAP` (1.2) · `LINE_H_RATIO` (2.2) · `WEAK_TEMPORAL_KF` (2)

**Calitate inpainting:** `PROC_MAX_SIDE` (832) · `MIN_PROC_SIDE` (320) ·
`MASK_DILATE_PX` (6) · `PXFRAMES_PER_GB` (5.0e6) · `CHUNK_OVERLAP` (12) ·
`PROC_FRAME_BUDGET` (0 = derivat din VRAM) · `MAX_FPS` (30)

#### Cum se alege rezoluția de inpainting

Rezoluția **nu depinde de lungimea clipului**. Latura lungă țintește `PROC_MAX_SIDE`;
clipurile prea lungi ca să încapă în VRAM se taie în bucăți temporale (cu
`CHUNK_OVERLAP` cadre de context, aruncate la lipire), fiecare procesată la
rezoluție plină.

Bugetul de cadre per bucată se derivă din VRAM-ul real al plăcii
(`VRAM_GB × PXFRAMES_PER_GB / (lățime × înălțime)`), nu dintr-o constantă.
Calibrarea: ~138M px·cadre erau dovediți siguri pe 24GB → ~5.75M px·cadre/GB,
din care luăm 5.0M ca marjă.

> Varianta veche scădea rezoluția cu `sqrt(600/cadre)` pe tot videoul. Un clip de
> 78s la 720x1280 ajungea reconstruit la **182x324** și ridicat înapoi cu lanczos
> de 3.95x (= blur vizibil), iar `mask_dilation=8` aplicat la acea rezoluție
> ștergea efectiv **32px** la rezoluția reală. Acum același clip se procesează la
> 468x832 în bucăți, iar dilatarea e exprimată în pixeli la rezoluția originală
> (`MASK_DILATE_PX`) și convertită la scara de procesare.

`MASK_DILATE_PX` e cât dilată inpainting-ul masca, **în pixeli la rezoluția
originală** — crește-l dacă rămân margini de litere, scade-l dacă șterge prea lat.
Compozitarea din `finalize()` mai adaugă doar ~2px + feather (înainte ~9px, care
se adunau peste dilatarea ProPainter).
