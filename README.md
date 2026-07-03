# auto-eraser-worker

<!-- build: v2.0 — greutăți baked în imagine + torch 2.7/cu128 (Blackwell) -->

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
   - Container Disk: 60 GB (imaginea are ~35GB — greutățile sunt BAKED în ea)
   - Network Volume: NU mai e nevoie (greutățile vin în imagine; dacă volumul
     și `WEIGHTS_DIR` rămân setate, handler-ul le ignoră — poți să le scoți)
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
`MAX_SECONDS` (90) · `DETECT_INTERVAL` (0.5) · `FLORENCE_INTERVAL` (2.0) ·
`OCR_CONF` (0.25) · `STATIC_RATIO` (0.60) · `MAX_BOX_AREA_PCT` (0.25)
