# auto-eraser-worker

<!-- build: v1.1 — fix PCM LoRA path (symlink weights/ -> WEIGHTS_DIR) -->

Worker RunPod Serverless: șterge **automat** captions, logo-uri și watermark-uri din video.
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
   - GPU: 24 GB (L4 / RTX 4090 / A5000); pt 1080p full: L40S 48GB
   - Container Disk: 30 GB
   - Network Volume: 40 GB, montat (aici se descarcă automat ~15GB de greutăți la primul start)
   - Env: `WEIGHTS_DIR=/runpod-volume/weights`
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
