# Deepfake Detector Architecture

## High-Level Flow

```mermaid
flowchart TD
    A[Client: Gradio UI / API Consumer] --> B[FastAPI App: web-app.py]
    B --> C{Endpoint Router}
    C --> D[/health, /live, /ready, /model-info, /capabilities, /eval/summary]
    C --> E[/detect]
    C --> F[/detect/batch]
    C --> G[/metrics]

    E --> H[Upload Validation]
    F --> H
    H --> I[Image/Video Preprocessing]
    I --> J[Model Inference]
    J --> K[Post-processing + Forensics]
    K --> L[JSON Response + Metadata]

    B --> M[In-memory Metrics Store]
    D --> M
    E --> M
    F --> M
    G --> M
```

## Core Components

- `web-app.py`:
  - Hosts both Gradio UI and FastAPI deployment endpoints.
  - Handles request validation, inference dispatch, and response shaping.
- `docs/eval_summary.json`:
  - Stores serialized evaluation metrics consumed by `/eval/summary`.
- `realeval.py`:
  - Supports generating evaluation metrics and exporting them as JSON with `--out`.
- `lightning_modules/detector.py`:
  - Defines `HybridEfficientNet` model for hybrid RGB+FFT inference path.
- `utils/diffusion_heuristics.py` and `utils/fft_utils.py`:
  - Provides forensic heuristics and FFT transformations.
- `tests/test_api_smoke.py`:
  - Verifies service contracts for health, model info, metrics, and required payload behavior.

## Reliability and Safety Design

- Input validation with extension and max-size controls.
- Structured error responses and status codes for invalid/unsupported requests.
- Request-level metadata (`request_id`, latency, hash) for traceability.
- Runtime observability via `/metrics`:
  - total requests, success/failure counts, endpoint-level average latency,
  - processed items and verdict distribution.

## Deployment Notes

- Single service process exposes both API and UI.
- Configurable via environment variables (`PORT`, `MAX_UPLOAD_MB`, `APP_VERSION`).
- Docker and Render deployment manifests are included for reproducible hosting.
